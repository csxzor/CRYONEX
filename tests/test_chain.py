"""Host kill-chain layer: per-host features, episodes/targets, causal tracker, DARPA labels."""

import numpy as np
import polars as pl

from kcwm.chain.hosts import HOST_FEATURES, host_windows
from kcwm.chain.targets import add_targets, anchor_grid, episodes, timeline_ends
from kcwm.chain.tracker import CURRENT, anchor_features, prior_next
from kcwm.stages import N_STAGES, STAGE_INDEX

CIDRS = ["10.0.0.0/8"]


def _flow(ts, src, dst, dport, stage="Benign", **kw):
    base = {"ts": float(ts), "te": float(ts) + 1, "src_ip": src, "dst_ip": dst, "src_port": 40000, "dst_port": dport,
            "protocol": 6, "duration": 1.0, "fwd_bytes": 100.0, "bwd_bytes": 50.0, "fwd_packets": 3.0,
            "bwd_packets": 2.0, "syn_count": 1.0, "ack_count": 1.0, "rst_count": 0.0, "stage": stage, "attempted": False}
    base.update(kw)
    return base


def test_host_windows_roles_and_labels():
    f = pl.DataFrame([
        _flow(0, "8.8.8.8", "10.0.0.5", 22, "Reconnaissance"),       # external scan of .5
        _flow(1, "8.8.8.8", "10.0.0.5", 23, "Reconnaissance", rst_count=1.0),
        _flow(15, "10.0.0.5", "10.0.0.9", 445, "LateralMovement"),   # .5 moves to .9
    ])
    hw = host_windows(f, CIDRS)
    assert set(HOST_FEATURES) <= set(hw.columns)
    r5 = hw.filter((pl.col("host") == "10.0.0.5") & (pl.col("window") == 0)).row(0, named=True)
    assert r5["i_n"] == 2 and r5["i_ports"] == 2 and r5["i_fail"] == 1 and r5["stage_now"] == STAGE_INDEX["Reconnaissance"]
    r9 = hw.filter(pl.col("host") == "10.0.0.9").row(0, named=True)
    assert r9["i_int_hosts"] == 1 and r9["stage_now"] == STAGE_INDEX["LateralMovement"]
    assert hw.filter(pl.col("host") == "8.8.8.8").height == 0  # external hosts are not tracked


def _toy_hw():
    rows = []
    # one host: recon at windows 0-5, initial access at 1000-1005 (a chain step), then quiet to 3000
    for w in range(0, 3001, 5):
        st = 1 if w <= 5 else 2 if 1000 <= w <= 1005 else 0
        rows.append({"timeline": "t", "host": "h", "window": w, "stage_now": st, **{c: 1.0 for c in CURRENT}})
    return pl.DataFrame(rows).with_columns(pl.col("stage_now").cast(pl.Int8))


def test_episodes_and_targets():
    hw = _toy_hw()
    ep = episodes(hw, merge_gap=360)
    assert ep.select(["stage", "start", "transition", "has_history"]).rows() == [(1, 0, True, False), (2, 1000, True, True)]
    anchors = add_targets(anchor_grid(hw, pl.DataFrame({"timeline": ["t"], "host": ["h"]}), stride=30), ep,
                          timeline_ends(hw))
    a = anchors.filter(pl.col("window") < 1000)
    assert (a.filter(pl.col("window") > 0)["next_stage"] == 2).all()  # next transition: initial access
    near = anchors.filter(pl.col("window").is_between(990, 999))
    assert near["y_5m"].all()
    late = anchors.filter(pl.col("window") > 2700)
    assert (~late["v_1h"]).all()  # recording ends within the hour: masked, not negative


def test_tracker_is_causal():
    """Changing the future (after an anchor) must not change the anchor's features."""
    hw = _toy_hw()
    probs = np.eye(N_STAGES)[hw["stage_now"].to_numpy()]
    anchors = pl.DataFrame({"timeline": ["t"], "host": ["h"], "window": [500]})
    X1, names, _ = anchor_features(anchors, hw, probs)
    probs2 = probs.copy()
    future = hw["window"].to_numpy() > 500
    probs2[future] = np.eye(N_STAGES)[6]
    X2, _, _ = anchor_features(anchors, hw, probs2)
    np.testing.assert_allclose(X1, X2)
    assert X1[0, names.index("ev1")] == 1.0 and X1[0, names.index("ev2")] == 0.0


def test_prior_next_excludes_current_and_benign():
    p = prior_next(np.array([1, 2]), np.array([1, 2]))
    assert p.shape == (2, N_STAGES - 1)
    np.testing.assert_allclose(p.sum(1), 1.0)
    assert p[0, 0] == 0 and p[1, 1] == 0  # the current stage is not a "next" stage
    assert p[0].argmax() + 1 == STAGE_INDEX["InitialAccess"]  # recon -> initial access


def test_darpa_phase_list_and_labels():
    from kcwm.ingest.darpa import label_flows

    sessions = pl.DataFrame({"phase": [3, 4], "t0": [100.0, 200.0], "t1": [101.0, 201.0], "sport": [None, None],
                             "dport": [111, 23], "a": ["202.77.162.213", "172.16.115.20"],
                             "b": ["172.16.115.20", "172.16.112.50"]},
                            schema_overrides={"sport": pl.Int32, "dport": pl.Int32, "phase": pl.Int8})
    f = pl.DataFrame([
        _flow(100.5, "202.77.162.213", "172.16.115.20", 111),
        _flow(200.2, "172.16.115.20", "172.16.112.50", 23),
        _flow(200.2, "172.16.115.20", "172.16.112.50", 80),   # port mismatch: benign
    ])
    out = label_flows(f.drop("stage"), sessions, {3: "InitialAccess", 4: "LateralMovement"})
    assert out["stage"].to_list() == ["InitialAccess", "LateralMovement", "Benign"]


def test_nowcast_features_are_causal():
    """Rolling host features at a window use only that host's rows at or before it."""
    from kcwm.chain.hosts import feature_matrix

    rows = [_flow(w * 10.0, "8.8.8.8", "10.0.0.5", 22, rst_count=1.0) for w in range(20)]
    base = host_windows(pl.DataFrame(rows), CIDRS)
    more = host_windows(pl.DataFrame(rows + [_flow(w * 10.0, "8.8.4.4", "10.0.0.5", 22) for w in range(10, 20)]), CIDRS)
    X1, X2 = feature_matrix(base), feature_matrix(more)
    np.testing.assert_allclose(X1[:10], X2[:10])      # the extra flows start at window 10
    assert not np.allclose(X1[10:], X2[10:])

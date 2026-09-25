"""State features: causality (no future information) and known attack signatures."""

import numpy as np
import polars as pl

from kcwm.features.build import build_capture
from kcwm.features.registry import FEATURE_NAMES, N_FEATURES, feature_mask_index

from .conftest import background, make_flows

CIDRS = ["192.168.0.0/16"]
T0 = 1_500_000_000.0  # on the 10-s grid


def _build(flows, cfg):
    _, win = build_capture(flows, dataset="synthetic", capture="syn", cfg=cfg, internal_cidrs=CIDRS)
    return win.sort("window")


def test_registry_is_consistent():
    assert len(set(FEATURE_NAMES)) == N_FEATURES
    idx = feature_mask_index()
    assert len(idx) == N_FEATURES and set(idx) <= {-1, 0, 1, 2}


def test_features_at_t_ignore_flows_after_t(rng, cfg):
    """Deleting everything after a cut must not change any feature at or before the cut."""
    rows = background(T0, T0 + 1200, rng)
    # a beaconing host and an internal scanner, both spanning the cut
    rows += [{"ts": T0 + 30.0 * i, "src_ip": "192.168.1.66", "dst_ip": "9.9.9.9", "dst_port": 8080,
              "duration": 0.2} for i in range(40)]
    rows += [{"ts": T0 + 500 + 0.5 * i, "src_ip": "192.168.1.77", "dst_ip": f"192.168.1.{100 + i % 50}",
              "dst_port": 445, "syn_count": 1.0, "ack_count": 0.0, "bwd_bytes": 0.0, "bwd_packets": 0.0}
             for i in range(600)]
    full = make_flows(rows)
    cut = T0 + 700.0
    win_full = _build(full, cfg)
    win_cut = _build(full.filter(pl.col("ts") < cut), cfg)
    last = int(cut // 10) - 1
    a = win_full.filter(pl.col("window") <= last).select(FEATURE_NAMES).to_numpy()
    b = win_cut.filter(pl.col("window") <= last).select(FEATURE_NAMES).to_numpy()
    assert a.shape == b.shape
    np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-9)


def test_beacon_and_scan_signatures(rng, cfg):
    rows = background(T0, T0 + 900, rng, rate=1.0)
    rows += [{"ts": T0 + 20.0 * i, "src_ip": "192.168.1.66", "dst_ip": "9.9.9.9", "dst_port": 8080,
              "duration": 0.2} for i in range(45)]
    # vertical sequential scan 1..200 at t = 600 s
    rows += [{"ts": T0 + 600 + 0.02 * p, "src_ip": "10.1.1.1", "dst_ip": "192.168.1.10", "dst_port": p,
              "syn_count": 1.0, "ack_count": 0.0, "bwd_bytes": 0.0, "bwd_packets": 0.0} for p in range(1, 201)]
    win = _build(make_flows(rows), cfg)
    late = win.filter(pl.col("window") >= int((T0 + 200) // 10))
    assert late["beacon_max"].max() > 0.9
    scan_w = win.filter(pl.col("window") == int((T0 + 600) // 10))
    assert scan_w["seq_port_score"].item() > 0.95
    assert scan_w["half_open_max"].item() >= 200
    assert win["beacon_max"].head(3).max() == 0.0  # too little history at the start


def test_lateral_and_exfil_signatures(rng, cfg):
    rows = background(T0, T0 + 300, rng, rate=1.0)
    rows += [{"ts": T0 + 100 + 0.1 * i, "src_ip": "192.168.1.77", "dst_ip": f"192.168.1.{100 + i}",
              "dst_port": 445} for i in range(60)]
    rows += [{"ts": T0 + 200, "src_ip": "192.168.1.12", "dst_ip": "7.7.7.7", "dst_port": 443,
              "duration": 90.0, "fwd_bytes": 5e8, "bwd_bytes": 1e4}]
    win = _build(make_flows(rows), cfg)
    lat = win.filter(pl.col("window") == int((T0 + 100) // 10))
    assert lat["int_fanout_max"].item() >= 60 and lat["int_scan_max"].item() >= 60
    ex = win.filter(pl.col("window") == int((T0 + 200) // 10))
    assert ex["out_asym_max"].item() > 0.99 and ex["long_out_bytes"].item() >= 5e8


def test_idle_windows_are_real_zero_states_and_masks(rng, cfg):
    rows = background(T0, T0 + 100, rng) + background(T0 + 200, T0 + 300, rng)  # 100 s lull
    win = _build(make_flows(rows), cfg)
    assert win["session"].n_unique() == 1  # a lull shorter than the session gap
    lull = win.filter((pl.col("window") > int((T0 + 110) // 10)) & (pl.col("window") < int((T0 + 190) // 10)))
    assert lull.height > 0 and (lull["flow_count"] == 0).all()
    assert (win["m_packet"] == 0).all()  # CSV-like input: no PCAP headers
    assert (win["m_ip"] == 1).all()
    assert win["warmup"].head(cfg["window"]["warmup_windows"]).all()

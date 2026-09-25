"""Purged splits and normalisation: the properties a hostile reviewer checks first.

Adapted from netforecast v1 ``tests/test_purged_split.py`` to the v2 windows table.
"""

import numpy as np
import polars as pl

from kcwm.data.splits import assert_purged, cross_split, lofo_split, p1_split
from kcwm.features.normalize import QuantileBinner, RobustScaler, transform_windows

H, WARM, CTX = 12, 40, 32


def _table(lengths=(400, 250, 330), families=None):
    rows = []
    for s, n in enumerate(lengths):
        for p in range(n):
            rows.append({"session_key": f"cap#{s}", "pos_in_session": p,
                         "dataset": "a" if s < 2 else "b",
                         "family_now": (families or {}).get((s, p))})
    return pl.DataFrame(rows, schema={"session_key": pl.Utf8, "pos_in_session": pl.Int64,
                                      "dataset": pl.Utf8, "family_now": pl.Utf8})


def test_p1_is_purged_time_ordered_and_skips_warmup():
    win = _table()
    sp = p1_split(win, horizon=H, warmup=WARM)
    assert_purged(sp, win, H)
    pos = win["pos_in_session"].to_numpy()
    for rows in (sp.train, sp.calibration, sp.test):
        assert (pos[rows] >= WARM).all()
    sid = win["session_key"].to_numpy()
    for s in np.unique(sid):
        def span(r):
            return pos[r[sid[r] == s]]
        assert span(sp.train).max() < span(sp.calibration).min() < span(sp.test).min()


def test_purge_bites_and_scaler_rows_are_training_span_only():
    win = _table()
    sp = p1_split(win, horizon=H, warmup=WARM)
    eligible = int((win["pos_in_session"] >= WARM).sum())
    kept = sp.train.size + sp.calibration.size + sp.test.size
    assert 0 < eligible - kept <= 2 * H * 3
    pos, sid = win["pos_in_session"].to_numpy(), win["session_key"].to_numpy()
    for s, n in zip(np.unique(sid), (400, 250, 330)):
        mine = pos[sp.fit_rows[sid[sp.fit_rows] == s]]
        assert mine.max() == int(n * 0.6) - 1


def test_lofo_removes_family_from_training_context_and_horizon():
    fam = {(0, p): "DoS" for p in range(150, 160)}
    win = _table(families=fam)
    sp = lofo_split(win, "DoS", horizon=H, warmup=WARM, context=CTX)
    pos, sid = win["pos_in_session"].to_numpy(), win["session_key"].to_numpy()
    for r in np.concatenate([sp.train, sp.calibration]):
        if sid[r] == "cap#0":
            assert not (pos[r] - CTX + 1 <= 159 and pos[r] + H >= 150)
    # every anchor that sees the family is in test
    touching = [r for r in range(win.height) if sid[r] == "cap#0" and pos[r] >= WARM
                and pos[r] - CTX + 1 <= 159 and pos[r] + H >= 150]
    assert set(touching) <= set(sp.test.tolist())


def test_cross_split_never_trains_on_target_dataset():
    win = _table()
    sp = cross_split(win, train_datasets=["a"], test_dataset="b", horizon=H, warmup=WARM)
    ds = win["dataset"].to_numpy()
    assert (ds[sp.train] == "a").all() and (ds[sp.calibration] == "a").all()
    assert (ds[sp.test] == "b").all() and (ds[sp.fit_rows] == "a").all()


def test_warmup_recentering_removes_a_level_shift(rng):
    n = 600
    base = rng.lognormal(3.0, 0.5, size=(n, 1))
    x = np.concatenate([base, base * 30.0])  # same traffic, second network 30x louder
    session = np.array([0] * n + [1] * n)
    warm = np.zeros(2 * n, dtype=bool)
    warm[:90] = warm[n:n + 90] = True
    sc = RobustScaler.fit(x[:n], names=["flow_count"])  # a log-scaled feature
    g = transform_windows(sc, x, mode="global")
    w = transform_windows(sc, x, mode="warmup", session=session, warmup=warm)
    assert abs(np.median(g[n:]) - np.median(g[:n])) > 2.0      # shift survives global scaling
    assert abs(np.median(w[n:]) - np.median(w[:n])) < 0.25     # removed by warm-up centring


def test_binner_round_trip_and_zero_inflation(rng):
    z = np.concatenate([np.zeros(700), rng.normal(size=300)])[:, None]
    b = QuantileBinner.fit(z, n_bins=16)
    idx = b.transform(z)
    assert idx.min() >= 0 and idx.max() < b.n_bins_per_feature()[0]
    # the zero mass collapses into one bin instead of ten identical ones
    assert b.n_bins_per_feature()[0] < 16
    assert len(np.unique(idx[:700])) == 1

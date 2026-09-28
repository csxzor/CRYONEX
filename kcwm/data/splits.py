"""Evaluation splits over the windows table. Temporal and purged; never random.

Ported from netforecast v1 ``ingest/pipeline.py`` (``purged_temporal_split`` and
``per_session_train_rows``, with their tests). In v2 an *anchor* is a row of the windows
table: the last observed window of a context. Its forecast targets lie in the ``K``
windows after it.

Rules every protocol obeys:

* **Anchors never sit in a session's warm-up** (the first ``warmup_windows`` windows). They
  are reserved for label-free normalisation, and they also guarantee a full context.
* **Purging:** a training anchor whose horizon reaches the calibration span is dropped, and
  a calibration anchor whose horizon reaches the test span is dropped. Context reaching
  *back* across a boundary is kept, since at inference that history is genuinely observed.
* **Normalisation statistics** come only from ``fit_rows`` (the training span).

Protocols: ``p1`` per-session temporal (in-distribution); ``lofo`` leave-one-family-out;
``cross`` train on some datasets, test on another.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl


@dataclass
class Split:
    name: str
    train: np.ndarray
    calibration: np.ndarray
    test: np.ndarray
    fit_rows: np.ndarray  # rows a scaler / binner may be fit on
    notes: dict = field(default_factory=dict)

    def sizes(self) -> dict[str, int]:
        return {"train": int(self.train.size), "calibration": int(self.calibration.size),
                "test": int(self.test.size), "fit_rows": int(self.fit_rows.size)}


def _session_layout(win: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(session id per row, position within session) for a table sorted by capture, window."""
    keys = win["session_key"].to_numpy()
    _, sid = np.unique(keys, return_inverse=True)
    pos = win["pos_in_session"].to_numpy().astype(np.int64)
    return sid, pos


def p1_split(
    win: pl.DataFrame,
    *,
    fractions: tuple[float, float, float] = (0.60, 0.15, 0.25),
    horizon: int,
    warmup: int,
    rows: np.ndarray | None = None,
) -> Split:
    """Per-session temporal split: each session cut 60/15/25 in window space, then purged."""
    sid, pos = _session_layout(win)
    n = win.height
    if rows is None:
        rows = np.arange(n)
    rows = np.asarray(rows)
    length = np.bincount(sid, minlength=sid.max() + 1)[sid]
    train_end = (length * fractions[0]).astype(np.int64)
    cal_end = train_end + (length * fractions[1]).astype(np.int64)

    p = pos[rows]
    ok = p >= warmup
    last_target = p + horizon
    in_train = ok & (p < train_end[rows]) & (last_target < train_end[rows])
    in_cal = ok & (p >= train_end[rows]) & (p < cal_end[rows]) & (last_target < cal_end[rows])
    in_test = ok & (p >= cal_end[rows])
    fit = np.flatnonzero(pos < train_end)
    return Split("p1", rows[in_train], rows[in_cal], rows[in_test], fit,
                 notes={"fractions": fractions, "horizon": horizon, "warmup": warmup})


def lofo_split(
    win: pl.DataFrame,
    family: str,
    *,
    horizon: int,
    warmup: int,
    context: int,
    fractions: tuple[float, float, float] = (0.60, 0.15, 0.25),
) -> Split:
    """Leave one attack family out of training entirely.

    Training and calibration keep the P1 spans but drop every anchor whose context or
    horizon touches a window of the held-out family. Test holds (a) every anchor whose
    horizon or current window contains the held-out family, from any span, and (b) P1 test
    anchors free of it, as negatives.
    """
    fam = win["family_now"].fill_null("").to_numpy()
    sid, pos = _session_layout(win)
    hit = (fam == family).astype(np.int64)
    # Touches the family within [t - context + 1, t + horizon], same session only.
    touches = np.zeros(win.height, dtype=bool)
    for s in np.unique(sid):
        idx = np.flatnonzero(sid == s)
        c = np.concatenate([[0], np.cumsum(hit[idx])])
        j = np.arange(idx.size)
        lo = np.clip(j - context + 1, 0, idx.size)
        hi = np.clip(j + horizon + 1, 0, idx.size)
        touches[idx] = (c[hi] - c[lo]) > 0
    base = p1_split(win, fractions=fractions, horizon=horizon, warmup=warmup)
    train = base.train[~touches[base.train]]
    cal = base.calibration[~touches[base.calibration]]
    eligible = np.flatnonzero((pos >= warmup) & touches)
    test = np.union1d(eligible, base.test[~touches[base.test]])
    fit = base.fit_rows[fam[base.fit_rows] != family]
    return Split(f"lofo-{family}", train, cal, test, fit,
                 notes={"family": family, "held_out_anchors": int(eligible.size)})


def cross_split(
    win: pl.DataFrame,
    *,
    train_datasets: list[str],
    test_dataset: str,
    horizon: int,
    warmup: int,
    fractions: tuple[float, float, float] = (0.60, 0.15, 0.25),
) -> Split:
    """Train and calibrate on some datasets; test on every scorable window of another."""
    ds = win["dataset"].to_numpy()
    _, pos = _session_layout(win)
    source_rows = np.flatnonzero(np.isin(ds, train_datasets))
    base = p1_split(win, fractions=fractions, horizon=horizon, warmup=warmup, rows=source_rows)
    test = np.flatnonzero((ds == test_dataset) & (pos >= warmup))
    return Split(f"cross-{test_dataset}", base.train, base.calibration, test,
                 base.fit_rows[np.isin(ds[base.fit_rows], train_datasets)],
                 notes={"train_datasets": train_datasets, "test_dataset": test_dataset})


def assert_purged(split: Split, win: pl.DataFrame, horizon: int) -> None:
    """Within every session, each training anchor's last target precedes the first
    calibration/test anchor, and each calibration anchor's last target precedes the first
    test anchor. (For P1, where splits are contiguous spans in time.)"""
    sid, pos = _session_layout(win)
    for s in np.unique(sid):
        def span(rows: np.ndarray) -> np.ndarray:
            return pos[rows[sid[rows] == s]]
        tr, ca, te = span(split.train), span(split.calibration), span(split.test)
        later = np.concatenate([ca, te])
        if tr.size and later.size:
            assert tr.max() + horizon < later.min(), f"session {s}: train horizon leaks"
        if ca.size and te.size:
            assert ca.max() + horizon < te.min(), f"session {s}: calibration horizon leaks"

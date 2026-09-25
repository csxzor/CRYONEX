"""Forecast targets.

``y_k[t] = 1`` if any compromise-stage window falls in ``(t, t + k]``. The current window is
excluded: a model that scores well by noticing an attack already in progress is doing the
static classification the problem statement asks us to move beyond.

**Horizons that run past the end of a session are masked, not negative.** v1 filled them
with "no attack" (``fill_null(False)``), which labels the unknown future as benign.
``valid_k[t]`` is False there, and every loss and metric drops those rows.
"""

from __future__ import annotations

import numpy as np

from ..stages import compromise_mask


def any_attack_targets(stage_now: np.ndarray, session: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """v1's target, for the reproduction check: any attack (recon included) in (t, t+k].

    v1 filled horizons past the session end with "no attack". Here they are masked, as for
    the v2 target.
    """
    att = np.asarray(stage_now) > 0
    session = np.asarray(session)
    y = np.zeros(att.size, dtype=bool)
    valid = np.zeros(att.size, dtype=bool)
    for sid in np.unique(session):
        idx = np.flatnonzero(session == sid)
        a = att[idx].astype(np.int64)
        c = np.concatenate([[0], np.cumsum(a)])
        j = np.arange(idx.size)
        hi = np.minimum(j + k, idx.size - 1)
        y[idx] = (c[hi + 1] - c[j + 1]) > 0
        valid[idx] = (idx.size - 1 - j) >= k
    return y & valid, valid


def forecast_targets(
    stage_now: np.ndarray, session: np.ndarray, horizons: list[int]
) -> dict[str, np.ndarray]:
    """``y_{k}`` and ``valid_{k}`` for each horizon, plus time-to-compromise and next stage.

    Inputs are aligned per window and sorted by (session, window) with contiguous windows
    inside each session.
    """
    stage_now = np.asarray(stage_now, dtype=np.int64)
    session = np.asarray(session)
    comp = compromise_mask()[stage_now]
    n = stage_now.size
    out: dict[str, np.ndarray] = {}

    # Position of each window inside its session, and distance to the session's end.
    pos = np.zeros(n, dtype=np.int64)
    remaining = np.zeros(n, dtype=np.int64)
    for sid in np.unique(session):
        idx = np.flatnonzero(session == sid)
        pos[idx] = np.arange(idx.size)
        remaining[idx] = idx.size - 1 - np.arange(idx.size)

    # Next compromise window strictly after t, within the session (distance in windows).
    time_to_comp = np.full(n, -1, dtype=np.int64)
    next_stage = np.zeros(n, dtype=np.int64)  # first attack stage entered after t, 0 if none
    for sid in np.unique(session):
        idx = np.flatnonzero(session == sid)
        nearest = -1
        nearest_attack_stage = 0
        for j in range(idx.size - 1, -1, -1):
            i = idx[j]
            time_to_comp[i] = (nearest - j) if nearest >= 0 else -1
            next_stage[i] = nearest_attack_stage
            if comp[i]:
                nearest = j
            if stage_now[i] > 0:
                nearest_attack_stage = int(stage_now[i])

    for k in horizons:
        valid = remaining >= k
        y = (time_to_comp >= 1) & (time_to_comp <= k)
        out[f"y_{k}"] = y & valid
        out[f"valid_{k}"] = valid
    out["time_to_compromise"] = time_to_comp
    out["next_attack_stage"] = next_stage
    out["pos_in_session"] = pos
    return out

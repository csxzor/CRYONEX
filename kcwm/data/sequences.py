"""Model-ready arrays from the windows table.

The windows table is sorted by (capture, window) and every session is a contiguous run of
rows, so a window's history is simply the rows before it. An anchor's context is rows
``[a - N + 1, a]``. Anchors live outside the warm-up, which is at least ``N`` windows, so
the context never leaves the session. Its future is rows ``a + 1 .. a + K``, valid while
still inside the session.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from ..features.normalize import QuantileBinner, RobustScaler, transform_windows
from ..features.registry import MASK_NAMES


@dataclass
class Arrays:
    """Scaled per-row arrays shared by every model."""

    x: np.ndarray            # (rows, F) float32 scaled features
    m: np.ndarray            # (rows, 3) float32 group masks
    stage: np.ndarray        # (rows,) int64 current stage
    session: np.ndarray      # (rows,) int64 session id
    pos: np.ndarray          # (rows,) int64 position in session
    remaining: np.ndarray    # (rows,) int64 windows left in session after this one
    y: dict[int, np.ndarray]      # horizon -> (rows,) bool
    valid: dict[int, np.ndarray]  # horizon -> (rows,) bool
    bins: np.ndarray | None = None  # (rows, F) int64 quantile bin of x (world-model targets)
    names: list[str] | None = None  # feature order of x (defaults to the registry's)

    @property
    def n_features(self) -> int:
        return self.x.shape[1]


def build_arrays(
    win: pl.DataFrame,
    scaler: RobustScaler,
    *,
    mode: str,
    horizons: list[int],
    binner: QuantileBinner | None = None,
) -> Arrays:
    names = list(scaler.names)
    raw = win.select(names).to_numpy().astype(np.float64)
    _, session = np.unique(win["session_key"].to_numpy(), return_inverse=True)
    warm = win["warmup"].to_numpy().astype(bool)
    x = transform_windows(scaler, raw, mode=mode, session=session, warmup=warm)
    m = win.select(MASK_NAMES).to_numpy().astype(np.float32)
    pos = win["pos_in_session"].to_numpy().astype(np.int64)
    length = np.bincount(session)[session]
    return Arrays(
        x=x, m=m,
        stage=win["stage_now"].to_numpy().astype(np.int64),
        session=session.astype(np.int64), pos=pos, remaining=length - 1 - pos,
        y={k: win[f"y_{k}"].to_numpy().astype(bool) for k in horizons},
        valid={k: win[f"valid_{k}"].to_numpy().astype(bool) for k in horizons},
        bins=binner.transform(x) if binner is not None else None, names=names,
    )


def context_index(anchors: np.ndarray, length: int) -> np.ndarray:
    """(n, length) row indices of each anchor's context, oldest first."""
    anchors = np.asarray(anchors, dtype=np.int64)
    return anchors[:, None] + np.arange(-length + 1, 1)[None, :]


def future_index(anchors: np.ndarray, horizon: int, n_rows: int) -> np.ndarray:
    """(n, horizon) row indices of each anchor's future, clipped to the table."""
    idx = np.asarray(anchors, dtype=np.int64)[:, None] + np.arange(1, horizon + 1)[None, :]
    return np.clip(idx, 0, n_rows - 1)


def model_input(arr: Arrays, rows: np.ndarray) -> np.ndarray:
    """Scaled features with masked groups zeroed, plus the three mask channels."""
    from ..features.registry import feature_mask_index

    gate = np.asarray(feature_mask_index(arr.names))
    x = arr.x[rows].copy()
    m = arr.m[rows]
    for g in range(m.shape[-1]):
        cols = gate == g
        x[..., cols] *= m[..., g : g + 1]
    return np.concatenate([x, m], axis=-1).astype(np.float32)

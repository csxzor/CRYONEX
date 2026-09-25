"""Time windowing and capture sessions.

A flow table becomes an ordered sequence of network states: each flow is assigned to a
fixed-width window by its **start** time, and each window is one observation ``x_t``.

Ported from netforecast v1 ``ingest/windowing.py``, with two changes:

* **Absolute window grid.** A window index is ``floor(ts / seconds)`` on the epoch grid, not
  an offset from the capture's first flow. Every source that describes the same capture
  therefore agrees on window numbers by construction: the CSV flows, the PCAP-derived packet
  block and v1's cached packet aggregates.
* **Sessions are computed per capture.** A capture (one CSV day, one CTU-13 scenario) is split
  wherever traffic stops for longer than ``session_gap_seconds``. Inside a session the window
  spine is contiguous: an idle window is a real, all-quiet state and is kept. Sequences never
  cross a session boundary.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def assign_windows(df: pl.DataFrame, seconds: float) -> pl.DataFrame:
    """Add an absolute ``window`` index (``floor(ts / seconds)``) to every flow."""
    return df.with_columns((pl.col("ts") / seconds).floor().cast(pl.Int64).alias("window"))


def session_spine(occupied: np.ndarray, gap_windows: int) -> pl.DataFrame:
    """Contiguous window spine per session from the set of occupied windows.

    A session boundary falls wherever consecutive occupied windows are more than
    ``gap_windows`` apart. Returns ``window`` (Int64) and ``session`` (Int32, 0-based).
    """
    occupied = np.unique(np.asarray(occupied, dtype=np.int64))
    if occupied.size == 0:
        return pl.DataFrame(schema={"window": pl.Int64, "session": pl.Int32})
    breaks = np.flatnonzero(np.diff(occupied) > gap_windows)
    starts = np.concatenate([[occupied[0]], occupied[breaks + 1]])
    ends = np.concatenate([occupied[breaks], [occupied[-1]]])
    windows = np.concatenate([np.arange(s, e + 1) for s, e in zip(starts, ends)])
    sessions = np.concatenate(
        [np.full(e - s + 1, i, dtype=np.int32) for i, (s, e) in enumerate(zip(starts, ends))]
    )
    return pl.DataFrame({"window": windows, "session": sessions})


def attach_sessions(flows: pl.DataFrame, spine: pl.DataFrame) -> pl.DataFrame:
    """Join each flow to its session through its window."""
    return flows.join(spine, on="window", how="left")

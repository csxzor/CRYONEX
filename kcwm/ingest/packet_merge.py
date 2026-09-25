"""Packet-level block from raw PCAPs, merged into CSV-derived windows.

CIC-IDS2017's labelled flows come from the corrected CSVs. Their packet-level detail (TTL,
TCP window, fragments, retransmissions, payload sizes) comes from the raw captures.
Matching individual packets to individual CSV flows is fragile, because the two tools
disagree about where a flow begins and ends. v1 therefore joined at **window** granularity:
it aggregated each PCAP's NFStream flows into the same 10-s windows and merged the per-window
packet block. That covered 98.8% of windows.

v2 keeps the approach. Because windows now sit on the absolute epoch grid, the join is by
window number with no origin bookkeeping. Two sources produce the block:

* ``from_pcap``: NFStream over a raw capture (``ingest.pcap``). Reproducible from raw data,
  and slow (hours for the 50 GB week).
* ``from_v1_cache``: v1's already-merged block in
  ``<data_root>/processed/cicids2017/states_raw.parquet``, re-indexed from v1's
  origin-relative windows to the absolute grid. Fast, and it matches v1 exactly. It is the
  default for development, and ``from_pcap`` is the reproducibility check.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from ..features.registry import PACKET_FEATURES


def from_v1_cache(processed_v1: Path, seconds: float) -> pl.DataFrame:
    """v1's per-window packet block for CIC-IDS2017, on the absolute window grid."""
    meta = json.loads((processed_v1 / "metadata.json").read_text())
    if float(meta["window_seconds"]) != float(seconds):
        raise ValueError(f"v1 cache uses {meta['window_seconds']} s windows, need {seconds}")
    origin = float(meta["origin_epoch"])
    if origin % seconds:
        raise ValueError("v1 origin is not on the window grid; cannot re-index")
    offset = int(origin // seconds)
    states = pl.read_parquet(processed_v1 / "states_raw.parquet")
    return (
        states.filter(pl.col("packet_level_valid") > 0)
        .select(
            (pl.col("window") + offset).alias("window"),
            *[pl.col(c) for c in PACKET_FEATURES],
            pl.lit(1.0).alias("m_packet"),
        )
    )


def from_pcap(pcap_path: str | Path, seconds: float) -> pl.DataFrame:
    """Aggregate one raw capture's NFStream flows into the per-window packet block."""
    from ..features.vector import base_features
    from ..ingest.windowing import assign_windows
    from .pcap import read_pcap

    flows = assign_windows(read_pcap(pcap_path), seconds)
    block = base_features(flows, seconds)
    return block.filter(pl.col("m_packet") > 0).select("window", *PACKET_FEATURES, "m_packet")


def merge(windows: pl.DataFrame, block: pl.DataFrame) -> pl.DataFrame:
    """Overwrite the packet block (and ``m_packet``) wherever ``block`` has a real measurement."""
    joined = windows.join(block, on="window", how="left", suffix="_pcap")
    updates = [
        pl.coalesce([pl.col(f"{c}_pcap"), pl.col(c)]).alias(c)
        for c in (*PACKET_FEATURES, "m_packet")
        if f"{c}_pcap" in joined.columns
    ]
    return joined.with_columns(updates).select(windows.columns)

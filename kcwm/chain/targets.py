"""Per-host kill-chain labels: stage episodes, chain steps and forecast targets.

* **Timeline**: the span over which a host's history is remembered. One network recorded over
  several days (CIC-IDS2017's week, DAPT2020's week, CIC-IDS2018's days) is one timeline;
  every CTU-13 scenario and each DARPA 2000 scenario is its own experiment and timeline.
* **Episode**: a run of one stage on one host, merged across gaps up to ``merge_gap``
  windows (default 1 hour). Its first window is an *onset*.
* **Transition**: an onset whose stage differs from the stage the host was most recently
  doing (its latest-active earlier episode). A
  transition after an earlier episode is a **chain step** (e.g. Recon -> Initial Access on
  the same host); one with no earlier episode is a **first step**.
* **Anchors**: a forecast is issued for every tracked host every ``stride`` windows. Targets:
  ``next_stage`` (the stage of the host's next transition, whenever it comes; -1 if none is
  recorded) and ``y_<H>`` (a transition into a compromise stage within H windows; masked
  when the recording ends first).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ..stages import compromise_mask

HORIZONS = {"5m": 30, "1h": 360, "24h": 8640}
# Datasets whose captures are separate experiments (not one network's continuing recording):
# every capture is its own timeline, so no memory carries from one run into the next.
PER_CAPTURE = ("ctu13", "darpa2000")


def timeline_key(dataset: str, capture: str) -> str:
    return capture if dataset in PER_CAPTURE else dataset


def merge_timeline(hw: pl.DataFrame) -> pl.DataFrame:
    """One row per (timeline, host, window); sensors of one network (DAPT's public and
    private interfaces) are merged."""
    from .hosts import merge_host_tables

    hw = hw.with_columns(pl.when(pl.col("dataset").is_in(PER_CAPTURE)).then(pl.col("capture"))
                         .otherwise(pl.col("dataset")).alias("timeline"))
    parts = []
    for (tl, ds), g in hw.group_by(["timeline", "dataset"], maintain_order=True):
        if g.select(["host", "window"]).is_duplicated().any():
            # the capture a window belongs to (the busiest sensor wins), for grouping folds
            cap_of = g.group_by(["host", "window"]).agg(pl.col("capture").sort_by("o_n", descending=True).first())
            caps = g["capture"].unique().to_list()
            m = merge_host_tables([g.filter(pl.col("capture") == c).drop(["dataset", "capture", "timeline"])
                                   for c in caps])
            g = m.join(cap_of, on=["host", "window"], how="left").with_columns(
                pl.lit(tl).alias("timeline"), pl.lit(ds).alias("dataset"))
        parts.append(g)
    return pl.concat(parts, how="diagonal_relaxed").sort(["timeline", "host", "window"])


def episodes(hw: pl.DataFrame, *, merge_gap: int = 360, label: str = "stage_now") -> pl.DataFrame:
    """(timeline, host, stage, start, end) for every stage run, merged across gaps."""
    a = hw.filter(pl.col(label) > 0).select(["timeline", "host", "window", pl.col(label).alias("stage")])
    a = a.sort(["timeline", "host", "stage", "window"]).with_columns(
        (pl.col("window").diff().over(["timeline", "host", "stage"]).fill_null(merge_gap + 1) > merge_gap)
        .cum_sum().over(["timeline", "host", "stage"]).alias("_ep"))
    ep = a.group_by(["timeline", "host", "stage", "_ep"]).agg(
        pl.col("window").min().alias("start"), pl.col("window").max().alias("end")).drop("_ep")
    ep = ep.sort(["timeline", "host", "start", "stage"])
    # The previous stage of an onset is the stage the host was *most recently doing* before it:
    # among earlier-starting episodes, the one active latest (an episode still running at the
    # onset counts as active at the onset). Ordering by start instead would call "Impact, a
    # short C2 burst inside it, then Impact again after a pause" a C2 -> Impact step.
    prev, prog = [], []
    for (_, _), g in ep.group_by(["timeline", "host"], maintain_order=True):
        st, sa, en = g["stage"].to_list(), g["start"].to_list(), g["end"].to_list()
        for i in range(len(st)):
            cands = [(min(en[j], sa[i]), sa[j], st[j]) for j in range(len(st)) if sa[j] < sa[i]]
            prev.append(max(cands)[2] if cands else None)
            prog.append(max(c[2] for c in cands) if cands else None)
    comp = compromise_mask()
    return ep.with_columns(
        pl.Series("prev_stage", prev, dtype=pl.Int8),
        pl.Series("progress_before", prog, dtype=pl.Int8),
        pl.col("stage").is_in(np.flatnonzero(comp).tolist()).alias("compromise"),
    ).with_columns(
        (pl.col("prev_stage").is_null() | (pl.col("prev_stage") != pl.col("stage"))).alias("transition"),
        pl.col("prev_stage").is_not_null().alias("has_history"),
    )


def anchor_grid(hw: pl.DataFrame, hosts: pl.DataFrame, *, stride: int = 30) -> pl.DataFrame:
    """Anchors every ``stride`` windows of each timeline's recorded span, for ``hosts``
    (timeline, host), from the host's first activity to the end of the timeline. Windows in
    long gaps between recordings (nights) are skipped."""
    rec = hw.select(["timeline", "window"]).unique().with_columns((pl.col("window") // stride).alias("_b"))
    grid = rec.group_by(["timeline", "_b"]).agg(pl.col("window").max().alias("window")).drop("_b")
    first = hw.join(hosts, on=["timeline", "host"]).group_by(["timeline", "host"]).agg(pl.col("window").min().alias("_first"))
    return (grid.join(first, on="timeline").filter(pl.col("window") >= pl.col("_first"))
            .drop("_first").sort(["timeline", "host", "window"]))


def add_targets(anchors: pl.DataFrame, ep: pl.DataFrame, timeline_end: pl.DataFrame) -> pl.DataFrame:
    """Attach ``next_stage``, ``next_in`` (windows) and ``y_<H>`` / ``v_<H>`` to anchors."""
    tr = ep.filter(pl.col("transition")).select(["timeline", "host", pl.col("start").alias("_next_t"),
                                                  pl.col("stage").alias("next_stage")]).sort("_next_t")
    a = anchors.sort("window").join_asof(tr, left_on="window", right_on="_next_t", by=["timeline", "host"],
                                         strategy="forward", allow_exact_matches=False)
    a = a.join(timeline_end, on="timeline", how="left")
    comp_next = ep.filter(pl.col("transition") & pl.col("compromise")).select(
        ["timeline", "host", pl.col("start").alias("_next_c")]).sort("_next_c")
    a = a.sort("window").join_asof(comp_next, left_on="window", right_on="_next_c", by=["timeline", "host"],
                                   strategy="forward", allow_exact_matches=False)
    exprs = [(pl.col("_next_t") - pl.col("window")).alias("next_in"),
             pl.col("next_stage").fill_null(-1).cast(pl.Int8)]
    for name, h in HORIZONS.items():
        dt = pl.col("_next_c") - pl.col("window")
        y = dt.is_not_null() & (dt <= h)
        exprs += [y.alias(f"y_{name}"), (y | (pl.col("_end") - pl.col("window") >= h)).alias(f"v_{name}")]
    return a.with_columns(exprs).drop(["_next_t", "_next_c", "_end"]).sort(["timeline", "host", "window"])


def timeline_ends(hw: pl.DataFrame) -> pl.DataFrame:
    return hw.group_by("timeline").agg(pl.col("window").max().alias("_end"))

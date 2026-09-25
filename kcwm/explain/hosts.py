"""Who is driving the forecast? Host attribution by flow removal and re-aggregation.

For each candidate host, all flows it sent or received in the recent context are removed,
the windows are rebuilt with the same feature code, and the rollout is re-run. The drop in
P_infil is that host's contribution. This is a counterfactual on the raw flows, not on
features, so it respects every interaction between features. v1 used the same
remove-and-re-aggregate operation (``counterfactual/replay.py``) to simulate defender
actions. Here it only answers "which hosts", which is what the flagged-flows table needs.

Candidates are the most active hosts in the last ``recent`` windows (bounded, so a click in
the UI stays interactive).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import torch

from .. import config
from ..data.sequences import context_index, model_input
from ..features.build import build_capture
from ..features.normalize import transform_windows
from ..features.registry import MASK_NAMES
from ..model.rollout import analytic, estimate_progress


def _risk_for_flows(bundle, flows: pl.DataFrame, target_window: int, cidrs: list[str], mode: str) -> float:
    cfg = config.load()
    _, win = build_capture(flows, dataset="upload", capture="attr", cfg=cfg, internal_cidrs=cidrs)
    win = win.sort("window")
    idx = np.flatnonzero(win["window"].to_numpy() == target_window)
    if idx.size == 0 or idx[0] < bundle.context - 1:
        return float("nan")
    names = list(bundle.scaler.names)
    raw = win.select(names).to_numpy().astype(np.float64)
    _, session = np.unique(win["session_key"].to_numpy(), return_inverse=True)
    x = transform_windows(bundle.scaler, raw, mode=mode, session=session,
                          warmup=win["warmup"].to_numpy().astype(bool))
    from ..data.sequences import Arrays

    arr = Arrays(x=x, m=win.select(MASK_NAMES).to_numpy().astype(np.float32),
                 stage=np.zeros(win.height, dtype=np.int64), session=session, pos=np.zeros(win.height, dtype=np.int64),
                 remaining=np.zeros(win.height, dtype=np.int64), y={}, valid={}, names=names)
    with torch.no_grad():
        xt = torch.from_numpy(model_input(arr, context_index(idx[:1], bundle.context)))
        m = bundle.model
        H = m.encode(xt)
        fc = analytic(m, H[:, -1], torch.softmax(m.nowcast(H[:, -1]), -1), estimate_progress(m, H), bundle.horizon)
    return float(fc.p_infil[0, -1])


def host_attribution(analysis, bundle, target_window: int, *, recent: int = 12, top_hosts: int = 8) -> pl.DataFrame:
    """Rank hosts by how much removing their recent flows lowers P_infil at ``target_window``."""
    flows = analysis.flows
    cidrs = analysis.meta["internal_cidrs"]
    mode = analysis.meta["mode"]
    seconds = float(config.load()["window"]["seconds"])
    lookback = bundle.context * 2 + 1
    lo = target_window - lookback
    seg = flows.filter(pl.col("window").is_between(lo, target_window)).drop(["window", "session"], strict=False)
    recent_flows = seg.filter(pl.col("ts") >= (target_window - recent + 1) * seconds)
    hosts = (
        pl.concat([recent_flows.select(pl.col("src_ip").alias("host")),
                   recent_flows.select(pl.col("dst_ip").alias("host"))])
        .group_by("host").len().sort("len", descending=True).head(top_hosts)
    )
    base = _risk_for_flows(bundle, seg, target_window, cidrs, mode)
    rows = []
    for h, n in hosts.iter_rows():
        # remove the host's flows only in the recent span: its older history stays observed
        keep = seg.filter(~(((pl.col("src_ip") == h) | (pl.col("dst_ip") == h))
                            & (pl.col("ts") >= (target_window - recent + 1) * seconds)))
        r = _risk_for_flows(bundle, keep, target_window, cidrs, mode)
        rows.append({"host": h, "recent_flows": int(n), "risk_without": r, "contribution": base - r})
    return pl.DataFrame(rows).sort("contribution", descending=True).with_columns(pl.lit(base).alias("risk"))


def flagged_flows(analysis, target_window: int, hosts: list[str], *, recent: int = 12, limit: int = 200) -> pl.DataFrame:
    seconds = float(config.load()["window"]["seconds"])
    f = analysis.flows.filter(
        (pl.col("ts") >= (target_window - recent + 1) * seconds)
        & (pl.col("ts") < (target_window + 1) * seconds)
        & (pl.col("src_ip").is_in(hosts) | pl.col("dst_ip").is_in(hosts))
    )
    cols = ["ts", "src_ip", "src_port", "dst_ip", "dst_port", "protocol", "duration", "fwd_packets",
            "bwd_packets", "fwd_bytes", "bwd_bytes", "syn_count", "ack_count", "rst_count"]
    extra = [c for c in ("label", "stage") if c in f.columns]
    return (f.select(cols + extra).sort("ts").head(limit)
            .with_columns(pl.from_epoch(pl.col("ts").cast(pl.Int64)).alias("time")))

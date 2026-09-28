"""Causal per-host kill-chain memory, evaluated at forecast anchors.

The tracker consumes per-host-window stage probabilities (from the nowcast model; or the true
labels, one-hot, for the "oracle history" upper bound) and keeps, per host and stage:

* the strongest evidence ever seen, and how many windows showed the stage;
* minutes since the stage was first and last seen on this host;
* the same "last seen" and "hosts affected so far" for the whole network, because an attack
  on one host raises the odds for its neighbours (lateral movement).

At an anchor it also summarises the host's activity in the current 5-minute bucket, and the
ATT&CK prior's view of what follows the host's latest stage given its progress. Only rows at or
before the anchor are used.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ..model.world_model import prior_progress_bonus, prior_transition
from ..stages import N_STAGES

TAU = 0.5
NEVER = float(np.log1p(1e5))  # "never seen": log(1 + minutes) of about 70 days
MINUTES_PER_WINDOW = 10.0 / 60.0
ATTACK = range(1, N_STAGES)
CURRENT = ["o_n", "i_n", "o_int_hosts", "i_ext_hosts", "o_ports", "i_ports", "o_new_int", "i_auth"]


def prior_next(last: np.ndarray, progress: np.ndarray) -> np.ndarray:
    """ATT&CK prior P(next *different* attack stage | last stage, progress); (n, 6) over stages 1..6."""
    logits = np.log(prior_transition())[last] + prior_progress_bonus()[progress]
    logits[:, 0] = -np.inf
    logits[np.arange(len(last)), last] = np.where(last > 0, -np.inf, logits[np.arange(len(last)), last])
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p = p / p.sum(axis=1, keepdims=True)
    return p[:, 1:]


def _host_memory(hw: pl.DataFrame, probs: np.ndarray) -> pl.DataFrame:
    """Per host row: cumulative evidence, counts, first/last windows per stage."""
    cols = {f"p{s}": probs[:, s] for s in ATTACK}
    d = hw.select(["timeline", "host", "window"]).with_columns([pl.Series(k, v) for k, v in cols.items()])
    d = d.sort(["timeline", "host", "window"])
    exprs = []
    for s in ATTACK:
        hit = pl.col(f"p{s}") >= TAU
        exprs += [
            pl.col(f"p{s}").cum_max().over(["timeline", "host"]).alias(f"ev{s}"),
            hit.cast(pl.Int32).cum_sum().over(["timeline", "host"]).alias(f"cnt{s}"),
            pl.when(hit).then(pl.col("window")).forward_fill().over(["timeline", "host"]).alias(f"last{s}"),
            pl.when(hit).then(pl.col("window")).min().over(["timeline", "host"]).alias(f"_firstall{s}"),
        ]
    d = d.with_columns(exprs)
    # the first hit is only known once it has happened
    d = d.with_columns([pl.when(pl.col("window") >= pl.col(f"_firstall{s}")).then(pl.col(f"_firstall{s}"))
                        .alias(f"first{s}") for s in ATTACK]).drop([f"_firstall{s}" for s in ATTACK])
    return d.with_columns(pl.int_range(1, pl.len() + 1).over(["timeline", "host"]).alias("rows_seen"))


def _network_memory(hw: pl.DataFrame, probs: np.ndarray) -> pl.DataFrame:
    """Per stage: (stage, hosts-so-far table, windows where any host showed it)."""
    parts = []
    base = hw.select(["timeline", "host", "window"])
    for s in ATTACK:
        hit = base.filter(pl.Series(probs[:, s] >= TAU))
        if hit.height == 0:
            continue
        firsts = hit.group_by(["timeline", "host"]).agg(pl.col("window").min().alias("window"))
        n_hosts = (firsts.group_by(["timeline", "window"]).len().sort(["timeline", "window"])
                   .with_columns(pl.col("len").cum_sum().over("timeline").alias(f"net_hosts{s}")).drop("len"))
        last = hit.select(["timeline", "window"]).unique().sort("window").with_columns(pl.col("window").alias("_nl"))
        parts.append((s, n_hosts, last))
    return parts


def anchor_features(anchors: pl.DataFrame, hw: pl.DataFrame, probs: np.ndarray, *, stride: int = 30
                    ) -> tuple[np.ndarray, list[str], pl.DataFrame]:
    """Feature matrix for ``anchors`` (timeline, host, window). Returns (X, names, info), where
    ``info`` has the tracker's predicted progress and latest stage per anchor."""
    mem = _host_memory(hw, probs)
    a = anchors.select(["timeline", "host", "window"]).with_row_index("_i").sort("window")
    a = a.join_asof(mem.sort("window").rename({"window": "_w"}), left_on="window", right_on="_w",
                    by=["timeline", "host"], strategy="backward")
    names = []

    def mins(expr):
        return ((pl.col("window") - expr) * MINUTES_PER_WINDOW).log1p().fill_null(NEVER)

    a = a.with_columns(
        [mins(pl.col(f"last{s}")).alias(f"since_last{s}") for s in ATTACK]
        + [mins(pl.col(f"first{s}")).alias(f"since_first{s}") for s in ATTACK]
        + [pl.col(f"ev{s}").fill_null(0.0) for s in ATTACK]
        + [pl.col(f"cnt{s}").fill_null(0).cast(pl.Float64).log1p().alias(f"lcnt{s}") for s in ATTACK]
        + [mins(pl.col("_w")).alias("since_active"), pl.col("rows_seen").fill_null(0).cast(pl.Float64).log1p().alias("lrows")]
    )
    for s in ATTACK:
        names += [f"since_last{s}", f"since_first{s}", f"ev{s}", f"lcnt{s}"]
    names += ["since_active", "lrows"]
    # network-wide memory
    for s, n_hosts, last in _network_memory(hw, probs):
        a = a.sort("window").join_asof(n_hosts, on="window", by="timeline", strategy="backward")
        a = a.sort("window").join_asof(last, on="window", by="timeline", strategy="backward")
        a = a.with_columns(mins(pl.col("_nl")).alias(f"net_since{s}"),
                           pl.col(f"net_hosts{s}").fill_null(0).cast(pl.Float64).log1p().alias(f"net_lhosts{s}")).drop("_nl")
    for s in ATTACK:
        if f"net_since{s}" not in a.columns:
            a = a.with_columns(pl.lit(NEVER).alias(f"net_since{s}"), pl.lit(0.0).alias(f"net_lhosts{s}"))
        names += [f"net_since{s}", f"net_lhosts{s}"]
    # this host's activity in the ``stride`` windows up to and including the anchor: each host
    # row goes to the first anchor at or after it, and only if it lies within the stride
    cur = hw.select(["timeline", "host", "window", *CURRENT]).with_columns(
        [pl.Series(f"cp{s}", probs[:, s]) for s in ATTACK]).sort("window")
    anc = a.select(["timeline", "host", pl.col("window").alias("_aw")]).unique().sort("_aw")
    cur = cur.join_asof(anc, left_on="window", right_on="_aw", by=["timeline", "host"], strategy="forward")
    cur = cur.filter(pl.col("_aw").is_not_null() & (pl.col("window") > pl.col("_aw") - stride))
    cur = cur.group_by(["timeline", "host", "_aw"]).agg(
        [pl.col(c).sum().log1p().alias(f"cur_{c}") for c in CURRENT] + [pl.col(f"cp{s}").max().alias(f"cur_p{s}") for s in ATTACK])
    a = a.join(cur, left_on=["timeline", "host", "window"], right_on=["timeline", "host", "_aw"], how="left")
    cur_names = [f"cur_{c}" for c in CURRENT] + [f"cur_p{s}" for s in ATTACK]
    a = a.with_columns([pl.col(c).fill_null(0.0) for c in cur_names])
    names += cur_names
    a = a.sort("_i")
    # predicted progress and latest stage, and the ATT&CK prior's next-stage distribution
    last_w = np.stack([a[f"last{s}"].fill_null(-1).to_numpy() for s in ATTACK], axis=1)
    seen = last_w >= 0
    progress = np.where(seen.any(1), N_STAGES - 1 - np.argmax(seen[:, ::-1], axis=1), 0)
    latest = np.where(seen.any(1), np.argmax(last_w, axis=1) + 1, 0)
    pri = prior_next(latest, progress)
    X = np.column_stack([a.select(names).to_numpy().astype(np.float32), progress, latest, pri]).astype(np.float32)
    names = names + ["progress", "latest"] + [f"prior{s}" for s in ATTACK]
    info = pl.DataFrame({"progress": progress.astype(np.int8), "latest": latest.astype(np.int8)})
    return X, names, info

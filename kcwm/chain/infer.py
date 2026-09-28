"""Host kill-chain analysis of one capture, with the released chain models.

``analyze_hosts(flows, cidrs, bundle)`` returns, for every internal host that shows attack
evidence:

* ``hosts``: the stages seen on it (first / last time, predicted from traffic), its kill-chain
  progress, the forecast next stage with probabilities, and P(it enters a compromise stage
  within 1 h / 24 h), as of the end of the capture;
* ``timeline``: the same forecast every 5 minutes (for the "what did we know, when" view);
* ``evidence``: per host and 10-s window, the stage probabilities the tracker consumed.

Ground-truth stages are attached when the file carries labels, for display only.
"""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import polars as pl

from ..stages import N_STAGES, STAGES
from .hosts import feature_matrix, host_windows
from .model import Forecaster, fit_forecaster, fit_nowcast, predict_nowcast
from .targets import HORIZONS, add_targets, anchor_grid, episodes, timeline_ends
from .tracker import TAU, anchor_features

STRIDE = 30


@dataclass
class ChainBundle:
    nowcast: object
    forecaster: Forecaster
    meta: dict

    def save(self, path) -> None:
        joblib.dump({"nowcast": self.nowcast, "forecaster": self.forecaster, "meta": self.meta}, path)

    @classmethod
    def load(cls, path) -> ChainBundle:
        d = joblib.load(path)
        return cls(d["nowcast"], d["forecaster"], d["meta"])


def train_bundle(hw: pl.DataFrame, anchors: pl.DataFrame, groups: list[str], *, seed: int = 17,
                 meta: dict | None = None) -> ChainBundle:
    """Nowcast on every row of ``groups``; forecaster on tracker features built from 2-way
    cross-fitted nowcast probabilities (the same recipe the chain protocol scores)."""
    from .run import _event_ids

    Xh = feature_matrix(hw)
    y = hw["stage_now"].to_numpy().astype(int)
    rg = hw["group"].to_numpy()
    rows = np.isin(rg, groups)
    nowcast = fit_nowcast(Xh[rows], y[rows], seed=seed)
    probs = np.zeros((hw.height, N_STAGES))
    probs[:, 0] = 1.0
    halves = [groups[0::2], groups[1::2]]
    for a, b in (halves, halves[::-1]):
        fr, pr = np.isin(rg, a), np.isin(rg, b)
        if fr.any() and pr.any():
            probs[pr] = predict_nowcast(fit_nowcast(Xh[fr], y[fr], seed=seed), Xh[pr])
    X, names, _ = anchor_features(anchors, hw, probs, stride=STRIDE)
    tr = np.isin(anchors["group"].to_numpy(), groups)
    haz = {h: (anchors[f"y_{h}"].to_numpy()[tr], anchors[f"v_{h}"].to_numpy()[tr]) for h in HORIZONS}
    fc = fit_forecaster(X[tr], anchors["next_stage"].to_numpy().astype(int)[tr], _event_ids(anchors)[tr], haz,
                        names=names, seed=seed)
    return ChainBundle(nowcast, fc, {"groups": groups, "seed": seed, **(meta or {})})


@dataclass
class HostAnalysis:
    hosts: pl.DataFrame
    timeline: pl.DataFrame
    evidence: pl.DataFrame
    truth: pl.DataFrame | None


def analyze_hosts(flows: pl.DataFrame, cidrs: list[str], bundle: ChainBundle, *, seconds: float = 10.0,
                  max_hosts: int = 25) -> HostAnalysis:
    hw = host_windows(flows, cidrs, seconds=seconds).with_columns(pl.lit("upload").alias("timeline"))
    empty = HostAnalysis(pl.DataFrame(), pl.DataFrame(), pl.DataFrame(), None)
    if hw.height == 0:
        return empty
    probs = predict_nowcast(bundle.nowcast, feature_matrix(hw))
    # hosts worth tracking: any attack-stage evidence at the tracker's threshold
    ev = hw.select(["host"]).with_columns(pl.Series("_m", probs[:, 1:].max(1)))
    score = ev.group_by("host").agg(pl.col("_m").max().alias("peak"), (pl.col("_m") >= TAU).sum().alias("hits"))
    tracked = score.filter(pl.col("hits") > 0).sort(["hits", "peak"], descending=True).head(max_hosts)
    if tracked.height == 0:
        return empty
    scope = tracked.select("host").with_columns(pl.lit("upload").alias("timeline"))
    anchors = anchor_grid(hw, scope, stride=STRIDE)
    X, _, info = anchor_features(anchors, hw, probs, stride=STRIDE)
    pn = bundle.forecaster.predict_next(X)
    hz = bundle.forecaster.predict_hazard(X)
    tl = anchors.with_columns(
        [pl.Series(f"next_{STAGES[s]}", pn[:, s - 1]) for s in range(1, N_STAGES)]
        + [pl.Series(f"p_{h}", hz[h]) for h in HORIZONS]
        + [pl.Series("progress", info["progress"]), pl.Series("latest", info["latest"])]
    ).with_columns(pl.from_epoch((pl.col("window") * seconds).cast(pl.Int64)).alias("time"))
    evid = hw.select(["host", "window"]).with_columns(
        [pl.Series(STAGES[s], probs[:, s]) for s in range(1, N_STAGES)]
    ).join(scope.select("host"), on="host", how="semi").with_columns(
        pl.from_epoch((pl.col("window") * seconds).cast(pl.Int64)).alias("time"))
    rows = []
    last = tl.sort("window").group_by("host").agg(pl.all().last())
    for r in last.iter_rows(named=True):
        e = evid.filter(pl.col("host") == r["host"])
        seen = []
        for s in range(1, N_STAGES):
            hit = e.filter(pl.col(STAGES[s]) >= TAU)
            if hit.height:
                seen.append((hit["window"].min(), STAGES[s], hit["time"].min(), hit["time"].max(), hit.height))
        seen.sort()
        nxt = sorted(((r[f"next_{STAGES[s]}"], STAGES[s]) for s in range(1, N_STAGES)), reverse=True)
        rows.append({
            "host": r["host"],
            "chain so far": " → ".join(x[1] for x in seen) or "-",
            "first seen": seen[0][2] if seen else None,
            "latest": STAGES[int(r["latest"])],
            "progress": STAGES[int(r["progress"])],
            "next stage": nxt[0][1], "p(next)": round(float(nxt[0][0]), 2),
            "2nd": nxt[1][1], "p(2nd)": round(float(nxt[1][0]), 2),
            "P(compromise step ≤1h)": round(float(r["p_1h"]), 2),
            "P(compromise step ≤24h)": round(float(r["p_24h"]), 2),
        })
    hosts = pl.DataFrame(rows).sort("P(compromise step ≤24h)", descending=True) if rows else pl.DataFrame()
    truth = None
    if bool((hw["stage_now"] > 0).any()):
        ep = episodes(hw.with_columns(pl.lit("upload").alias("timeline")), merge_gap=360)
        truth = ep.with_columns(
            pl.col("stage").replace_strict(dict(enumerate(STAGES)), return_dtype=pl.Utf8).alias("stage_name"),
            pl.from_epoch((pl.col("start") * seconds).cast(pl.Int64)).alias("start_time"),
            pl.from_epoch((pl.col("end") * seconds).cast(pl.Int64)).alias("end_time"))
    return HostAnalysis(hosts, tl, evid, truth)


def default_chain_bundle() -> str | None:
    from .. import config

    p = config.resolve(config.load()["paths"]["release"]) / "chain.joblib"
    return str(p) if p.exists() else None


__all__ = ["ChainBundle", "HostAnalysis", "analyze_hosts", "train_bundle", "default_chain_bundle",
           "add_targets", "timeline_ends"]

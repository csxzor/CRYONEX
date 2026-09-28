"""Chain protocol: leave-one-group-out over capture groups, so every real chain step is
forecast by models that never saw its recording.

Groups: CTU-13 scenarios in three blocks, each CIC-IDS2017 day, CIC-IDS2018 (all days),
DAPT2020 (the whole week: one APT) and DARPA 2000 (test only, never trained on). For each
held-out group the nowcast is trained on the other groups; the other groups' own stage
probabilities come from a 2-way cross-fit, so the memory the forecaster learns from is as
noisy as the memory it will see at test time.

Two variants share every step except the tracker's input:
* ``deployable``: the tracker reads predicted stage probabilities (what a live system has);
* ``oracle``: the tracker reads the true labels (upper bound: is the transition model right
  when the history is right?).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from .. import config
from ..stages import N_STAGES
from .hosts import feature_matrix
from .model import fit_forecaster, fit_nowcast, predict_nowcast
from .targets import HORIZONS, PER_CAPTURE, add_targets, anchor_grid, episodes, merge_timeline
from .tracker import anchor_features

CTU_BLOCKS = {**{f"ctu-s{i:02d}": "ctu-A" for i in (1, 2, 3, 4)},
              **{f"ctu-s{i:02d}": "ctu-B" for i in (5, 6, 7, 8, 9)},
              **{f"ctu-s{i:02d}": "ctu-C" for i in (10, 11, 12, 13)}}
TEST_ONLY = {"darpa2000"}


def group_expr() -> pl.Expr:
    ctu = pl.col("capture").replace_strict(CTU_BLOCKS, default=pl.col("capture"))
    return (pl.when(pl.col("dataset") == "ctu13").then(ctu)
            .when(pl.col("dataset") == "cicids2017").then(pl.col("capture"))
            .otherwise(pl.col("dataset")).alias("group"))


def prepare(cfg: dict | None = None, *, benign_hosts: int = 60, seed: int = 0, log=print) -> dict:
    """Tracked hosts' rows, true episodes and anchors with targets. Hosts are chosen from a
    light projection first, so only their rows (about a tenth of all) are ever loaded."""
    cfg = cfg or config.load()
    proc = config.processed_dir(cfg)
    cache = proc / "chain_hw.parquet"
    tl = (pl.when(pl.col("dataset").is_in(PER_CAPTURE)).then(pl.col("capture")).otherwise(pl.col("dataset"))
          .alias("timeline"))
    files = sorted((proc / "hosts").glob("*.parquet"))
    if cache.exists():
        hw = pl.read_parquet(cache)
        ends = pl.read_parquet(cache.with_suffix(".ends.parquet"))
        n_all = int(ends["_rows"].sum())
        ends = ends.drop("_rows")
    else:
        light = pl.concat([pl.scan_parquet(f).select(["dataset", "capture", "host", "window", "stage_now"])
                           for f in files]).with_columns(tl).collect()
        n_all = light.height
        rng = np.random.default_rng(seed)
        att = light.filter(pl.col("stage_now") > 0).select(["timeline", "host"]).unique().sort(["timeline", "host"])
        busy = (light.group_by(["timeline", "host"]).len().filter(pl.col("len") >= 60)
                .join(att, on=["timeline", "host"], how="anti"))
        picks = []
        for _, g in busy.sort(["timeline", "host"]).group_by("timeline", maintain_order=True):
            k = min(benign_hosts, g.height)
            picks.append(g[np.sort(rng.choice(g.height, k, replace=False)).tolist()].select(["timeline", "host"]))
        scope = pl.concat([att, *picks])
        ends = light.group_by("timeline").agg(pl.col("window").max().alias("_end"), pl.len().alias("_rows"))
        ends.write_parquet(cache.with_suffix(".ends.parquet"))
        ends = ends.drop("_rows")
        del light
        # Only tracked hosts are kept. Network-wide memory is then built from them too: an
        # approximation of a network with hundreds of quiet hosts, stated in BENCHMARKS.
        rows = pl.concat([pl.scan_parquet(f).with_columns(tl).join(scope.lazy(), on=["timeline", "host"], how="semi")
                          for f in files], how="diagonal_relaxed").collect().drop("timeline")
        hw = merge_timeline(rows).with_columns(group_expr())
        hw.write_parquet(cache, compression="zstd")
    scope = hw.select(["timeline", "host"]).unique()
    ep = episodes(hw)
    anchors = add_targets(anchor_grid(hw, scope), ep, ends)
    # anchor group = group of the recording that covers the anchor window
    cov = hw.select(["timeline", "window", "group"]).unique(subset=["timeline", "window"]).sort("window")
    anchors = anchors.sort("window").join_asof(cov, on="window", by="timeline", strategy="backward")
    anchors = anchors.sort(["timeline", "host", "window"])
    log(f"host rows {hw.height:,} of {n_all:,}; tracked hosts {scope.height:,}; anchors {anchors.height:,}; "
        f"transitions {int(ep['transition'].sum())} ({int((ep['transition'] & ep['has_history']).sum())} chain steps)")
    return {"hw": hw, "ep": ep, "anchors": anchors}


def _event_ids(anchors: pl.DataFrame) -> np.ndarray:
    key = anchors.select(pl.concat_str([pl.col("timeline"), pl.col("host"),
                                        (pl.col("window") + pl.col("next_in").fill_null(-1)).cast(pl.Utf8)], separator="|"))
    return key.to_series().to_numpy()


def run(tag: str = "chain", *, cfg: dict | None = None, seed: int = 17, groups: list[str] | None = None,
        log=print) -> Path:
    cfg = cfg or config.load()
    t0 = time.time()
    P = prepare(cfg, log=log)
    hw, anchors = P["hw"], P["anchors"]
    Xh = feature_matrix(hw)
    y_now = hw["stage_now"].to_numpy().astype(int)
    row_group = hw["group"].to_numpy()
    anc_group = anchors["group"].to_numpy()
    all_groups = sorted(set(row_group))
    folds = groups or all_groups
    trainable = [g for g in all_groups if not any(g.startswith(t) for t in TEST_ONLY)]
    onehot = np.eye(N_STAGES)[y_now]
    next_stage = anchors["next_stage"].to_numpy().astype(int)
    events = _event_ids(anchors)
    outs = []
    for g in folds:
        tr_groups = [x for x in trainable if x != g]
        tr_rows = np.isin(row_group, tr_groups)
        te_anc = anc_group == g
        tr_anc = np.isin(anc_group, tr_groups)
        if not te_anc.any():
            continue
        # nowcast: full-train model for the held-out group, 2-way cross-fit for the training groups
        probs = np.zeros((hw.height, N_STAGES))
        probs[:, 0] = 1.0
        full = fit_nowcast(Xh[tr_rows], y_now[tr_rows], seed=seed)
        rest = ~tr_rows
        probs[rest] = predict_nowcast(full, Xh[rest])
        halves = [tr_groups[0::2], tr_groups[1::2]]
        for a, b in (halves, halves[::-1]):
            fit_rows = np.isin(row_group, a)
            pred_rows = np.isin(row_group, b)
            if fit_rows.any() and pred_rows.any():
                probs[pred_rows] = predict_nowcast(fit_nowcast(Xh[fit_rows], y_now[fit_rows], seed=seed), Xh[pred_rows])
        rec = {"group": g}
        for variant, pr in (("deployable", probs), ("oracle", onehot)):
            X, names, info = anchor_features(anchors, hw, pr)
            haz = {h: (anchors[f"y_{h}"].to_numpy()[tr_anc], anchors[f"v_{h}"].to_numpy()[tr_anc]) for h in HORIZONS}
            fc = fit_forecaster(X[tr_anc], next_stage[tr_anc], events[tr_anc], haz, names=names, seed=seed)
            pn = fc.predict_next(X[te_anc])
            hz = fc.predict_hazard(X[te_anc])
            rec[variant] = {"next": pn, "hazard": hz, "progress": info["progress"].to_numpy()[te_anc],
                            "latest": info["latest"].to_numpy()[te_anc]}
        rows = anchors.filter(pl.Series(te_anc)).select(
            ["timeline", "host", "window", "group", "next_stage", "next_in",
             *[f"y_{h}" for h in HORIZONS], *[f"v_{h}" for h in HORIZONS]])
        cols = []
        for variant in ("deployable", "oracle"):
            r = rec[variant]
            cols += [pl.Series(f"{variant}_next{s}", r["next"][:, s - 1]) for s in range(1, N_STAGES)]
            cols += [pl.Series(f"{variant}_haz_{h}", r["hazard"][h]) for h in HORIZONS]
            cols += [pl.Series(f"{variant}_progress", r["progress"]), pl.Series(f"{variant}_latest", r["latest"])]
        outs.append(rows.with_columns(cols))
        # nowcast quality on the held-out rows
        te_rows = row_group == g
        yp = probs[te_rows].argmax(1)
        att = y_now[te_rows] > 0
        log(f"[{tag}] {g}: {int(te_anc.sum()):,} anchors; nowcast attack-row recall "
            f"{(yp[att] > 0).mean() if att.any() else float('nan'):.2f}, stage acc on attack rows "
            f"{(yp[att] == y_now[te_rows][att]).mean() if att.any() else float('nan'):.2f}; {time.time() - t0:.0f}s")
    out_dir = config.resolve(cfg["paths"]["results"]) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    preds = pl.concat(outs, how="vertical_relaxed")
    preds.write_parquet(out_dir / "oof_predictions.parquet")
    P["ep"].write_parquet(out_dir / "episodes.parquet")
    (out_dir / "run.json").write_text(json.dumps({"tag": tag, "seed": seed, "folds": folds,
                                                   "seconds": time.time() - t0}, indent=2))
    return out_dir

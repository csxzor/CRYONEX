"""Generate campaign captures for a dataset and write their windows next to the real ones.

Output: ``<processed>/campaigns/<dataset>/<split>/camp-<n>.parquet`` (windows, built with the
same ``features.build.build_capture`` as real data) plus ``plans.jsonl`` (what was injected,
where). Campaign windows carry ``synthetic = True`` and ``campaign_split``. The P1 real-data
splits never include them; ``kcwm evaluate --campaigns`` adds the train campaigns to the
training anchors, and the P4 protocol scores the test campaigns.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl

from .. import config
from ..features.build import build_capture
from .campaign import benign_backgrounds, build_campaign, extract_snippets


def split_lookup(win: pl.DataFrame, fractions=(0.60, 0.15, 0.25)):
    """window -> split name for one capture, using P1's per-session cut points."""
    pos = win["pos_in_session"].to_numpy()
    length = win.group_by("session").agg(pl.len().alias("n")).join(win.select("session"), on="session", how="right")["n"].to_numpy()
    tr = (length * fractions[0]).astype(int)
    ca = tr + (length * fractions[1]).astype(int)
    names = np.where(pos < tr, "train", np.where(pos < ca, "calibration", "test"))
    table = dict(zip(win["window"].to_list(), names.tolist()))
    return lambda w: table.get(w, "none")


def generate(dataset: str, *, per_split: dict[str, int], length: float = 2700.0, seed: int = 0,
             cfg: dict | None = None, log=print) -> dict:
    cfg = cfg or config.load()
    proc = config.processed_dir(cfg)
    cidrs = cfg["datasets"][dataset]["internal_cidrs"]
    rng = np.random.default_rng(seed)
    captures = sorted(p.stem for p in (proc / "windows").glob("*.parquet")
                      if pl.read_parquet(p, n_rows=1)["dataset"][0] == dataset)
    snippets, backgrounds = [], []
    for cap in captures:
        win = pl.read_parquet(proc / "windows" / f"{cap}.parquet")
        lookup = split_lookup(win)
        flows = pl.read_parquet(proc / "flows" / f"{cap}.parquet").drop(["window", "session", "capture"], strict=False)
        snippets += extract_snippets(flows, cap, lookup)
        backgrounds += [(cap, *b) for b in benign_backgrounds(win, lookup, length=length)]
    log(f"{dataset}: {len(snippets)} snippets, {len(backgrounds)} backgrounds of {length / 60:.0f} min")
    out_root = proc / "campaigns" / dataset
    summary = {}
    for split, n in per_split.items():
        pool = {}
        for s in snippets:
            if s.split == split:
                pool.setdefault(s.stage, []).append(s)
        bgs = [b for b in backgrounds if b[1] == split]
        log(f"  {split}: stages {{{', '.join(f'{k}: {len(v)}' for k, v in sorted(pool.items()))}}}, {len(bgs)} backgrounds")
        if not bgs:
            continue
        d = out_root / split
        d.mkdir(parents=True, exist_ok=True)
        plans = []
        for i in range(n):
            cap, _, t0, t1 = bgs[int(rng.integers(len(bgs)))]
            bg = (pl.scan_parquet(proc / "flows" / f"{cap}.parquet").filter(pl.col("ts").is_between(t0, t1))
                  .drop(["window", "session", "capture"], strict=False).collect())
            flows, plan = build_campaign(bg, pool, rng, cidrs=cidrs, lead_in=1000.0)
            name = f"camp-{dataset}-{split}-{i:03d}"
            _, win = build_capture(flows, dataset=f"campaign-{dataset}", capture=name, cfg=cfg, internal_cidrs=cidrs)
            win = win.with_columns(pl.lit(True).alias("synthetic"), pl.lit(split).alias("campaign_split"))
            win.write_parquet(d / f"{name}.parquet", compression="zstd")
            plans.append({"capture": name, "background": cap, "t0": t0, "plan": plan})
        (d / "plans.jsonl").write_text("\n".join(json.dumps(p) for p in plans))
        stages = [len(p["plan"]) for p in plans]
        summary[split] = {"campaigns": n, "mean_stages": float(np.mean(stages)), "benign_only": int(sum(s == 0 for s in stages))}
        log(f"  {split}: wrote {n} campaigns, mean {np.mean(stages):.1f} stages")
    return summary


def load_campaigns(datasets: list[str], split: str, cfg: dict | None = None) -> pl.DataFrame | None:
    cfg = cfg or config.load()
    files = []
    for ds in datasets:
        files += sorted((config.processed_dir(cfg) / "campaigns" / ds / split).glob("camp-*.parquet"))
    if not files:
        return None
    return pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed").sort(["capture", "window"])

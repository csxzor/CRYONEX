"""Build per-host tables for every processed capture (and campaign) into
``<processed>/hosts/<capture>.parquet``. Reads the processed flow tables written by
``kcwm build``, so no raw data is re-read."""

from __future__ import annotations

import polars as pl

from .. import config
from .hosts import host_windows


def capture_dataset(proc, capture: str) -> str:
    return pl.read_parquet(proc / "windows" / f"{capture}.parquet", n_rows=1)["dataset"][0]


def build_hosts(cfg: dict | None = None, *, datasets: list[str] | None = None, force: bool = False,
                log=print) -> list[str]:
    cfg = cfg or config.load()
    proc = config.processed_dir(cfg)
    out = proc / "hosts"
    out.mkdir(parents=True, exist_ok=True)
    seconds = float(cfg["window"]["seconds"])
    done = []
    for fp in sorted((proc / "flows").glob("*.parquet")):
        cap = fp.stem
        ds = capture_dataset(proc, cap)
        if datasets and ds not in datasets:
            continue
        dst = out / f"{cap}.parquet"
        if dst.exists() and not force:
            done.append(cap)
            continue
        cidrs = cfg["datasets"][ds]["internal_cidrs"]
        flows = pl.read_parquet(fp)
        hw = host_windows(flows, cidrs, seconds=seconds).with_columns(
            pl.lit(ds).alias("dataset"), pl.lit(cap).alias("capture"))
        hw.write_parquet(dst, compression="zstd")
        log(f"{cap}: {flows.height:,} flows -> {hw.height:,} host windows, "
            f"{hw['host'].n_unique()} hosts, {int((hw['stage_now'] > 0).sum()):,} attack host windows")
        done.append(cap)
    return done


def load_hosts(datasets: list[str] | None = None, cfg: dict | None = None) -> pl.DataFrame:
    cfg = cfg or config.load()
    files = sorted((config.processed_dir(cfg) / "hosts").glob("*.parquet"))
    parts = [pl.read_parquet(f) for f in files]
    t = pl.concat(parts, how="vertical_relaxed")
    return t.filter(pl.col("dataset").is_in(datasets)) if datasets else t

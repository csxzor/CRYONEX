"""Dataset build orchestration: raw captures -> per-capture flows and windows on disk.

Output layout (under ``paths.processed``):

* ``flows/<capture>.parquet``: canonical labelled flows with ``window``/``session``. The
  campaign synthesizer and host attribution replay these.
* ``windows/<capture>.parquet``: one row per window (see ``features.build``).
* ``manifest.json``: per-capture summaries plus the size and mtime of every raw input, so a
  result can be traced back to the exact files it came from.

Captures are processed one at a time and freed, which keeps CSE-CIC-IDS2018 (37.7M flows)
inside 15 GB of RAM.
"""

from __future__ import annotations

import gc
import json
import time
from collections.abc import Iterator
from pathlib import Path

import polars as pl
from rich.console import Console

from . import config
from .features.build import build_capture, summarise
from .ingest.flowcsv import read_cicids, read_ctu13, read_dapt2020

console = Console()


def _capture_name(dataset: str, stem: str) -> str:
    short = {"cicids2017": "cic17", "cicids2018": "cic18", "ctu13": "ctu", "dapt2020": "dapt"}[dataset]
    return f"{short}-{stem.lower()}"


def iter_captures(dataset: str, cfg: dict) -> Iterator[tuple[str, Path, callable]]:
    """Yield ``(capture_name, raw_path, reader)`` for every capture of a dataset."""
    spec = cfg["datasets"][dataset]
    root = config.dataset_dir(dataset, cfg)
    if spec["reader"] == "cicids":
        for name in spec["files"]:
            path = root / name
            yield (_capture_name(dataset, Path(name).stem), path,
                   lambda p, d=dataset: read_cicids(p, dataset=d))
    elif spec["reader"] == "ctu13":
        for sc in spec["scenarios"]:
            files = sorted((root / str(sc)).glob("*.binetflow"))
            if not files:
                raise FileNotFoundError(f"CTU-13 scenario {sc}: no .binetflow under {root / str(sc)}")
            yield (_capture_name(dataset, f"s{sc:02d}"), files[0],
                   lambda p, s=sc: read_ctu13(p, scenario=s))
    elif spec["reader"] == "dapt2020":
        files = sorted(root.rglob("*_Flow.csv"))
        if not files:
            raise FileNotFoundError(f"DAPT2020: no CSVs under {root} (download: docs/DATASETS.md)")
        for path in files:
            yield _capture_name(dataset, path.stem.replace(" ", "_")), path, read_dapt2020
    else:
        raise KeyError(spec["reader"])


def packet_block_for(dataset: str, capture_file: Path, cfg: dict) -> pl.DataFrame | None:
    """The PCAP-derived packet block for a capture, when one exists (CIC-IDS2017)."""
    if dataset != "cicids2017":
        return None
    from .ingest.packet_merge import from_v1_cache

    cache = config.data_root(cfg) / "processed" / "cicids2017"
    if (cache / "states_raw.parquet").exists():
        return from_v1_cache(cache, float(cfg["window"]["seconds"]))
    return None


def build_dataset(dataset: str, cfg: dict | None = None, *, only: list[str] | None = None) -> dict:
    cfg = cfg or config.load()
    out = config.processed_dir(cfg)
    (out / "flows").mkdir(parents=True, exist_ok=True)
    (out / "windows").mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    cidrs = cfg["datasets"][dataset].get("internal_cidrs", [])

    packet_cache: pl.DataFrame | None = None
    for capture, path, reader in iter_captures(dataset, cfg):
        if only and capture not in only:
            continue
        t0 = time.time()
        console.print(f"[bold]{capture}[/] <- {path}")
        flows = reader(path)
        if dataset == "cicids2017" and packet_cache is None:
            packet_cache = packet_block_for(dataset, path, cfg)
        flows, win = build_capture(
            flows, dataset=dataset, capture=capture, cfg=cfg,
            internal_cidrs=cidrs, packet_block=packet_cache,
        )
        flows.with_columns(pl.lit(capture).alias("capture")).write_parquet(
            out / "flows" / f"{capture}.parquet", compression="zstd"
        )
        win.write_parquet(out / "windows" / f"{capture}.parquet", compression="zstd")
        stat = path.stat()
        summary = summarise(win) | {
            "dataset": dataset, "flows": int(flows.height), "raw": str(path),
            "raw_bytes": stat.st_size, "raw_mtime": stat.st_mtime,
            "seconds": round(time.time() - t0, 1),
        }
        manifest[capture] = summary
        manifest_path.write_text(json.dumps(manifest, indent=2))
        console.print(
            f"  {summary['flows']:,} flows -> {summary['windows']:,} windows, "
            f"{summary['episodes']} episodes, {summary['onsets']} onsets, "
            f"packet {summary['packet_coverage']:.1%} ({summary['seconds']} s)"
        )
        del flows, win
        gc.collect()
    return manifest


def load_windows(cfg: dict | None = None, *, datasets: list[str] | None = None) -> pl.DataFrame:
    """All built windows, sorted by (capture, window)."""
    cfg = cfg or config.load()
    files = sorted((config.processed_dir(cfg) / "windows").glob("*.parquet"))
    if not files:
        raise FileNotFoundError("no built windows; run `kcwm build` first")
    frames = [pl.read_parquet(f) for f in files]
    df = pl.concat(frames, how="diagonal_relaxed")
    if datasets:
        df = df.filter(pl.col("dataset").is_in(datasets))
    return df.sort(["capture", "window"])

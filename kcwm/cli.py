"""Command-line interface: ``kcwm <command>``."""

from __future__ import annotations

import typer

app = typer.Typer(add_completion=False, help="KC-WM: kill-chain world model for network attack forecasting")


@app.callback()
def _main() -> None:
    """KC-WM command group."""


@app.command()
def build(
    dataset: list[str] = typer.Argument(..., help="cicids2017 | cicids2018 | ctu13 | dapt2020"),
    only: list[str] = typer.Option(None, help="restrict to these capture names"),
) -> None:
    """Raw captures -> labelled flows and per-window features/targets."""
    from .pipeline import build_dataset

    for name in dataset:
        build_dataset(name, only=only or None)


@app.command()
def evaluate(
    protocol: str = typer.Option("p1", help="p1 | lofo | cross"),
    tag: str = typer.Option(None, help="results sub-directory (default: protocol)"),
    datasets: str = typer.Option(None, help="comma list; default all built"),
    models: str = typer.Option("logreg,logreg_stack,hgb,lstm"),
    seeds: str = typer.Option("17"),
    horizon: int = typer.Option(30),
    mode: str = typer.Option("global", help="global | warmup normalisation"),
    target: str = typer.Option("compromise", help="compromise | any (v1-compatible)"),
    family: str = typer.Option(None),
    train_datasets: str = typer.Option(None),
    test_dataset: str = typer.Option(None),
    wm: str = typer.Option(None, help='world-model overrides, e.g. "epochs=2,anchors_per_epoch=2000"'),
    dev: bool = typer.Option(False, help="score on the calibration split; never touch test"),
    campaigns: bool = typer.Option(False, help="add training-split synthetic campaigns to every model's training anchors"),
    device: str = typer.Option("cpu", help="cpu | cuda (training only; bundles always load on CPU)"),
) -> None:
    """Score baselines and the world model (model name ``kcwm``) under one protocol."""
    from . import device as _dev
    from .eval.run import run

    _dev.set(device)

    def split_list(s):
        return [x for x in s.split(",") if x] if s else None

    overrides = {}
    for item in split_list(wm) or []:
        k, v = item.split("=")
        overrides[k] = float(v) if "." in v or "e" in v.lower() else int(v)
    run(protocol=protocol, tag=tag or protocol, datasets=split_list(datasets),
        models=split_list(models), seeds=[int(s) for s in seeds.split(",")], horizon=horizon,
        mode=mode, target=target, family=family, train_datasets=split_list(train_datasets),
        test_dataset=test_dataset, wm_overrides=overrides, dev=dev, campaigns=campaigns)


@app.command()
def report(tag: str = typer.Argument(...), budget: float = typer.Option(0.03)) -> None:
    """Print the results table for a results sub-directory."""
    from .eval.report import table

    typer.echo(table(tag, budget))


@app.command()
def forecast(
    input: str = typer.Argument(..., help="PCAP/PCAPNG, CICFlowMeter CSV or CTU binetflow"),
    bundle: str = typer.Option(None, help="model bundle (.pt)"),
    internal: str = typer.Option(None, help="internal CIDRs, comma separated (default RFC1918)"),
    mode: str = typer.Option(None, help="global | warmup"),
    out: str = typer.Option(None, help="write the full timeline as JSON"),
) -> None:
    """Forecast infiltration risk over a capture, offline."""
    from pathlib import Path

    from .inference.engine import analyze, default_bundle, load_bundle, summary, to_json

    b = load_bundle(bundle or default_bundle())
    a = analyze(input, b, internal_cidrs=internal.split(",") if internal else None, mode=mode)
    for k, v in summary(a).items():
        typer.echo(f"{k:24s} {v}")
    if out:
        Path(out).write_text(to_json(a))


@app.command()
def serve(port: int = typer.Option(8501)) -> None:
    """Launch the offline operator console."""
    import subprocess
    import sys
    from pathlib import Path

    app_path = Path(__file__).parent / "ui" / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port),
                    "--browser.gatherUsageStats", "false"], check=False)


@app.command()
def campaigns(
    dataset: list[str] = typer.Argument(..., help="datasets to synthesize from (same-network only)"),
    train: int = typer.Option(60), calibration: int = typer.Option(10), test: int = typer.Option(30),
    seed: int = typer.Option(0),
) -> None:
    """Synthesize kill-chain campaigns from real snippets + real benign traffic (per split)."""
    from .synth.generate import generate

    for ds in dataset:
        generate(ds, per_split={"train": train, "calibration": calibration, "test": test}, seed=seed)


@app.command()
def refeature() -> None:
    """Add/refresh the long-memory history features on every built capture and campaign."""
    import polars as pl

    from . import config
    from .features.build import add_history

    root = config.processed_dir()
    files = sorted((root / "windows").glob("*.parquet")) + sorted((root / "campaigns").rglob("camp-*.parquet"))
    for path in files:
        add_history(pl.read_parquet(path)).write_parquet(path, compression="zstd")
    typer.echo(f"history features added to {len(files)} files")


@app.command()
def retarget() -> None:
    """Recompute episodes and forecast targets for every built capture (no raw re-read)."""
    import polars as pl

    from . import config
    from .features.build import add_targets

    cfg = config.load()
    for path in sorted((config.processed_dir(cfg) / "windows").glob("*.parquet")):
        win = add_targets(pl.read_parquet(path), cfg)
        win.write_parquet(path, compression="zstd")
        typer.echo(f"{path.stem}: {int(win['is_onset'].sum())} onsets, "
                   f"{int(win['is_stage_onset'].sum())} stage onsets")


if __name__ == "__main__":
    app()

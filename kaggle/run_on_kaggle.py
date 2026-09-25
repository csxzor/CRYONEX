"""Run KC-WM improvement experiments on a Kaggle GPU (dev split only; test is never read).

In a Kaggle notebook (Settings: Accelerator = GPU T4, Internet = On), attach the uploaded
dataset, then run in one cell:

    !python /kaggle/input/kcwm-package/kaggle/run_on_kaggle.py

It copies the package to /kaggle/working (the only writable place), installs the few
packages Kaggle lacks, runs the experiments below one after another (about 1-2 h on a T4),
and writes /kaggle/working/kcwm_results.zip for download (Output tab).

Experiments (3 seeds each, hybrid = world model + gradient boosting, same data recipe as the
frozen final model; history features are now in every window):
  E1  history features only                (ew_weight = 1)
  E2  history features + early-warning weight 5
  E3  history features + early-warning weight 10
  C1  Transformer classifier with history features (control, 1 seed)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("KCWM_WORK", "/kaggle/working/kcwm"))
OUT = Path(os.environ.get("KCWM_OUT", "/kaggle/working"))
DS = "cicids2017,cicids2018,ctu13,dapt2020"
REG = "epochs=15,anchors_per_epoch=8000,patience=4,d=96,layers=3,dropout=0.2,weight_decay=0.05,direct_risk=1"
EXPERIMENTS = [
    ("e1-history", ["--models", "hybrid", "--seeds", "17,23,29", "--wm", REG]),
    ("e2-history-ew5", ["--models", "hybrid", "--seeds", "17,23,29", "--wm", REG + ",ew_weight=5"]),
    ("e3-history-ew10", ["--models", "hybrid", "--seeds", "17,23,29", "--wm", REG + ",ew_weight=10"]),
    ("c1-transformer-history", ["--models", "transformer_clf", "--seeds", "17"]),
]


def sh(cmd: list[str], **kw) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def main() -> None:
    if not WORK.exists():
        shutil.copytree(SRC, WORK)
    os.chdir(WORK)
    import importlib.util

    need = {"polars": "polars>=1.0", "typer": "typer>=0.12", "rich": "rich>=13.0", "yaml": "pyyaml",
            "joblib": "joblib", "sklearn": "scikit-learn", "scipy": "scipy"}
    missing = [spec for mod, spec in need.items() if importlib.util.find_spec(mod) is None]
    if missing:
        sh([sys.executable, "-m", "pip", "install", "-q", *missing])
    # No "pip install -e": Kaggle may run an older Python than pyproject pins; import from the folder.
    import torch

    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE (check Settings > Accelerator)", flush=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "KCWM_DEVICE": "cuda", "PYTHONPATH": str(WORK)}
    only = set(sys.argv[1:])
    experiments = EXPERIMENTS
    if "smoke" in only:  # 2-minute end-to-end check of this script
        experiments = [("smoke", ["--models", "hybrid", "--seeds", "17", "--wm",
                                  "epochs=1,anchors_per_epoch=256,d=32,layers=1,direct_risk=1,ew_weight=5"])]
        only = set()
    for tag, args in experiments:
        if only and tag not in only:
            continue
        log = WORK / "runs" / f"{tag}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "kcwm.cli", "evaluate", "--dev", "--protocol", "p1", "--tag", tag,
               "--datasets", DS, "--mode", "warmup", "--campaigns", "--device", "cuda", *args]
        with log.open("w") as fh:
            print(f"=== {tag} (log: {log})", flush=True)
            subprocess.run(cmd, check=False, stdout=fh, stderr=subprocess.STDOUT, env=env)
        for line in log.read_text().splitlines():
            if " s17 " in line or " s23 " in line or " s29 " in line or "Traceback" in line or "Error" in line:
                print("  ", line, flush=True)
    out = OUT / "kcwm_results"
    shutil.rmtree(out, ignore_errors=True)
    (out / "results").mkdir(parents=True)
    for d in (WORK / "results").glob("dev-*"):
        shutil.copytree(d, out / "results" / d.name, dirs_exist_ok=True)
    shutil.copytree(WORK / "runs", out / "runs", dirs_exist_ok=True)
    shutil.make_archive(str(OUT / "kcwm_results"), "zip", out)
    print(f"DONE: download {OUT / 'kcwm_results.zip'} from the Output tab", flush=True)


if __name__ == "__main__":
    main()

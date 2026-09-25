"""Where does a model win or lose on the dev split? Per-dataset / per-capture AUPRC.

Reloads a saved KC-WM checkpoint (no retraining), refits logistic regression, and scores
both on the dev test rows (second half of calibration). Never touches the test split.

Usage: python scripts/diagnose_dev.py runs/dev-p1-all/kcwm-s17.pt
"""

from __future__ import annotations

import sys

import numpy as np
import polars as pl
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from kcwm import config
from kcwm.baselines import tabular
from kcwm.data.sequences import build_arrays
from kcwm.eval.run import dev_split, make_split
from kcwm.features.normalize import QuantileBinner, RobustScaler
from kcwm.inference.engine import load_bundle
from kcwm.model.train import TrainConfig, forecast_rows
from kcwm.pipeline import load_windows


def main(ckpt: str, K: int = 30) -> None:
    cfg = config.load()
    ds = ["cicids2017", "cicids2018", "ctu13"]
    win = load_windows(cfg, datasets=ds)
    split = dev_split(make_split("p1", win, cfg, horizon=K), win, K)
    b = load_bundle(ckpt)
    raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    scaler = RobustScaler.from_dict(raw["scaler"])
    arr = build_arrays(win, scaler, mode=raw["mode"], horizons=list(cfg["window"]["horizons"]))
    arr.bins = QuantileBinner.from_dict(raw["binner"]).transform(arr.x)
    y, v = arr.y[K], arr.valid[K]
    tr = split.train[v[split.train]]
    te = split.test[v[split.test]]
    tc = TrainConfig(**{k: raw["config"][k] for k in ("context", "horizons", "primary_horizon")})
    wm = forecast_rows(b.model, arr, te, tc)["p_infil"][:, K - 1]
    lr = tabular.fit("logreg", tabular.features_for("logreg", arr, tr), y[tr], seed=17)
    lrs = tabular.predict(lr, tabular.features_for("logreg", arr, te))
    meta = win.select("dataset", "capture")[te.tolist()].with_columns(
        pl.Series("y", y[te]), pl.Series("wm", wm), pl.Series("lr", lrs))

    def row(name, d):
        yy = d["y"].to_numpy()
        if len(np.unique(yy)) < 2:
            return f"{name:28s} n={d.height:5d} pos={yy.mean():.2f}  (one class) wm_mean={d['wm'].mean():.2f} lr_mean={d['lr'].mean():.2f}"
        f = lambda s: (average_precision_score(yy, d[s]), roc_auc_score(yy, d[s]))  # noqa: E731
        (wa, wr), (la, lro) = f("wm"), f("lr")
        return f"{name:28s} n={d.height:5d} pos={yy.mean():.2f}  WM AUPRC {wa:.3f} ROC {wr:.3f} | LR AUPRC {la:.3f} ROC {lro:.3f}"

    print(row("ALL", meta))
    for key in ("dataset", "capture"):
        for name in sorted(meta[key].unique().to_list()):
            print(row(f"{key}={name}", meta.filter(pl.col(key) == name)))


if __name__ == "__main__":
    main(sys.argv[1])

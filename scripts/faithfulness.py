"""Gate G8: are the explanations faithful?

For the highest-risk evaluation anchors of a saved KC-WM checkpoint, delete (reset to the
baseline) the top-5 features by integrated gradients, and separately 5 random features, over
the whole context. Pass bar: the top-5 deletion lowers P_infil at least 2x more than the random
one. Uses the calibration split only (never test).

Usage: python scripts/faithfulness.py runs/dev-r3-kcwm/kcwm-s17.pt [n_anchors]
"""

from __future__ import annotations

import json
import sys

import numpy as np
import torch

from kcwm import config
from kcwm.data.sequences import build_arrays, context_index, model_input
from kcwm.eval.run import make_split
from kcwm.explain.counterfactual import deletion_test
from kcwm.features.normalize import RobustScaler
from kcwm.features.registry import N_FEATURES
from kcwm.inference.engine import load_bundle
from kcwm.model.train import TrainConfig, forecast_rows
from kcwm.pipeline import load_windows


def main(ckpt: str, n: int = 60) -> None:
    cfg = config.load()
    raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    b = load_bundle(ckpt)
    win = load_windows(cfg, datasets=["cicids2017", "cicids2018", "ctu13", "dapt2020"])
    K = int(raw["config"]["primary_horizon"])
    split = make_split("p1", win, cfg, horizon=K)
    arr = build_arrays(win, RobustScaler.from_dict(raw["scaler"]), mode=raw["mode"],
                       horizons=list(cfg["window"]["horizons"]))
    tc = TrainConfig(**{k: raw["config"][k] for k in ("context", "horizons", "primary_horizon", "direct_risk")})
    rng = np.random.default_rng(0)
    pool = rng.choice(split.calibration, size=min(3000, split.calibration.size), replace=False)
    risk = forecast_rows(b.model, arr, pool, tc)["p_infil"][:, K - 1]
    top = pool[np.argsort(-risk)[:n]]
    contexts = [model_input(arr, context_index(np.array([r]), tc.context))[0] for r in top]
    out = deletion_test(b.model, contexts, horizon=K, n_features=N_FEATURES, k=5)
    out["mean_risk_of_explained_anchors"] = float(np.sort(risk)[-n:].mean())
    path = config.results_dir() / "g8_faithfulness.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 60)

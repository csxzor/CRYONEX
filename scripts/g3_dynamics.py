"""Gate G3 on the frozen final models: does the world model predict the *next network state*
better than simple predictors? (P1 test split; the world model's free-running K-step rollout.)

For each seed's checkpoint, every predictor outputs a distribution over the same 16 quantile
bins per feature, and we score the negative log-likelihood of the bin the real next window
falls in, k = 1, 6, 30 windows ahead (10 s, 1 min, 5 min). Lower is better. The gain over the
recent-history histogram gets a 30-minute block-bootstrap CI.

Usage: python scripts/g3_dynamics.py  ->  results/final-p1/g3_dynamics.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kcwm.data.sequences import Arrays
from kcwm.eval.dynamics import evaluate_dynamics
from kcwm.eval.run import assemble
from kcwm.features.normalize import transform_windows
from kcwm.features.registry import MASK_NAMES
from kcwm.inference.engine import load_bundle


def arrays_for(win, b) -> Arrays:
    names = list(b.scaler.names)
    raw = win.select(names).to_numpy().astype(np.float64)
    _, session = np.unique(win["session_key"].to_numpy(), return_inverse=True)
    x = transform_windows(b.scaler, raw, mode=b.mode, session=session, warmup=win["warmup"].to_numpy().astype(bool))
    pos = win["pos_in_session"].to_numpy().astype(np.int64)
    length = np.bincount(session)[session]
    return Arrays(x=x, m=win.select(MASK_NAMES).to_numpy().astype(np.float32),
                  stage=win["stage_now"].to_numpy().astype(np.int64), session=session.astype(np.int64),
                  pos=pos, remaining=length - 1 - pos, y={}, valid={}, bins=b.binner.transform(x), names=names)


def main(d: str = "results/final-p1", seeds=(17, 23, 29)) -> None:
    rd = Path(d)
    r0 = json.loads((rd / f"kcwm-s{seeds[0]}.json").read_text())
    dev = bool(r0["split_notes"].get("dev"))
    win, split, _, _, _ = assemble(protocol="p1", tag="x", datasets=r0["datasets"], horizon=r0["horizon"],
                                   dev=dev, campaigns=bool(r0.get("campaigns")), log=lambda *_: None)
    out = {"split": "P1 dev test (calibration span)" if dev else "P1 test (real data only)",
           "horizons_windows": [1, 6, 30], "seeds": {}}
    for seed in seeds:
        ck = Path(json.loads((rd / f"kcwm-s{seed}.json").read_text())["world_model"]["checkpoint"])
        b = load_bundle(ck)
        arr = arrays_for(win, b)
        res = evaluate_dynamics(b.model, arr, b.binner, split.train, split.test, context=b.context,
                                horizons=(1, 6, 30), seed=seed)
        out["seeds"][seed] = {str(k): v for k, v in res.items()}
        for k, v in res.items():
            print(f"seed {seed} k={k:2}: WM {v['nll_world_model']:.3f}  recent-hist {v['nll_context_hist']:.3f}  "
                  f"persistence {v['nll_persistence']:.3f}  ridge-AR {v['nll_ridge_ar']:.3f}  "
                  f"gain {v['gain_vs_context_hist']:+.3f} CI [{v['gain_ci'][0]:+.3f}, {v['gain_ci'][1]:+.3f}]")
    (rd / "g3_dynamics.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    import sys

    main(*(sys.argv[1:2] or ["results/final-p1"]), seeds=tuple(int(x) for x in sys.argv[2].split(",")) if len(sys.argv) > 2 else (17, 23, 29))

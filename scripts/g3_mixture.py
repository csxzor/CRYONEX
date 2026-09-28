"""G3 fallback: does the world model *add* information to the recent-history predictor?

p_mix = lam * p_worldmodel + (1 - lam) * p_recent_history, with lam chosen per horizon on the
calibration split (never on test). If p_mix beats p_recent_history on test with a block-
bootstrap CI above zero, the world model's learned dynamics carry information that the recent
average does not. Output: results/final-p1/g3_mixture.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from g3_dynamics import arrays_for  # noqa: E402

from kcwm.data.sequences import context_index  # noqa: E402
from kcwm.eval.dynamics import (  # noqa: E402
    _feature_mask,
    _nll_from_probs,
    context_hist_probs,
    world_model_probs,
)
from kcwm.eval.metrics import time_blocks  # noqa: E402
from kcwm.eval.run import assemble  # noqa: E402
from kcwm.inference.engine import load_bundle  # noqa: E402

LAMS = np.linspace(0, 1, 21)
HORIZONS = (1, 6, 30)


def nlls(b, arr, rows, k):
    ok = arr.remaining[rows] >= k
    r = rows[ok]
    wm = world_model_probs(b.model, arr, r, b.context, [k])[k]
    ch = context_hist_probs(b.binner, arr.bins[context_index(r, b.context)])
    tgt, mask = arr.bins[r + k], _feature_mask(arr, r + k)
    return r, {lam: _nll_from_probs(lam * wm + (1 - lam) * ch, tgt, mask) for lam in LAMS}


def main(d: str = "results/final-p1") -> None:
    rd = Path(d)
    r0 = json.loads((rd / "kcwm-s17.json").read_text())
    win, split, _, _, _ = assemble(protocol="p1", tag="x", datasets=r0["datasets"], horizon=r0["horizon"],
                                   dev=False, campaigns=bool(r0.get("campaigns")), log=lambda *_: None)
    out = {"seeds": {}}
    for seed in (17, 23, 29):
        rng = np.random.default_rng(seed)
        b = load_bundle(json.loads((rd / f"kcwm-s{seed}.json").read_text())["world_model"]["checkpoint"])
        arr = arrays_for(win, b)
        cal = np.sort(rng.choice(split.calibration, 3000, replace=False))
        test = np.sort(rng.choice(split.test, 4000, replace=False))
        res = {}
        for k in HORIZONS:
            _, c = nlls(b, arr, cal, k)
            lam = float(min(LAMS, key=lambda x: c[x].mean()))
            rows, t = nlls(b, arr, test, k)
            gain = t[0.0] - t[lam]
            blocks = time_blocks(arr.session[rows], arr.pos[rows])
            ids, inv = np.unique(blocks, return_inverse=True)
            members = [np.flatnonzero(inv == i) for i in range(ids.size)]
            boots = [gain[np.concatenate([members[j] for j in rng.integers(0, ids.size, ids.size)])].mean()
                     for _ in range(300)]
            res[k] = {"lambda_from_calibration": lam, "nll_recent_history": float(t[0.0].mean()),
                      "nll_world_model_alone": float(t[1.0].mean()), "nll_mixture": float(t[lam].mean()),
                      "gain": float(gain.mean()), "gain_ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
            v = res[k]
            print(f"seed {seed} k={k:2}: lam={lam:.2f}  recent {v['nll_recent_history']:.3f}  WM {v['nll_world_model_alone']:.3f}"
                  f"  mix {v['nll_mixture']:.3f}  gain {v['gain']:+.4f} CI [{v['gain_ci'][0]:+.4f}, {v['gain_ci'][1]:+.4f}]")
        out["seeds"][seed] = {str(k): v for k, v in res.items()}
    (rd / "g3_mixture.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

"""Next-stage prediction at real stage changes: neural head vs a learned transition table.

At every stage onset r (an attack stage absent for the previous 5 minutes; gate G5), the
forecast issued at r - lead is asked: which *new* stage comes next? Predictors:

* ``wm``: the world model's rollout (stage mass over the 5-minute horizon, excluding the
  stage it believes is active now). This is what the dashboard's "heading to" showed.
* ``markov_true``: counts of (current stage, progress) -> next stage on training labels, read
  at the TRUE current stage (an upper-bound control: a live system does not know it).
* ``table``: the same learned table, read at the world model's *own* current-stage belief and
  progress: deployable.
* ``blend_w``: p ∝ wm^w · table^(1-w).

Usage: python scripts/next_stage_eval.py <results dir> [seeds] -> <dir>/next_stage.json
Select on a dev directory (results/dev-r6); then read the test directory (results/final-p1).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from g3_dynamics import arrays_for  # noqa: E402

from kcwm.data.sequences import context_index, model_input  # noqa: E402
from kcwm.eval.run import assemble  # noqa: E402
from kcwm.eval.stages import markov_table  # noqa: E402
from kcwm.inference.engine import load_bundle  # noqa: E402
from kcwm.model.rollout import analytic, estimate_progress  # noqa: E402

WEIGHTS = (0.25, 0.5, 0.75)
LEADS = (1, 6)


@torch.no_grad()
def forecast(b, arr, rows, K):
    sm, now, prog = [], [], []
    for i in range(0, len(rows), 256):
        r = rows[i : i + 256]
        H = b.model.encode(torch.from_numpy(model_input(arr, context_index(r, b.context))))
        b0 = torch.softmax(b.model.nowcast(H[:, -1]), -1)
        p0 = estimate_progress(b.model, H)
        fc = analytic(b.model, H[:, -1], b0, p0, K)
        sm.append(fc.stage_marg.numpy())
        now.append(b0.numpy())
        prog.append(p0.numpy())
    return np.concatenate(sm), np.concatenate(now), np.concatenate(prog)


def table_probs(table: np.ndarray, cur: np.ndarray, prog: np.ndarray) -> np.ndarray:
    """Row of the learned table at (cur, max(prog, cur)); the current stage and Benign excluded."""
    p = table[cur, np.maximum(prog, cur)].copy()
    p[:, 0] = 0
    p[np.arange(len(cur)), cur] = np.where(cur > 0, 0, p[np.arange(len(cur)), cur])
    return p / np.clip(p.sum(1, keepdims=True), 1e-12, None)


def predictions(sm, now, prog, true_cur, table):
    cur = now.argmax(1)
    wm = sm.sum(1).copy()
    wm[:, 0] = 0
    wm[np.arange(len(cur)), cur] = np.where(cur > 0, 0, wm[np.arange(len(cur)), cur])
    wm = wm / np.clip(wm.sum(1, keepdims=True), 1e-12, None)
    tab = table_probs(table, cur, prog)
    out = {"wm": wm, "markov_true": table_probs(table, true_cur, np.maximum(prog, true_cur)), "table": tab}
    for w in WEIGHTS:
        b = np.exp(w * np.log(np.clip(wm, 1e-6, None)) + (1 - w) * np.log(np.clip(tab, 1e-6, None)))
        b[:, 0] = 0
        out[f"blend_{w}"] = b / b.sum(1, keepdims=True)
    return out


def main(d: str, seeds=(17, 23, 29)) -> None:
    rd = Path(d)
    r0 = json.loads((rd / f"kcwm-s{seeds[0]}.json").read_text())
    dev = bool(r0["split_notes"].get("dev"))
    win, split, _, _, cfg = assemble(protocol="p1", tag="x", datasets=r0["datasets"], horizon=r0["horizon"],
                                     dev=dev, campaigns=bool(r0.get("campaigns")), log=lambda *_: None)
    K = int(r0["horizon"])
    onset = win["is_stage_onset"].to_numpy().astype(bool)
    stage = win["stage_now"].to_numpy().astype(int)
    sess = win["session_key"].to_numpy()
    test = np.zeros(win.height, bool)
    test[split.test] = True
    out = {"split": "dev" if dev else "test (post-hoc: read after the frozen test read)", "seeds": {}}
    for seed in seeds:
        b = load_bundle(json.loads((rd / f"kcwm-s{seed}.json").read_text())["world_model"]["checkpoint"])
        arr = arrays_for(win, b)
        table = markov_table(arr.stage, arr.session, onset, split.train, b.context)
        res = {}
        for lead in LEADS:
            r = np.flatnonzero(onset & test)
            a = r - lead
            ok = (a >= 0) & (arr.pos[np.clip(a, 0, None)] >= b.context - 1) & (sess[np.clip(a, 0, None)] == sess[r])
            r, a = r[ok], a[ok]
            sm, now, prog = forecast(b, arr, a, K)
            preds = predictions(sm, now, prog, stage[a], table)
            truth = stage[r]
            res[f"lead_{lead}"] = {"n": int(len(r)), **{k: float((v.argmax(1) == truth).mean()) for k, v in preds.items()},
                                   "current_stage_correct": float((now.argmax(1) == stage[a]).mean())}
        out["seeds"][seed] = res
        print(seed, json.dumps(res))
    agg = {}
    for lead in LEADS:
        keys = [k for k in out["seeds"][seeds[0]][f"lead_{lead}"] if k != "n"]
        agg[f"lead_{lead}"] = {k: float(np.mean([out["seeds"][s][f"lead_{lead}"][k] for s in seeds])) for k in keys}
        agg[f"lead_{lead}"]["n_per_seed"] = out["seeds"][seeds[0]][f"lead_{lead}"]["n"]
    out["mean"] = agg
    print("MEAN", json.dumps(agg, indent=1))
    (rd / "next_stage.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], tuple(int(x) for x in sys.argv[2].split(",")) if len(sys.argv) > 2 else (17, 23, 29))

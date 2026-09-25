"""Counterfactuals and faithfulness.

* ``group_counterfactual``: which feature groups, reset to the baseline over the last
  ``recent`` windows, would bring the risk below the alert threshold? Groups are removed
  greedily, largest drop first. The answer reads "if the half-open SYN surge and the new
  internal connections were normal, risk would fall from 0.87 to 0.21".
* ``deletion_test`` (gate G8): does deleting the top-k attributed features lower the risk
  more than deleting k random features? If not, the explanation is decoration.
"""

from __future__ import annotations

import numpy as np
import torch

from ..features.registry import BY_NAME, FEATURES, GROUPS
from ..model.world_model import KillChainWorldModel
from .attributions import integrated_gradients, risk_fn


@torch.no_grad()
def group_counterfactual(model: KillChainWorldModel, x: np.ndarray, *, horizon: int, n_features: int,
                         threshold: float, recent: int = 12, max_groups: int = 4,
                         names: list[str] | None = None) -> dict:
    f = risk_fn(model, horizon)
    base = torch.from_numpy(x[None].astype(np.float32))
    risk0 = float(f(base)[0])
    feats = FEATURES if names is None else [BY_NAME[n] for n in names]
    cols = {g: [i for i, ft in enumerate(feats) if ft.group == g] for g in GROUPS}
    cur, chosen, path = base.clone(), [], []
    for _ in range(max_groups):
        best = None
        for g, c in cols.items():
            if g in chosen or not c:
                continue
            v = cur.clone()
            v[0, -recent:, c] = 0.0
            r = float(f(v)[0])
            if best is None or r < best[1]:
                best = (g, r, v)
        if best is None:
            break
        chosen.append(best[0])
        cur = best[2]
        path.append({"group": best[0], "risk_after": best[1]})
        if best[1] < threshold:
            break
    return {"risk": risk0, "threshold": threshold, "steps": path,
            "crosses_threshold": bool(path and path[-1]["risk_after"] < threshold)}


def deletion_test(model: KillChainWorldModel, contexts: list[np.ndarray], *, horizon: int, n_features: int,
                  k: int = 5, seed: int = 0) -> dict:
    """Mean risk drop from deleting the top-k IG features vs k random features (all windows)."""
    rng = np.random.default_rng(seed)
    f = risk_fn(model, horizon)
    top_drops, rand_drops = [], []
    for x in contexts:
        att = integrated_gradients(model, x, horizon=horizon, n_features=n_features, steps=16)
        top = np.argsort(-att.per_feature)[:k]  # features pushing risk *up*
        rnd = rng.choice(n_features, size=k, replace=False)
        with torch.no_grad():
            xt = torch.from_numpy(x[None].astype(np.float32))
            r = float(f(xt)[0])
            v1, v2 = xt.clone(), xt.clone()
            v1[0, :, top] = 0.0
            v2[0, :, rnd] = 0.0
            top_drops.append(r - float(f(v1)[0]))
            rand_drops.append(r - float(f(v2)[0]))
    t, rd = float(np.mean(top_drops)), float(np.mean(rand_drops))
    return {"n": len(contexts), "k": k, "mean_drop_top": t, "mean_drop_random": rd,
            "ratio": t / rd if abs(rd) > 1e-9 else float("inf"), "passes_G8": bool(t >= 2 * max(rd, 1e-9))}

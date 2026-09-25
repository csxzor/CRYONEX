"""Gate G3: did the world model learn transition dynamics, or only classification?

v1 scored dynamics by MSE on standardised features. There, "predict the context mean" nearly
matches any model, and a linear autoregression matched v1's. v2 scores the proper
probabilistic question: how much probability does each predictor put on the bin the next
window actually falls in, k steps ahead? Every predictor below is a distribution over the
same quantile bins, so NLLs are directly comparable (nats per observed feature).

Controls (all given the same context, no future):

* ``persistence``: a sharpened distribution on the current window's bin (``S_{t+k} = S_t``);
* ``context_hist``: the empirical bin histogram of the last N windows (Laplace smoothed),
  i.e. the probabilistic "recent average";
* ``ridge_ar``: a linear autoregression on the last 6 windows predicting the next value,
  turned into a bin distribution with a Gaussian whose width is the training residual spread.

The world model's k-step prediction is ``p(x_{t+k} | h_{t+k})`` along its teacher-free
imagination path (analytic mean-field latent), which is exactly what a forecast rests on.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import norm
from sklearn.linear_model import Ridge

from ..data.sequences import Arrays, context_index, model_input
from ..features.normalize import QuantileBinner
from ..features.registry import feature_mask_index
from ..model.rollout import estimate_progress
from ..model.world_model import KillChainWorldModel

EPS = 1e-4


def _feature_mask(arr: Arrays, rows: np.ndarray) -> np.ndarray:
    gate = np.asarray(feature_mask_index())
    fm = np.ones((len(rows), arr.x.shape[1]), dtype=np.float64)
    for g in range(arr.m.shape[1]):
        fm[:, gate == g] = arr.m[rows, g : g + 1]
    return fm


def _nll_from_probs(probs: np.ndarray, bins: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """probs (n, F, B), bins (n, F), mask (n, F) -> (n,) mean NLL over observed features."""
    p = np.take_along_axis(probs, bins[..., None], axis=-1)[..., 0]
    ll = np.log(np.clip(p, EPS, 1.0)) * mask
    return -ll.sum(-1) / np.clip(mask.sum(-1), 1, None)


def persistence_probs(binner: QuantileBinner, cur_bins: np.ndarray, sharp: float = 0.9) -> np.ndarray:
    nb = binner.n_bins_per_feature()
    B = int(nb.max())
    n, F = cur_bins.shape
    probs = np.zeros((n, F, B))
    for j in range(F):
        rest = (1 - sharp) / max(nb[j] - 1, 1)
        probs[:, j, : nb[j]] = rest
        probs[np.arange(n), j, cur_bins[:, j]] = sharp
    return probs


def context_hist_probs(binner: QuantileBinner, ctx_bins: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """ctx_bins (n, N, F) -> Laplace-smoothed histogram (n, F, B)."""
    nb = binner.n_bins_per_feature()
    B = int(nb.max())
    n, N, F = ctx_bins.shape
    probs = np.zeros((n, F, B))
    for b in range(B):
        probs[..., b] = (ctx_bins == b).sum(1)
    valid = np.arange(B)[None, :] < nb[:, None]
    probs = (probs + alpha) * valid[None]
    return probs / probs.sum(-1, keepdims=True)


class RidgeAR:
    """Linear AR(6) on scaled features predicting x_{t+k} for each k; residual-width Gaussians."""

    def __init__(self, lags: int = 6):
        self.lags = lags
        self.models: dict[int, Ridge] = {}
        self.sigma: dict[int, np.ndarray] = {}

    def fit(self, arr: Arrays, anchors: np.ndarray, horizons: list[int]) -> RidgeAR:
        X = arr.x[context_index(anchors, self.lags)].reshape(len(anchors), -1)
        for k in horizons:
            ok = arr.remaining[anchors] >= k
            Y = arr.x[anchors[ok] + k]
            m = Ridge(alpha=1.0).fit(X[ok], Y)
            self.models[k] = m
            self.sigma[k] = np.maximum((Y - m.predict(X[ok])).std(0), 0.05)
        return self

    def probs(self, arr: Arrays, anchors: np.ndarray, k: int, binner: QuantileBinner) -> np.ndarray:
        X = arr.x[context_index(anchors, self.lags)].reshape(len(anchors), -1)
        mu = self.models[k].predict(X)
        nb = binner.n_bins_per_feature()
        B = int(nb.max())
        probs = np.zeros((len(anchors), arr.x.shape[1], B))
        for j, e in enumerate(binner.edges):
            edges = np.concatenate([[-np.inf], e, [np.inf]])
            cdf = norm.cdf((edges[None, :] - mu[:, j : j + 1]) / self.sigma[k][j])
            probs[:, j, : nb[j]] = np.diff(cdf, axis=1)
        return probs


@torch.no_grad()
def world_model_probs(model: KillChainWorldModel, arr: Arrays, anchors: np.ndarray, context: int,
                      horizons: list[int], batch: int = 256) -> dict[int, np.ndarray]:
    """p(x_{t+k} | imagined h_{t+k}) along the free-running mean-field path, for each k."""
    from ..model.rollout import analytic

    K = max(horizons)
    out = {k: [] for k in horizons}
    for i in range(0, len(anchors), batch):
        r = anchors[i : i + batch]
        H = model.encode(torch.from_numpy(model_input(arr, context_index(r, context))))
        b0 = torch.softmax(model.nowcast(H[:, -1]), -1)
        p0 = estimate_progress(model, H)
        # replay the analytic path, keeping latents
        fc = analytic(model, H[:, -1], b0, p0, K)
        h = H[:, -1]
        for k in range(1, K + 1):
            z_m, p_m = fc.stage_marg[:, k - 1], fc.progress_marg[:, k - 1]
            h = model.step(h, z_m @ model.stage_emb.weight, p_m @ model.prog_emb.weight)
            if k in out:
                out[k].append(torch.softmax(model.emission_logits(h), -1).numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def evaluate_dynamics(model: KillChainWorldModel, arr: Arrays, binner: QuantileBinner,
                      train: np.ndarray, test: np.ndarray, *, context: int,
                      horizons: tuple[int, ...] = (1, 6, 30), max_rows: int = 4000,
                      seed: int = 0, blocks_fn=None) -> dict:
    """Per-horizon mean NLL for the world model and every control, on the same test rows."""
    from .metrics import time_blocks

    rng = np.random.default_rng(seed)
    test = np.sort(rng.choice(test, size=min(max_rows, len(test)), replace=False))
    ar = RidgeAR().fit(arr, rng.choice(train, size=min(20_000, len(train)), replace=False), list(horizons))
    wm = world_model_probs(model, arr, test, context, list(horizons))
    ctx_bins = arr.bins[context_index(test, context)]
    ch = context_hist_probs(binner, ctx_bins)
    per = persistence_probs(binner, arr.bins[test])
    out = {}
    for k in horizons:
        ok = arr.remaining[test] >= k
        rows = test[ok]
        tgt = arr.bins[rows + k]
        mask = _feature_mask(arr, rows + k)
        nll = {
            "world_model": _nll_from_probs(wm[k][ok], tgt, mask),
            "context_hist": _nll_from_probs(ch[ok], tgt, mask),
            "persistence": _nll_from_probs(per[ok], tgt, mask),
            "ridge_ar": _nll_from_probs(ar.probs(arr, rows, k, binner), tgt, mask),
        }
        blocks = time_blocks(arr.session[rows], arr.pos[rows])
        gain = nll["context_hist"] - nll["world_model"]
        # block bootstrap CI on the mean gain over the strongest simple control
        ids, inv = np.unique(blocks, return_inverse=True)
        members = [np.flatnonzero(inv == b) for b in range(ids.size)]
        boots = []
        for _ in range(300):
            pick = np.concatenate([members[b] for b in rng.integers(0, ids.size, ids.size)])
            boots.append(gain[pick].mean())
        out[k] = {
            "n": int(ok.sum()),
            **{f"nll_{name}": float(v.mean()) for name, v in nll.items()},
            "gain_vs_context_hist": float(gain.mean()),
            "gain_ci": (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))),
            "gain_vs_ridge_ar": float((nll["ridge_ar"] - nll["world_model"]).mean()),
        }
    return out

"""Why did the forecast say that? Attribution on the world model's own rollout output.

Every method explains ``P_infil(K)``, the analytic rollout's probability that a compromise
window occurs within the horizon. That is the exact number the alert fires on. v1's
attributions explained a separate risk head.

* ``integrated_gradients``: IG over the (context x feature) input against a benign
  baseline. The baseline is the training median in global mode, or the session's own
  warm-up median in warm-up mode; both are the zero vector in scaled space. Completeness
  (attributions sum to f(x) - f(baseline)) is checked in the tests.
* ``attention_rollout``: Abnar & Zuidema (2020) across the causal Transformer's layers, i.e.
  which past windows the belief state draws on.
* ``temporal_occlusion``: reset blocks of past windows to baseline and measure the drop.
  This cross-checks attention, which is not guaranteed to be faithful.
* ``group_attribution``: IG summed per feature group (flags, ports, timing, c2, ...), which
  is what the UI's "what" panel shows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..features.registry import FEATURES, GROUPS
from ..model.rollout import analytic, estimate_progress
from ..model.world_model import KillChainWorldModel


def risk_fn(model: KillChainWorldModel, horizon: int):
    """Differentiable map: input context (B, N, F+M) -> P_infil at ``horizon`` (B,)."""
    def f(x: torch.Tensor) -> torch.Tensor:
        H = model.encode(x)
        b0 = torch.softmax(model.nowcast(H[:, -1]), -1)
        p0 = estimate_progress(model, H)
        return analytic(model, H[:, -1], b0, p0, horizon).p_infil[:, -1]
    return f


@dataclass
class Attribution:
    values: np.ndarray        # (N, F) IG attribution per window x feature
    risk: float               # f(x)
    baseline_risk: float      # f(baseline)
    per_feature: np.ndarray   # (F,)
    per_time: np.ndarray      # (N,)
    per_group: dict[str, float]

    def top_features(self, k: int = 6) -> list[tuple[str, float]]:
        order = np.argsort(-np.abs(self.per_feature))[:k]
        return [(FEATURES[i].name, float(self.per_feature[i])) for i in order]


def integrated_gradients(model: KillChainWorldModel, x: np.ndarray, *, horizon: int,
                         n_features: int, steps: int = 32) -> Attribution:
    """``x``: one context (N, F+M). Mask channels are held fixed; only features are attributed."""
    model.eval()
    f = risk_fn(model, horizon)
    xt = torch.from_numpy(x[None].astype(np.float32))
    base = xt.clone()
    base[..., :n_features] = 0.0
    alphas = torch.linspace(0, 1, steps + 1)[1:]
    path = base + alphas[:, None, None] * (xt - base)          # (steps, N, F+M)
    path.requires_grad_(True)
    out = f(path)
    grads = torch.autograd.grad(out.sum(), path)[0]
    ig = ((xt - base) * grads.mean(0, keepdim=True))[0, :, :n_features].detach().numpy()
    with torch.no_grad():
        r, r0 = float(f(xt)[0]), float(f(base)[0])
    per_feature = ig.sum(0)
    groups = {g: 0.0 for g in GROUPS}
    for i, feat in enumerate(FEATURES):
        groups[feat.group] += float(per_feature[i])
    return Attribution(ig, r, r0, per_feature, ig.sum(1), groups)


@torch.no_grad()
def attention_rollout(model: KillChainWorldModel, x: np.ndarray) -> np.ndarray:
    """(N,) attention of the last position over the context, rolled out across layers."""
    enc = model.encoder
    xt = torch.from_numpy(x[None].astype(np.float32))
    t = xt.shape[1]
    h = enc.embed(xt) + enc.pos[:, -t:]
    mask = torch.triu(torch.full((t, t), float("-inf")), diagonal=1)
    rollout = torch.eye(t)
    for layer in enc.encoder.layers:
        z = layer.norm1(h)
        _, w = layer.self_attn(z, z, z, attn_mask=mask, need_weights=True, average_attn_weights=True)
        a = 0.5 * w[0] + 0.5 * torch.eye(t)
        a = a / a.sum(-1, keepdim=True)
        rollout = a @ rollout
        h = h + layer._sa_block(z, mask, None, is_causal=True)
        h = h + layer._ff_block(layer.norm2(h))
    return rollout[-1].numpy()


@torch.no_grad()
def temporal_occlusion(model: KillChainWorldModel, x: np.ndarray, *, horizon: int, n_features: int,
                       block: int = 4) -> np.ndarray:
    """(N,) drop in P_infil when each block of windows is reset to baseline."""
    f = risk_fn(model, horizon)
    xt = torch.from_numpy(x[None].astype(np.float32))
    ref = float(f(xt)[0])
    N = x.shape[0]
    out = np.zeros(N)
    variants = []
    for s in range(0, N, block):
        v = xt.clone()
        v[0, s : s + block, :n_features] = 0.0
        variants.append(v)
    risks = f(torch.cat(variants)).numpy()
    for j, s in enumerate(range(0, N, block)):
        out[s : s + block] = ref - risks[j]
    return out

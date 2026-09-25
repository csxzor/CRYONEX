"""The Kill-Chain World Model (KC-WM).

World state ``s_t = (h_t, z_t, p_t)``:

* ``h_t``: belief over the recent past. A causal Transformer over the last N windows; its
  attention is the "when" explanation.
* ``z_t``: kill-chain stage (7 ATT&CK phases, ``stages.STAGES``).
* ``p_t``: progress, the furthest stage reached. It updates deterministically as
  ``p' = max(p, z')``.

Learned pieces:

* nowcast ``q(z_t | h_t)``;
* transition ``P(z' | z, p, h) = softmax(logA[z] + logB[p] + W_z h)``. ``A`` and ``B``
  start from an ATT&CK-ordered prior and are regularised toward it, so the kill-chain
  structure is explicit, inspectable, and needs little data to be sensible (v1's tactic
  head, with no such structure, scored 18.7%);
* latent dynamics ``h' = LN(h + MLP[h, emb(z'), emb(p')])``;
* emission ``p(x' | h') = prod_f Categorical(bins_f)``, a full next-state distribution.

Forecasts come from rolling the model forward. ``rollout.analytic`` propagates the joint
belief over (z, p) exactly under a mean-field latent path. It is differentiable, and it is
what the risk loss trains through (the "value-equivalent" coupling that fixes v1's
bottlenecked risk head). ``rollout.monte_carlo`` samples full trajectories.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..baselines.neural import CausalTransformer
from ..stages import N_STAGES, compromise_mask

S = N_STAGES


def prior_transition() -> np.ndarray:
    """ATT&CK-ordered prior over next stage given current stage (rows sum to 1).

    Stages persist over several 10-s windows. Movement is mostly *forward* along the kill
    chain (Recon -> Initial Access -> Lateral / C2 -> Exfiltration / Impact) or back to
    Benign when activity pauses. These are weak starting beliefs; the data moves them.
    """
    B, R, IA, LM, C2, EX, IM = range(S)
    a = np.full((S, S), 1e-4)
    a[B, [B, R, IA, LM, C2, EX, IM]] = [0.975, 0.01, 0.006, 0.001, 0.005, 0.001, 0.002]
    a[R, [B, R, IA, LM, C2, IM]] = [0.08, 0.85, 0.05, 0.005, 0.01, 0.005]
    a[IA, [B, R, IA, LM, C2, EX, IM]] = [0.08, 0.01, 0.84, 0.02, 0.04, 0.005, 0.005]
    a[LM, [B, IA, LM, C2, EX, IM]] = [0.07, 0.01, 0.84, 0.05, 0.02, 0.01]
    a[C2, [B, R, IA, LM, C2, EX, IM]] = [0.05, 0.005, 0.005, 0.02, 0.89, 0.015, 0.015]
    a[EX, [B, LM, C2, EX, IM]] = [0.06, 0.01, 0.03, 0.89, 0.01]
    a[IM, [B, C2, IM]] = [0.08, 0.02, 0.90]
    return a / a.sum(axis=1, keepdims=True)


def prior_progress_bonus() -> np.ndarray:
    """Additive logit bonus ``B[p, z']``: once the adversary has reached stage p, the next
    kill-chain stages become more likely (forward progression), and nothing else changes."""
    b = np.zeros((S, S))
    for p in range(1, S):
        for z in range(p + 1, S):
            b[p, z] = 0.5
    return b


def progress_update_tensor() -> torch.Tensor:
    """E[p, z', p'] = 1 if max(p, z') == p' (the deterministic progress update)."""
    e = torch.zeros(S, S, S)
    for p in range(S):
        for z in range(S):
            e[p, z, max(p, z)] = 1.0
    return e


class KillChainWorldModel(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_masks: int,
        *,
        n_bins: np.ndarray,
        d: int = 128,
        layers: int = 4,
        heads: int = 4,
        ff: int = 256,
        dropout: float = 0.1,
        max_len: int = 64,
        stage_dim: int = 32,
        n_direct: int = 0,
    ):
        super().__init__()
        self.n_features, self.n_masks = n_features, n_masks
        self.max_bins = int(np.max(n_bins))
        self.encoder = CausalTransformer(n_features + n_masks, d=d, layers=layers, heads=heads,
                                         ff=ff, dropout=dropout, max_len=max_len)
        self.d = d
        self.nowcast = nn.Linear(d, S)
        # Fallback F-A (pre-registered): a direct risk head on the belief state, one output per
        # forecast horizon. Its probability is averaged with the rollout's; both come from the
        # same world-model latent. Absent (n_direct=0) unless enabled.
        self.direct_risk = nn.Linear(d, n_direct) if n_direct else None
        self.logA = nn.Parameter(torch.log(torch.tensor(prior_transition(), dtype=torch.float32)))
        self.logB = nn.Parameter(torch.tensor(prior_progress_bonus(), dtype=torch.float32))
        self.register_buffer("logA0", self.logA.detach().clone())
        self.register_buffer("logB0", self.logB.detach().clone())
        self.ctx_trans = nn.Linear(d, S * S)  # W_z h for every source stage z
        nn.init.zeros_(self.ctx_trans.weight)
        nn.init.zeros_(self.ctx_trans.bias)
        self.stage_emb = nn.Embedding(S, stage_dim)
        self.prog_emb = nn.Embedding(S, stage_dim)
        self.dynamics = nn.Sequential(
            nn.Linear(d + 2 * stage_dim, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d)
        )
        self.dyn_norm = nn.LayerNorm(d)
        self.decoder = nn.Sequential(
            nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, n_features * self.max_bins)
        )
        # Bins a feature does not have are masked to -inf.
        valid = torch.arange(self.max_bins)[None, :] < torch.as_tensor(n_bins)[:, None]
        self.register_buffer("bin_mask", torch.where(valid, 0.0, float("-inf")))
        self.register_buffer("comp", torch.as_tensor(compromise_mask()))
        self.register_buffer("E", progress_update_tensor())

    # --- pieces -------------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, F+M) -> (B, T, d)."""
        return self.encoder(x)

    def transition_logits(self, h: torch.Tensor) -> torch.Tensor:
        """Logits over z' for every (z, p): (..., S_z, S_p, S')."""
        ctx = self.ctx_trans(h).view(*h.shape[:-1], S, 1, S)
        return self.logA[:, None, :] + self.logB[None, :, :] + ctx

    def transition_logits_at(self, h: torch.Tensor, z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Logits over z' for given integer (z, p): (..., S')."""
        ctx = self.ctx_trans(h).view(*h.shape[:-1], S, S)
        ctx_z = torch.gather(ctx, -2, z[..., None, None].expand(*z.shape, 1, S)).squeeze(-2)
        return self.logA[z] + self.logB[p] + ctx_z

    def step(self, h: torch.Tensor, z_emb: torch.Tensor, p_emb: torch.Tensor) -> torch.Tensor:
        return self.dyn_norm(h + self.dynamics(torch.cat([h, z_emb, p_emb], dim=-1)))

    def emission_logits(self, h: torch.Tensor) -> torch.Tensor:
        """(..., d) -> (..., F, max_bins) with invalid bins at -inf."""
        out = self.decoder(h).view(*h.shape[:-1], self.n_features, self.max_bins)
        return out + self.bin_mask

    def prior_penalty(self) -> torch.Tensor:
        """KL(A0 || A) per row, plus L2 on B's drift: keep the kill chain recognisable."""
        la = torch.log_softmax(self.logA, dim=-1)
        la0 = torch.log_softmax(self.logA0, dim=-1)
        kl = (la0.exp() * (la0 - la)).sum(-1).mean()
        return kl + 0.1 * (self.logB - self.logB0).pow(2).mean()

    def transition_matrix(self) -> np.ndarray:
        """The context-free kill-chain matrix softmax(logA), for display."""
        return torch.softmax(self.logA.detach(), dim=-1).cpu().numpy()

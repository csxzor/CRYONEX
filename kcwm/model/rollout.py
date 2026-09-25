"""Forward simulation: from the belief at time t, roll the world model K windows ahead.

Two estimators of the same quantity:

* ``analytic``: exact propagation of the joint belief over (stage z, progress p). With
  7 x 7 = 49 discrete states it is cheap, and the latent follows the *expected* stage/progress
  embedding (mean-field). It is differentiable, so the risk loss trains the whole simulator
  through it, and it scores the large evaluation sets.
* ``monte_carlo``: samples M full trajectories (stages, progress and a latent path per sample).
  It gives per-trajectory quantities (first attack stage entered, time to stage), and it is
  the demo's "simulate 256 futures". ``tests/test_rollout.py`` checks that it agrees with
  ``analytic`` in the regime where they must match.

``P_infil(k)`` is the probability that some window in (t, t+k] is in a compromise stage
(Initial Access, Lateral Movement, C2, Exfiltration, Impact). That is exactly target ``y_k``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .world_model import KillChainWorldModel, S


@dataclass
class Forecast:
    p_infil: torch.Tensor        # (B, K) P(any compromise window in (t, t+k])
    stage_marg: torch.Tensor     # (B, K, S) P(z_{t+k} = s)
    progress_marg: torch.Tensor  # (B, K, S) P(p_{t+k} = s)
    belief0: torch.Tensor        # (B, S) nowcast at t
    progress0: torch.Tensor      # (B,) estimated progress at t


def estimate_progress(model: KillChainWorldModel, H: torch.Tensor, *, confidence: float = 0.5) -> torch.Tensor:
    """Furthest stage the nowcast confidently saw anywhere in the context: (B, T, d) -> (B,)."""
    with torch.no_grad():
        q = torch.softmax(model.nowcast(H), dim=-1)            # (B, T, S)
        conf, arg = q.max(dim=-1)
        arg = torch.where(conf >= confidence, arg, torch.zeros_like(arg))
        return arg.max(dim=1).values


def analytic(
    model: KillChainWorldModel,
    h0: torch.Tensor,
    belief0: torch.Tensor,
    progress0: torch.Tensor,
    horizon: int,
) -> Forecast:
    """Mean-field rollout. ``h0`` (B, d); ``belief0`` (B, S) probabilities; ``progress0`` (B,) int."""
    # joint[b, z, q] = belief0[b, z] * 1[q == max(progress0[b], z)]: the current stage also counts
    # as reached, exactly as in ``monte_carlo``.
    onehot_p = torch.nn.functional.one_hot(progress0, S).float()                  # (B, p)
    joint = belief0[:, :, None] * torch.einsum("bp,pzq->bzq", onehot_p, model.E)  # (B, z, p)
    nohit = joint.clone()
    h = h0
    comp = model.comp.float()
    p_infil, stage_m, prog_m = [], [], []
    for _ in range(horizon):
        T = torch.softmax(model.transition_logits(h), dim=-1)            # (B, z, p, z')
        flow = joint[..., None] * T                                       # (B, z, p, z')
        by_pz = flow.sum(dim=1)                                           # (B, p, z')
        joint = torch.einsum("bpz,pzq->bzq", by_pz, model.E)              # (B, z', p')
        nh = (nohit[..., None] * T).sum(dim=1) * (1.0 - comp)[None, None, :]  # (B, p, z') no-hit mass
        nohit = torch.einsum("bpz,pzq->bzq", nh, model.E)
        z_m = joint.sum(dim=2)                                            # (B, S)
        p_m = joint.sum(dim=1)                                            # (B, S)
        p_infil.append(1.0 - nohit.sum(dim=(1, 2)))
        stage_m.append(z_m)
        prog_m.append(p_m)
        h = model.step(h, z_m @ model.stage_emb.weight, p_m @ model.prog_emb.weight)
    return Forecast(
        p_infil=torch.stack(p_infil, dim=1).clamp(0.0, 1.0),
        stage_marg=torch.stack(stage_m, dim=1),
        progress_marg=torch.stack(prog_m, dim=1),
        belief0=belief0, progress0=progress0,
    )


@dataclass
class Trajectories:
    stages: torch.Tensor      # (B, M, K) sampled stage path
    progress: torch.Tensor    # (B, M, K)
    p_infil: torch.Tensor     # (B, K) fraction of samples with a compromise window by k
    first_attack_stage: torch.Tensor  # (B, M) first non-benign stage entered (0 if none)
    first_attack_step: torch.Tensor   # (B, M) step (1-based) it was entered (0 if none)
    latent: torch.Tensor | None       # (B, M, K, d) when requested


@torch.no_grad()
def monte_carlo(
    model: KillChainWorldModel,
    h0: torch.Tensor,
    belief0: torch.Tensor,
    progress0: torch.Tensor,
    horizon: int,
    *,
    samples: int = 256,
    generator: torch.Generator | None = None,
    keep_latent: bool = False,
    mean_field_latent: bool = False,
) -> Trajectories:
    """Sample ``samples`` trajectories per batch row.

    ``mean_field_latent=True`` drives every sample's latent with the expected embeddings,
    which makes the estimator converge to ``analytic`` exactly (the consistency test).
    """
    Bsz, d = h0.shape
    M = samples
    z = torch.multinomial(belief0, M, replacement=True, generator=generator)       # (B, M)
    p = progress0[:, None].expand(Bsz, M).clone()
    p = torch.maximum(p, z)
    h = h0[:, None, :].expand(Bsz, M, d).reshape(Bsz * M, d)
    comp = model.comp
    hit = torch.zeros(Bsz, M, dtype=torch.bool)
    first_stage = torch.zeros(Bsz, M, dtype=torch.long)
    first_step = torch.zeros(Bsz, M, dtype=torch.long)
    zs, ps, lat, pinf = [], [], [], []
    h_mf = h0
    for k in range(horizon):
        logits = model.transition_logits_at(h, z.reshape(-1), p.reshape(-1))   # (B*M, S)
        z = torch.multinomial(torch.softmax(logits, -1), 1, generator=generator).view(Bsz, M)
        p = torch.maximum(p, z)
        hit |= comp[z]
        new_attack = (first_stage == 0) & (z > 0)
        first_stage = torch.where(new_attack, z, first_stage)
        first_step = torch.where(new_attack, torch.full_like(first_step, k + 1), first_step)
        if mean_field_latent:
            zm = torch.nn.functional.one_hot(z, S).float().mean(1)
            pm = torch.nn.functional.one_hot(p, S).float().mean(1)
            h_mf = model.step(h_mf, zm @ model.stage_emb.weight, pm @ model.prog_emb.weight)
            h = h_mf[:, None, :].expand(Bsz, M, d).reshape(Bsz * M, d)
        else:
            h = model.step(h, model.stage_emb(z.reshape(-1)), model.prog_emb(p.reshape(-1)))
        zs.append(z)
        ps.append(p)
        pinf.append(hit.float().mean(1))
        if keep_latent:
            lat.append(h.view(Bsz, M, d))
    return Trajectories(
        stages=torch.stack(zs, -1), progress=torch.stack(ps, -1),
        p_infil=torch.stack(pinf, -1), first_attack_stage=first_stage,
        first_attack_step=first_step,
        latent=torch.stack(lat, 2) if keep_latent else None,
    )


def one_step_nll(model: KillChainWorldModel, h_prev: torch.Tensor, stage_prev: torch.Tensor,
                 progress_prev: torch.Tensor, bins_next: torch.Tensor, feat_mask: torch.Tensor) -> torch.Tensor:
    """Surprise: -log p(x_t | h_{t-1}), marginalised over the unknown next stage.

    ``h_prev`` (B, d); ``stage_prev``/``progress_prev`` (B,) int; ``bins_next`` (B, F) int;
    ``feat_mask`` (B, F) 1 where the feature was observed. Returns (B,) NLL per observed feature.
    """
    Bsz = h_prev.shape[0]
    logT = torch.log_softmax(model.transition_logits_at(h_prev, stage_prev, progress_prev), -1)  # (B, S)
    zs = torch.arange(S).repeat(Bsz)                                   # (B*S,)
    ps = torch.maximum(progress_prev.repeat_interleave(S), zs)
    hs = h_prev.repeat_interleave(S, dim=0)
    h1 = model.step(hs, model.stage_emb(zs), model.prog_emb(ps))
    logp = torch.log_softmax(model.emission_logits(h1), -1)            # (B*S, F, bins)
    tgt = bins_next.repeat_interleave(S, dim=0)[..., None]
    ll = torch.gather(logp, -1, tgt).squeeze(-1)                       # (B*S, F)
    ll = (ll * feat_mask.repeat_interleave(S, dim=0)).sum(-1).view(Bsz, S)
    total = torch.logsumexp(ll + logT, dim=-1)
    return -total / feat_mask.sum(-1).clamp(min=1.0)

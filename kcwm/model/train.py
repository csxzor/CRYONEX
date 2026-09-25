"""Training the Kill-Chain World Model.

One batch = B anchors. For each anchor: the N-window context (model input), the stage labels
over the context, and the next U windows (stages, bins and validity).

Loss terms (weights in ``TrainConfig``):

1. ``now``: nowcast cross-entropy at every context position (class-balanced).
2. ``trans``: teacher-forced transition cross-entropy, z_{tau+1} from (h_tau, z_tau, p_tau).
3. ``emit``: one-step next-state NLL at sampled context positions,
   ``p(x_{tau+1} | step(h_tau, z_{tau+1}, p_{tau+1}))``. This is the dynamics objective,
   and it is also what the surprise score reads.
4. ``imag``: an imagination unroll from h_t over U future windows, with scheduled sampling
   of stages. It applies emission, nowcast and transition losses on the *imagined* latents,
   so the heads stay valid K steps into the simulation.
5. ``risk``: BCE of the analytic rollout's P_infil(k) against y_k, for k in horizons. This
   free-running path uses only the model's own beliefs (no future labels). It is the
   rollout-trained risk that targets v1's main failure.
6. ``prior``: keeps the transition matrix recognisably ATT&CK-shaped.

Model selection: calibration-split AUPRC of P_infil at the primary horizon, with the
calibration loss as tie-break. The test split is never touched here.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn

from ..data.sequences import Arrays, context_index, model_input
from ..features.registry import feature_mask_index
from .rollout import analytic, estimate_progress
from .world_model import KillChainWorldModel, S


@dataclass
class TrainConfig:
    context: int = 64
    unroll: int = 30
    horizons: tuple[int, ...] = (6, 12, 30)
    primary_horizon: int = 30
    d: int = 128
    layers: int = 4
    heads: int = 4
    ff: int = 256
    dropout: float = 0.1
    batch: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.01
    epochs: int = 25
    anchors_per_epoch: int = 24_000
    patience: int = 5
    threads: int = 6
    seed: int = 17
    w_now: float = 1.0
    w_trans: float = 1.0
    w_emit: float = 1.0
    w_imag: float = 1.0
    w_risk: float = 2.0
    w_prior: float = 0.05
    emit_positions: int = 16     # context positions per sequence for the one-step emission loss
    imag_steps: int = 10         # unroll steps (of `unroll`) that get the emission loss
    scheduled_sampling_max: float = 0.5
    group_dropout: float = 0.3   # randomly hide the packet group so CSV-only inputs work
    risk_pos_weight: str = "sqrt"  # none | sqrt | balanced
    stop_on_train_tail: int = 0  # rejected in dev round 3; kept for the record
    direct_risk: int = 0          # 1 = fallback F-A: add a direct risk head, average with rollout


class Batcher:
    """Assembles training tensors for a set of anchors from the shared per-row arrays."""

    def __init__(self, arr: Arrays, cfg: TrainConfig):
        if arr.bins is None:
            raise ValueError("Arrays need quantile bins for world-model training")
        self.arr, self.cfg = arr, cfg
        gate = np.asarray(feature_mask_index())
        self.gate = gate
        # per-row per-feature observation mask (1 = observed)
        fm = np.ones((arr.x.shape[0], arr.x.shape[1]), dtype=np.float32)
        for g in range(arr.m.shape[1]):
            fm[:, gate == g] = arr.m[:, g : g + 1]
        self.feat_mask = fm

    def __call__(self, anchors: np.ndarray, *, train: bool, rng: np.random.Generator | None = None):
        a, cfg = self.arr, self.cfg
        N, U = cfg.context, cfg.unroll
        ctx = context_index(anchors, N)
        x = model_input(a, ctx)                                     # (B, N, F+M)
        if train and cfg.group_dropout > 0 and rng is not None:
            drop = rng.random(len(anchors)) < cfg.group_dropout
            if drop.any():
                F = a.x.shape[1]
                cols = np.flatnonzero(self.gate == 0)
                x[np.ix_(drop, np.arange(N), cols)] = 0.0
                x[drop, :, F + 0] = 0.0                               # m_packet channel
        fut = anchors[:, None] + np.arange(1, U + 1)[None, :]
        fut_valid = (np.arange(1, U + 1)[None, :] <= a.remaining[anchors][:, None])
        fut_c = np.clip(fut, 0, a.x.shape[0] - 1)
        stage_ctx = a.stage[ctx]
        prog_ctx = np.maximum.accumulate(stage_ctx, axis=1)
        out = {
            "x": torch.from_numpy(x),
            "stage_ctx": torch.from_numpy(stage_ctx),
            "prog_ctx": torch.from_numpy(prog_ctx),
            "bins_ctx": torch.from_numpy(a.bins[ctx]),
            "fmask_ctx": torch.from_numpy(self.feat_mask[ctx]),
            "stage_fut": torch.from_numpy(np.where(fut_valid, a.stage[fut_c], 0)),
            "bins_fut": torch.from_numpy(a.bins[fut_c]),
            "fmask_fut": torch.from_numpy(self.feat_mask[fut_c] * fut_valid[..., None]),
            "fut_valid": torch.from_numpy(fut_valid),
        }
        for k in cfg.horizons:
            out[f"y_{k}"] = torch.from_numpy(a.y[k][anchors].astype(np.float32))
            out[f"v_{k}"] = torch.from_numpy(a.valid[k][anchors].astype(np.float32))
        return out


def stage_weights(stage: np.ndarray) -> torch.Tensor:
    counts = np.bincount(stage, minlength=S).astype(np.float64) + 1.0
    w = counts ** -0.5
    return torch.tensor(w / w[stage].mean(), dtype=torch.float32)


def emission_nll(model: KillChainWorldModel, h: torch.Tensor, bins: torch.Tensor, fmask: torch.Tensor) -> torch.Tensor:
    """Mean per-observed-feature NLL of bins under p(x | h). Shapes (..., d), (..., F), (..., F)."""
    logp = torch.log_softmax(model.emission_logits(h), dim=-1)
    ll = torch.gather(logp, -1, bins[..., None]).squeeze(-1)
    return -(ll * fmask).sum() / fmask.sum().clamp(min=1.0)


def compute_losses(model: KillChainWorldModel, b: dict, cfg: TrainConfig, sw: torch.Tensor,
                   *, ss_prob: float, pos_weight: dict[int, float], gen: torch.Generator | None) -> dict:
    H = model.encode(b["x"])                                          # (B, N, d)
    Bsz, N, _ = H.shape
    z, p = b["stage_ctx"], b["prog_ctx"]
    ce = nn.functional.cross_entropy
    losses = {}

    losses["now"] = ce(model.nowcast(H).reshape(-1, S), z.reshape(-1), weight=sw)

    # transition inside the context, plus the step into the first future window
    z_next = torch.cat([z[:, 1:], b["stage_fut"][:, :1]], dim=1)
    tv = torch.cat([torch.ones(Bsz, N - 1, dtype=torch.bool), b["fut_valid"][:, :1]], dim=1)
    tl = model.transition_logits_at(H, z, p)
    losses["trans"] = (ce(tl.reshape(-1, S), z_next.reshape(-1), weight=sw, reduction="none")
                       * tv.reshape(-1).float()).sum() / tv.float().sum()

    # one-step emission at sampled context positions (tau -> tau+1 inside the context)
    pos = torch.randint(0, N - 1, (cfg.emit_positions,), generator=gen)
    hs = H[:, pos]
    h1 = model.step(hs, model.stage_emb(z[:, pos + 1]), model.prog_emb(torch.maximum(p[:, pos], z[:, pos + 1])))
    losses["emit"] = emission_nll(model, h1, b["bins_ctx"][:, pos + 1], b["fmask_ctx"][:, pos + 1])

    # imagination unroll with scheduled sampling
    h = H[:, -1]
    zc, pc = z[:, -1], p[:, -1]
    fut_z, fut_v = b["stage_fut"], b["fut_valid"]
    emit_steps = set(torch.randperm(cfg.unroll, generator=gen)[: cfg.imag_steps].tolist())
    imag_now, imag_trans, imag_emit = [], [], []
    for k in range(cfg.unroll):
        logits = model.transition_logits_at(h, zc, pc)
        if k > 0:  # the k=0 transition is already in `trans`
            imag_trans.append((ce(logits, fut_z[:, k], weight=sw, reduction="none"), fut_v[:, k]))
        zt = fut_z[:, k]
        if ss_prob > 0:
            sampled = torch.multinomial(torch.softmax(logits.detach(), -1), 1, generator=gen).squeeze(-1)
            use = torch.rand(Bsz, generator=gen) < ss_prob
            zt = torch.where(use, sampled, zt)
        pc = torch.maximum(pc, zt)
        h = model.step(h, model.stage_emb(zt), model.prog_emb(pc))
        zc = zt
        imag_now.append((ce(model.nowcast(h), fut_z[:, k], weight=sw, reduction="none"), fut_v[:, k]))
        if k in emit_steps:
            imag_emit.append(emission_nll(model, h, b["bins_fut"][:, k], b["fmask_fut"][:, k]))

    def masked_mean(pairs):
        num = sum((loss * v.float()).sum() for loss, v in pairs)
        den = sum(v.float().sum() for _, v in pairs).clamp(min=1.0)
        return num / den

    losses["imag"] = masked_mean(imag_now) + masked_mean(imag_trans) + torch.stack(imag_emit).mean()

    # rollout-trained risk: free-running, the model's own beliefs only
    belief0 = torch.softmax(model.nowcast(H[:, -1]), dim=-1)
    prog0 = estimate_progress(model, H)
    fc = analytic(model, H[:, -1], belief0, prog0, max(cfg.horizons))
    risk = 0.0
    for k in cfg.horizons:
        pk = fc.p_infil[:, k - 1].clamp(1e-5, 1 - 1e-5)
        y, v = b[f"y_{k}"], b[f"v_{k}"]
        w = torch.where(y > 0, torch.full_like(y, pos_weight[k]), torch.ones_like(y)) * v
        risk = risk + (nn.functional.binary_cross_entropy(pk, y, reduction="none") * w).sum() / w.sum().clamp(min=1.0)
    losses["risk"] = risk / len(cfg.horizons)
    if model.direct_risk is not None:
        logits = model.direct_risk(H[:, -1])
        dr = 0.0
        for j, k in enumerate(cfg.horizons):
            y, v = b[f"y_{k}"], b[f"v_{k}"]
            w = torch.where(y > 0, torch.full_like(y, pos_weight[k]), torch.ones_like(y)) * v
            dr = dr + (nn.functional.binary_cross_entropy_with_logits(logits[:, j], y, reduction="none") * w).sum() / w.sum().clamp(min=1.0)
        losses["risk"] = losses["risk"] + dr / len(cfg.horizons)
    losses["prior"] = model.prior_penalty()
    losses["total"] = (cfg.w_now * losses["now"] + cfg.w_trans * losses["trans"] + cfg.w_emit * losses["emit"]
                       + cfg.w_imag * losses["imag"] + cfg.w_risk * losses["risk"] + cfg.w_prior * losses["prior"])
    return losses


@torch.no_grad()
def forecast_rows(model: KillChainWorldModel, arr: Arrays, rows: np.ndarray, cfg: TrainConfig,
                  *, horizon: int | None = None, batch: int = 512) -> dict[str, np.ndarray]:
    """Analytic forecast for anchor rows: P_infil (n, K), stage marginals (n, K, S), nowcast (n, S)."""
    model.eval()
    K = horizon or max(cfg.horizons)
    pin, sm, now, prog = [], [], [], []
    for i in range(0, len(rows), batch):
        r = rows[i : i + batch]
        x = torch.from_numpy(model_input(arr, context_index(r, cfg.context)))
        H = model.encode(x)
        b0 = torch.softmax(model.nowcast(H[:, -1]), -1)
        p0 = estimate_progress(model, H)
        fc = analytic(model, H[:, -1], b0, p0, K)
        p_roll = fc.p_infil
        if model.direct_risk is not None:
            # average the direct head with the rollout at the trained horizons; scale the rest
            # of the curve so it stays monotone and passes through the averaged points
            d = torch.sigmoid(model.direct_risk(H[:, -1]))
            j = list(cfg.horizons).index(cfg.primary_horizon)
            ratio = (0.5 * (p_roll[:, K - 1] + d[:, j])) / p_roll[:, K - 1].clamp(min=1e-6)
            p_roll = (p_roll * ratio[:, None]).clamp(0.0, 1.0)
        pin.append(p_roll.numpy())
        sm.append(fc.stage_marg.numpy())
        now.append(b0.numpy())
        prog.append(p0.numpy())
    cat = (lambda xs: np.concatenate(xs) if xs else np.zeros(0))
    return {"p_infil": cat(pin), "stage_marg": cat(sm), "nowcast": cat(now), "progress": cat(prog)}


def fit(arr: Arrays, train: np.ndarray, cal: np.ndarray, cfg: TrainConfig, *, n_bins: np.ndarray,
        log=print) -> tuple[KillChainWorldModel, dict]:
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.threads)
    rng = np.random.default_rng(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    model = KillChainWorldModel(arr.n_features, arr.m.shape[1], n_bins=n_bins, d=cfg.d,
                                layers=cfg.layers, heads=cfg.heads, ff=cfg.ff,
                                dropout=cfg.dropout, max_len=cfg.context,
                                n_direct=len(cfg.horizons) if cfg.direct_risk else 0)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    batcher = Batcher(arr, cfg)
    sw = stage_weights(arr.stage[train])
    pos_weight = {}
    for k in cfg.horizons:
        v = arr.valid[k][train]
        pr = float(arr.y[k][train][v].mean()) if v.any() else 0.5
        ratio = (1 - pr) / max(pr, 1e-6)
        pos_weight[k] = {"none": 1.0, "sqrt": math.sqrt(ratio), "balanced": ratio}[cfg.risk_pos_weight]
    Kp = cfg.primary_horizon
    cal_v = cal[arr.valid[Kp][cal]]
    history, best, best_state, stale = [], -1.0, None, 0
    n_params = sum(p.numel() for p in model.parameters())
    log(f"  KC-WM {n_params / 1e6:.2f}M params; train anchors {len(train)}, cal {len(cal_v)}; pos_weight {pos_weight}")
    for epoch in range(cfg.epochs):
        model.train()
        t0 = time.time()
        pick = rng.choice(train, size=min(cfg.anchors_per_epoch, len(train)), replace=False)
        ss = cfg.scheduled_sampling_max * min(1.0, epoch / max(1, cfg.epochs // 2))
        agg: dict[str, float] = {}
        nb = 0
        for i in range(0, len(pick), cfg.batch):
            b = batcher(pick[i : i + cfg.batch], train=True, rng=rng)
            losses = compute_losses(model, b, cfg, sw, ss_prob=ss, pos_weight=pos_weight, gen=gen)
            opt.zero_grad()
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for k, v in losses.items():
                agg[k] = agg.get(k, 0.0) + float(v.detach())
            nb += 1
        fc = forecast_rows(model, arr, cal_v, cfg, horizon=Kp)
        y = arr.y[Kp][cal_v]
        auprc = float(average_precision_score(y, fc["p_infil"][:, Kp - 1])) if 0 < y.sum() < len(y) else float("nan")
        rec = {"epoch": epoch + 1, **{k: v / nb for k, v in agg.items()}, "cal_auprc": auprc,
               "seconds": round(time.time() - t0, 1), "ss": round(ss, 3)}
        history.append(rec)
        log(f"    epoch {epoch + 1:2d} total {rec['total']:.3f} now {rec['now']:.3f} trans {rec['trans']:.3f} "
            f"emit {rec['emit']:.3f} imag {rec['imag']:.3f} risk {rec['risk']:.3f} | cal AUPRC {auprc:.3f} ({rec['seconds']:.0f}s)")
        if auprc > best + 1e-4:
            best, stale = auprc, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, {"history": history, "best_cal_auprc": best, "config": asdict(cfg), "params": n_params}

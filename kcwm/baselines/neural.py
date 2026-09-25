"""Sequence-classifier baselines: LSTM (ported from v1) and a Transformer classifier.

The Transformer classifier uses the **same backbone as the world model**, with a direct risk
head and no dynamics, stage structure or rollout. It is the control that answers "does
learning the dynamics help?" (gate G4). The LSTM is v1's strongest competitor, which beat
v1's world model at risk ranking (AUPRC 0.868 vs 0.768).
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
from torch import nn

from ..data.sequences import Arrays, context_index, model_input


class LSTMClassifier(nn.Module):
    def __init__(self, n_in: int, hidden: int = 128, layers: int = 2, dropout: float = 0.2, n_out: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden // 2, n_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


class CausalTransformer(nn.Module):
    """Pre-LN causal Transformer encoder over a window sequence. Shared with the world model."""

    def __init__(self, n_in: int, d: int = 128, layers: int = 4, heads: int = 4,
                 ff: int = 256, dropout: float = 0.1, max_len: int = 128):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(n_in, d), nn.GELU(), nn.Linear(d, d))
        self.pos = nn.Parameter(torch.zeros(1, max_len, d))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.d = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: (B, T, n_in) -> (B, T, d) causal hidden states."""
        t = x.shape[1]
        h = self.embed(x) + self.pos[:, -t:]
        mask = torch.triu(torch.full((t, t), float("-inf"), device=x.device), diagonal=1)
        return self.norm(self.encoder(h, mask=mask, is_causal=True))


class TransformerClassifier(nn.Module):
    def __init__(self, n_in: int, n_out: int = 1, **kw):
        super().__init__()
        self.backbone = CausalTransformer(n_in, **kw)
        self.head = nn.Linear(self.backbone.d, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x)[:, -1])


def _batches(n: int, size: int, rng: np.random.Generator | None):
    order = rng.permutation(n) if rng is not None else np.arange(n)
    for i in range(0, n, size):
        yield order[i : i + size]


def fit_sequence_classifier(
    kind: str,
    arr: Arrays,
    train: np.ndarray,
    y_train: np.ndarray,
    cal: np.ndarray,
    y_cal: np.ndarray,
    *,
    context: int,
    seed: int,
    epochs: int = 20,
    batch: int = 128,
    lr: float = 1e-3,
    patience: int = 4,
    threads: int = 6,
    log=print,
) -> nn.Module:
    """Train with BCE (positive-class weight), early stopping on calibration loss."""
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    rng = np.random.default_rng(seed)
    n_in = arr.n_features + arr.m.shape[1]
    model = LSTMClassifier(n_in) if kind == "lstm" else TransformerClassifier(n_in, max_len=context)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    pos = float(y_train.sum())
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(y_train) - pos) / max(pos, 1.0)))
    cal_x = torch.from_numpy(model_input(arr, context_index(cal, context)))
    cal_y = torch.from_numpy(y_cal.astype(np.float32))
    best, best_state, stale = math.inf, None, 0
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        total = 0.0
        for b in _batches(len(train), batch, rng):
            xb = torch.from_numpy(model_input(arr, context_index(train[b], context)))
            yb = torch.from_numpy(y_train[b].astype(np.float32))
            opt.zero_grad()
            loss = crit(model(xb).squeeze(-1), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            val = sum(crit(model(cal_x[i:i + 512]).squeeze(-1), cal_y[i:i + 512]).item() * len(cal_y[i:i + 512])
                      for i in range(0, len(cal_y), 512)) / max(len(cal_y), 1)
        log(f"    {kind} epoch {epoch + 1:2d} train {total / len(train):.4f} cal {val:.4f} ({time.time() - t0:.0f}s)")
        if val < best - 1e-4:
            best, stale = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def predict_sequence_classifier(model: nn.Module, arr: Arrays, rows: np.ndarray, *, context: int) -> np.ndarray:
    out = []
    for i in range(0, len(rows), 512):
        xb = torch.from_numpy(model_input(arr, context_index(rows[i:i + 512], context)))
        out.append(torch.sigmoid(model(xb).squeeze(-1)).numpy())
    return np.concatenate(out) if out else np.zeros(0)

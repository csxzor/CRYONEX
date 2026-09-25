"""Feature scaling and quantile binning, fit on training rows only.

Three rules, each of which fixes a failure v1 measured:

1. **Statistics come from training rows alone** (``Split.fit_rows``), and they are saved to
   JSON and reloaded verbatim at evaluation and inference time.
2. **Robust, log-compressed scaling.** Heavy-tailed counts get ``sign(x)*log1p(|x|)``, then
   ``(x - median) / scale``, where scale is the IQR (falling back to the standard deviation
   for sparse features), clipped to +/-10. A DDoS window cannot own the scale.
3. **Label-free warm-up re-centering for a new network** (``recenter_by_session``). v1's
   cross-dataset failure was a 20-40x level shift (CTU-13 ``bytes_rate``) that saturated
   every feature under the source scaler. After log compression a multiplicative shift is
   additive, so subtracting each session's own warm-up median (its first 15 minutes, no
   labels) removes it. The global scale is kept, because 15 minutes are too few to estimate
   spread. Both modes are always reported, so the adaptation's effect is measured, not
   assumed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .registry import FEATURE_NAMES, LOG_FEATURES

CLIP = 10.0
EPS = 1e-9


def compress(x: np.ndarray, log_mask: np.ndarray) -> np.ndarray:
    out = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0).copy()
    out[..., log_mask] = np.sign(out[..., log_mask]) * np.log1p(np.abs(out[..., log_mask]))
    return out


@dataclass
class RobustScaler:
    names: list[str]
    log_mask: np.ndarray
    center: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, names: list[str] | None = None) -> RobustScaler:
        names = list(names or FEATURE_NAMES)
        log_mask = np.array([n in LOG_FEATURES for n in names], dtype=bool)
        c = compress(x, log_mask)
        center = np.median(c, axis=0)
        q75, q25 = np.percentile(c, [75, 25], axis=0)
        scale = (q75 - q25) / 1.349  # IQR of a standard normal
        std = c.std(axis=0)
        scale = np.where(scale > EPS, scale, std)
        scale = np.where(scale > EPS, scale, 1.0)
        return cls(names, log_mask, center, scale)

    def transform(self, x: np.ndarray, *, center: np.ndarray | None = None) -> np.ndarray:
        c = compress(x, self.log_mask)
        z = (c - (self.center if center is None else center)) / self.scale
        return np.clip(z, -CLIP, CLIP).astype(np.float32)

    def inverse(self, z: np.ndarray, *, center: np.ndarray | None = None) -> np.ndarray:
        """Back to native units, for explanations ("SYN 41/s vs baseline 3/s")."""
        c = np.asarray(z, dtype=np.float64) * self.scale + (self.center if center is None else center)
        out = c.copy()
        out[..., self.log_mask] = np.sign(c[..., self.log_mask]) * np.expm1(np.abs(c[..., self.log_mask]))
        return out

    def warmup_centers(self, x: np.ndarray, session: np.ndarray, warmup: np.ndarray) -> np.ndarray:
        """Per-row centers: the median of each session's warm-up windows in compressed space."""
        c = compress(x, self.log_mask)
        centers = np.tile(self.center, (x.shape[0], 1))
        for s in np.unique(session):
            rows = np.flatnonzero(session == s)
            wu = rows[warmup[rows]]
            if wu.size >= 10:
                centers[rows] = np.median(c[wu], axis=0)
        return centers

    def to_dict(self) -> dict:
        return {"names": self.names, "log": [n for n, m in zip(self.names, self.log_mask) if m],
                "center": self.center.tolist(), "scale": self.scale.tolist(), "clip": CLIP}

    @classmethod
    def from_dict(cls, d: dict) -> RobustScaler:
        names = list(d["names"])
        log = set(d["log"])
        return cls(names, np.array([n in log for n in names]), np.asarray(d["center"]),
                   np.asarray(d["scale"]))


def transform_windows(
    scaler: RobustScaler, x: np.ndarray, *, mode: str, session: np.ndarray | None = None,
    warmup: np.ndarray | None = None,
) -> np.ndarray:
    """``mode="global"``: training-row statistics. ``mode="warmup"``: per-session re-centering."""
    if mode == "global":
        return scaler.transform(x)
    if mode == "warmup":
        centers = scaler.warmup_centers(x, session, warmup)
        c = compress(x, scaler.log_mask)
        return np.clip((c - centers) / scaler.scale, -CLIP, CLIP).astype(np.float32)
    raise ValueError(mode)


@dataclass
class QuantileBinner:
    """Per-feature quantile bins: the world model's next-state targets.

    A categorical over bins is a full, multi-modal predictive distribution. It handles the
    zero-inflated, heavy-tailed shape of traffic features without a Gaussian assumption, and
    its negative log-likelihood is the proper score for G3 (unlike v1's MSE, which the
    context mean nearly matched).
    """

    edges: list[np.ndarray]  # per feature, interior edges (len <= n_bins - 1)
    centers: list[np.ndarray]
    n_bins: int

    @classmethod
    def fit(cls, z: np.ndarray, n_bins: int = 16) -> QuantileBinner:
        edges, centers = [], []
        qs = np.linspace(0, 1, n_bins + 1)[1:-1]
        for j in range(z.shape[1]):
            col = z[:, j]
            e = np.unique(np.quantile(col, qs))
            edges.append(e)
            full = np.concatenate([[col.min() - 1e-6], e, [col.max() + 1e-6]])
            idx = np.searchsorted(e, col, side="right")
            ctr = np.array([
                np.median(col[idx == b]) if np.any(idx == b) else 0.5 * (full[b] + full[b + 1])
                for b in range(e.size + 1)
            ])
            centers.append(ctr)
        return cls(edges, centers, n_bins)

    def transform(self, z: np.ndarray) -> np.ndarray:
        out = np.empty(z.shape, dtype=np.int64)
        for j, e in enumerate(self.edges):
            out[..., j] = np.searchsorted(e, z[..., j], side="right")
        return out

    def n_bins_per_feature(self) -> np.ndarray:
        return np.array([e.size + 1 for e in self.edges])

    def to_dict(self) -> dict:
        return {"n_bins": self.n_bins, "edges": [e.tolist() for e in self.edges],
                "centers": [c.tolist() for c in self.centers]}

    @classmethod
    def from_dict(cls, d: dict) -> QuantileBinner:
        return cls([np.asarray(e) for e in d["edges"]], [np.asarray(c) for c in d["centers"]],
                   int(d["n_bins"]))


def save_json(obj: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))

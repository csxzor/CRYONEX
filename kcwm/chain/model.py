"""Chain models: per-host-window stage nowcast, next-stage forecaster, time-to-next-stage hazards.

All three are gradient-boosted trees: the data are a few hundred real chain steps, too few
for a neural sequence model, and trees read the tracker's "minutes since stage s" features
without any scaling. The ATT&CK prior enters as input features (``prior1..6``), so the model
can lean on it where the data are thin and override it where the data disagree.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from ..stages import N_STAGES

ATTACK_CLASSES = np.arange(1, N_STAGES)


def _hgb(seed: int, **kw) -> HistGradientBoostingClassifier:
    params = dict(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=40,
                  l2_regularization=1.0, early_stopping=True, validation_fraction=0.15, n_iter_no_change=20,
                  random_state=seed)
    params.update(kw)
    return HistGradientBoostingClassifier(**params)


def _full_proba(model, X: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """predict_proba with a column for every class in ``classes`` (absent ones get 0)."""
    out = np.zeros((X.shape[0], classes.size), dtype=np.float64)
    if model is None:
        return out
    p = model.predict_proba(X)
    idx = {c: i for i, c in enumerate(classes)}
    for j, c in enumerate(model.classes_):
        out[:, idx[int(c)]] = p[:, j]
    return out


def balanced_weights(y: np.ndarray) -> np.ndarray:
    cls, cnt = np.unique(y, return_counts=True)
    w = {c: (len(y) / (len(cls) * n)) ** 0.5 for c, n in zip(cls, cnt)}
    return np.array([w[v] for v in y])


def fit_nowcast(X: np.ndarray, y: np.ndarray, *, seed: int = 17, benign_keep: int = 400_000):
    rng = np.random.default_rng(seed)
    ben = np.flatnonzero(y == 0)
    att = np.flatnonzero(y > 0)
    if ben.size > benign_keep:
        ben = rng.choice(ben, benign_keep, replace=False)
    idx = np.concatenate([att, ben])
    m = _hgb(seed)
    m.fit(X[idx], y[idx], sample_weight=balanced_weights(y[idx]))
    return m


def predict_nowcast(model, X: np.ndarray) -> np.ndarray:
    return _full_proba(model, X, np.arange(N_STAGES))


@dataclass
class Forecaster:
    next_stage: object = None
    hazards: dict = field(default_factory=dict)
    feature_names: list = field(default_factory=list)

    def predict_next(self, X: np.ndarray) -> np.ndarray:
        """(n, 6): P(next different stage = s | a transition comes), s = 1..6."""
        p = _full_proba(self.next_stage, X, ATTACK_CLASSES)
        return p / np.clip(p.sum(1, keepdims=True), 1e-12, None)

    def predict_hazard(self, X: np.ndarray) -> dict[str, np.ndarray]:
        return {h: (m.predict_proba(X)[:, list(m.classes_).index(1)] if m is not None and 1 in m.classes_
                    else np.zeros(len(X))) for h, m in self.hazards.items()}


def fit_forecaster(X: np.ndarray, next_stage: np.ndarray, event_id: np.ndarray, hazard_targets: dict,
                   *, names: list[str], seed: int = 17) -> Forecaster:
    """``event_id`` groups anchors that share one future transition; each event gets equal
    total weight, so a transition preceded by a long quiet spell does not dominate."""
    f = Forecaster(feature_names=list(names))
    m = next_stage > 0
    if m.sum() >= 20 and np.unique(next_stage[m]).size >= 2:
        _, inv, cnt = np.unique(event_id[m], return_inverse=True, return_counts=True)
        w = 1.0 / cnt[inv]
        w = w * (m.sum() / w.sum())
        f.next_stage = _hgb(seed, min_samples_leaf=20).fit(X[m], next_stage[m], sample_weight=w)
    for h, (y, valid) in hazard_targets.items():
        yy, vv = y[valid], valid
        if yy.sum() >= 5 and (~yy).sum() >= 5:
            f.hazards[h] = _hgb(seed).fit(X[vv], yy.astype(int), sample_weight=balanced_weights(yy.astype(int)))
        else:
            f.hazards[h] = None
    return f

"""Tabular baselines: logistic regression (required), stacked LR, gradient boosting.

These exist to be beaten and to make every claim falsifiable. All of them see exactly the
world model's inputs (``sequences.model_input``) at the same anchors, and are scored by the
same function with the same threshold rule.

* ``logreg``: current window only. This is what the problem statement asks us to beat.
* ``logreg_stack``: the last 6 windows concatenated. It separates "more history" from
  "learned dynamics".
* ``hgb``: sklearn HistGradientBoosting on the current window, a strong tabular reference.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from ..data.sequences import Arrays, context_index, model_input


def features_for(kind: str, arr: Arrays, anchors: np.ndarray) -> np.ndarray:
    if kind in ("logreg", "hgb"):
        return model_input(arr, anchors)
    if kind == "logreg_stack":
        idx = context_index(anchors, 6)
        return model_input(arr, idx).reshape(len(anchors), -1)
    raise KeyError(kind)


def fit(kind: str, x: np.ndarray, y: np.ndarray, *, seed: int):
    if kind in ("logreg", "logreg_stack"):
        model = LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0, random_state=seed)
    elif kind == "hgb":
        model = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0,
            class_weight="balanced", random_state=seed,
        )
    else:
        raise KeyError(kind)
    model.fit(x, y.astype(int))
    return model


def predict(model, x: np.ndarray) -> np.ndarray:
    return model.predict_proba(x)[:, 1]

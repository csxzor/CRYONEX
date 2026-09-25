"""Forecast metrics.

Ported from netforecast v1 ``eval/metrics.py`` (classification metrics, reliability curve)
and extended with what v1's audit showed was missing:

* **Episode lead time.** How long before an episode (or stage) onset a *sustained* alert
  first fires. Beacon recurrences are excluded (see ``targets.episodes``).
* **Block bootstrap.** Windows are autocorrelated, so resampling single windows understates
  the uncertainty. Whole 30-minute blocks are resampled instead.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)


@dataclass
class ClassificationMetrics:
    precision: float
    recall: float
    f1: float
    fpr: float
    auprc: float
    roc_auc: float
    brier: float
    ece: float
    threshold: float
    positives: int
    n: int

    def as_dict(self) -> dict:
        return asdict(self)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 0, 1)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def classification_metrics(y_true: np.ndarray, y_score: np.ndarray, *, threshold: float) -> ClassificationMetrics:
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    both = len(np.unique(y_true)) > 1
    return ClassificationMetrics(
        precision=float(precision), recall=float(recall), f1=float(f1),
        fpr=float(fp / (fp + tn)) if (fp + tn) else 0.0,
        auprc=float(average_precision_score(y_true, y_score)) if both else float("nan"),
        roc_auc=float(roc_auc_score(y_true, y_score)) if both else float("nan"),
        brier=float(brier_score_loss(y_true, np.clip(y_score, 0, 1))),
        ece=expected_calibration_error(y_true, y_score),
        threshold=float(threshold), positives=int(y_true.sum()), n=int(y_true.size),
    )


def sustained(alert: np.ndarray, session: np.ndarray, k: int) -> np.ndarray:
    """True where ``alert`` has held for ``k`` consecutive windows in the same session."""
    alert = np.asarray(alert, dtype=bool)
    session = np.asarray(session)
    out = alert.copy()
    for j in range(1, k):
        prev = np.zeros_like(alert)
        prev[j:] = alert[:-j] & (session[j:] == session[:-j])
        out &= prev
    return out


@dataclass
class LeadTime:
    n_onsets: int
    warned_before: int      # sustained alert already on at the onset window's predecessor
    detected_within: int    # warned before, or alerted within `grace` windows after onset
    median_lead_windows: float
    mean_lead_windows: float
    per_onset: list[dict]

    def as_dict(self) -> dict:
        d = asdict(self)
        d["warned_before_frac"] = self.warned_before / self.n_onsets if self.n_onsets else float("nan")
        d["detected_frac"] = self.detected_within / self.n_onsets if self.n_onsets else float("nan")
        return d


def lead_time(
    score: np.ndarray,
    onset_rows: np.ndarray,
    session: np.ndarray,
    *,
    threshold: float,
    max_lookback: int,
    hysteresis: int = 2,
    grace: int = 6,
) -> LeadTime:
    """Lead time of sustained alerts before each onset (rows index the same arrays).

    For each onset, walk back from the window before it while a sustained alert holds,
    within the session and at most ``max_lookback`` windows. The lead is the length of that
    run. An alert run that stopped before the onset does not count, because a warning that
    lapsed is not a warning an analyst would act on at the onset.
    """
    score = np.asarray(score, dtype=float)
    session = np.asarray(session)
    alert = sustained(score >= threshold, session, hysteresis)
    leads, per = [], []
    warned = detected = 0
    for i in np.asarray(onset_rows, dtype=int):
        lead = 0
        j = i - 1
        while j >= 0 and session[j] == session[i] and lead < max_lookback and alert[j]:
            lead += 1
            j -= 1
        after = alert[i : i + grace + 1]
        same = session[i : i + grace + 1] == session[i]
        late = bool(np.any(after & same))
        warned += lead > 0
        detected += (lead > 0) or late
        if lead > 0:
            leads.append(lead)
        per.append({"row": int(i), "lead_windows": int(lead), "detected": bool(lead > 0 or late)})
    return LeadTime(
        n_onsets=len(per), warned_before=int(warned), detected_within=int(detected),
        median_lead_windows=float(np.median(leads)) if leads else 0.0,
        mean_lead_windows=float(np.mean(leads)) if leads else 0.0,
        per_onset=per,
    )


def time_blocks(session: np.ndarray, pos: np.ndarray, block: int = 180) -> np.ndarray:
    """Block id per row: (session, position // block). 180 windows = 30 minutes at 10 s."""
    session = np.asarray(session)
    _, s = np.unique(session, return_inverse=True)
    return s.astype(np.int64) * 1_000_000 + (np.asarray(pos) // block)


def block_bootstrap(
    fn, y: np.ndarray, s: np.ndarray, blocks: np.ndarray, *, n_boot: int = 500, seed: int = 0
) -> tuple[float, float]:
    """95% percentile CI of ``fn(y, s)`` under block resampling."""
    y, s, blocks = np.asarray(y), np.asarray(s), np.asarray(blocks)
    ids, inv = np.unique(blocks, return_inverse=True)
    members = [np.flatnonzero(inv == b) for b in range(ids.size)]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, ids.size, ids.size)
        rows = np.concatenate([members[b] for b in pick])
        if y[rows].min() == y[rows].max():
            continue
        vals.append(fn(y[rows], s[rows]))
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def paired_block_bootstrap(
    fn, y: np.ndarray, s_a: np.ndarray, s_b: np.ndarray, blocks: np.ndarray,
    *, n_boot: int = 500, seed: int = 0,
) -> dict:
    """CI of ``fn(y, s_a) - fn(y, s_b)`` on identical resamples: is model A really better?"""
    y, s_a, s_b, blocks = map(np.asarray, (y, s_a, s_b, blocks))
    ids, inv = np.unique(blocks, return_inverse=True)
    members = [np.flatnonzero(inv == b) for b in range(ids.size)]
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        rows = np.concatenate([members[b] for b in rng.integers(0, ids.size, ids.size)])
        if y[rows].min() == y[rows].max():
            continue
        diffs.append(fn(y[rows], s_a[rows]) - fn(y[rows], s_b[rows]))
    diffs = np.asarray(diffs)
    return {"diff": float(fn(y, s_a) - fn(y, s_b)),
            "ci": (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))),
            "p_a_better": float((diffs > 0).mean())}


def reliability_curve(y: np.ndarray, p: np.ndarray, bins: int = 10) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    out = {"predicted": [], "observed": [], "count": []}
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.any():
            out["predicted"].append(float(p[m].mean()))
            out["observed"].append(float(y[m].mean()))
            out["count"].append(int(m.sum()))
    return out

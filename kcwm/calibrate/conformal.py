"""Conformal calibration: turning a score into an alert budget.

Ported unchanged from netforecast v1 ``calibrate/conformal.py``.

Our own baselines are the argument for this module. Under cross-family evaluation the LSTM
reached AUPRC 0.942 while recalling 11.5% of positives at a usable threshold, with a Brier
score worse than predicting the base rate. It ranked windows beautifully and its
probabilities meant nothing. A system whose threshold is 0.5 because 0.5 is the default
cannot tell an operator how many false alerts to expect, and at critical-infrastructure
scale that is the difference between a tool and shelfware.

Split-conformal prediction gives a distribution-free guarantee: choose the threshold on a
held-out calibration split and the false-positive rate on exchangeable future data is
bounded by the budget you asked for.

**The exchangeability caveat, stated rather than buried.** Network traffic is not
exchangeable. It drifts across days, shifts, and deployments, so the finite-sample
guarantee is approximate here. Two things are done about it rather than hoping: the
calibration split is contiguous and recent, and an adaptive variant re-estimates the
threshold over a sliding window so it tracks drift. Reported coverage on held-out data is
what should be believed, not the nominal rate.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class ConformalThreshold:
    """A calibrated alert threshold and what it actually achieved."""

    threshold: float
    target_fpr: float
    achieved_fpr: float
    achieved_recall: float
    alerts_per_hour: float
    n_calibration: int
    window_seconds: float
    method: str

    def summary(self) -> str:
        return (
            f"threshold {self.threshold:.4f} | target FPR {self.target_fpr:.1%} | "
            f"achieved {self.achieved_fpr:.1%} | recall {self.achieved_recall:.1%} | "
            f"~{self.alerts_per_hour:.1f} false alerts/hour"
        )

    def to_dict(self) -> dict:
        return asdict(self)


def calibrate(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    target_fpr: float = 0.01,
    window_seconds: float = 10.0,
) -> ConformalThreshold:
    """Split-conformal threshold for a false-positive budget.

    The threshold is the ``1 - target_fpr`` quantile of the scores assigned to **negative**
    calibration windows: by construction only that fraction of negatives exceed it. The
    finite-sample correction ``ceil((n+1)(1-a))/n`` is what makes this a conformal
    guarantee rather than an empirical quantile, and it matters when the calibration split
    is small.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).astype(bool).ravel()
    negatives = scores[~labels]

    if negatives.size == 0:
        return ConformalThreshold(
            threshold=0.5, target_fpr=target_fpr, achieved_fpr=0.0, achieved_recall=0.0,
            alerts_per_hour=0.0, n_calibration=0, window_seconds=window_seconds,
            method="split-conformal (no negatives; defaulted)",
        )

    n = negatives.size
    rank = int(np.ceil((n + 1) * (1 - target_fpr)))
    rank = min(max(rank, 1), n)
    threshold = float(np.sort(negatives)[rank - 1])

    alerts = scores >= threshold
    achieved_fpr = float(alerts[~labels].mean()) if (~labels).any() else 0.0
    achieved_recall = float(alerts[labels].mean()) if labels.any() else 0.0

    return ConformalThreshold(
        threshold=threshold,
        target_fpr=target_fpr,
        achieved_fpr=achieved_fpr,
        achieved_recall=achieved_recall,
        alerts_per_hour=achieved_fpr * 3600.0 / window_seconds,
        n_calibration=n,
        window_seconds=window_seconds,
        method="split-conformal",
    )


def calibrate_adaptive(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    target_fpr: float = 0.01,
    window: int = 500,
    window_seconds: float = 10.0,
) -> tuple[np.ndarray, ConformalThreshold]:
    """Rolling threshold that tracks drift.

    Traffic is not exchangeable across days, so a threshold fixed once slowly stops meaning
    what it meant. This re-estimates from the most recent ``window`` negatives at each
    step, trading the exact finite-sample guarantee for one that stays roughly true as the
    distribution moves. Returns the per-step thresholds and the summary of what they
    achieved.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).astype(bool).ravel()

    thresholds = np.empty(len(scores))
    running: list[float] = []
    fallback = float(np.quantile(scores, 1 - target_fpr)) if len(scores) else 0.5

    for i, (score, label) in enumerate(zip(scores, labels)):
        if len(running) >= 20:
            recent = np.asarray(running[-window:])
            n = recent.size
            rank = min(max(int(np.ceil((n + 1) * (1 - target_fpr))), 1), n)
            thresholds[i] = float(np.sort(recent)[rank - 1])
        else:
            thresholds[i] = fallback
        if not label:  # only negatives inform the false-positive budget
            running.append(score)

    alerts = scores >= thresholds
    achieved_fpr = float(alerts[~labels].mean()) if (~labels).any() else 0.0
    achieved_recall = float(alerts[labels].mean()) if labels.any() else 0.0

    summary = ConformalThreshold(
        threshold=float(np.median(thresholds)),
        target_fpr=target_fpr,
        achieved_fpr=achieved_fpr,
        achieved_recall=achieved_recall,
        alerts_per_hour=achieved_fpr * 3600.0 / window_seconds,
        n_calibration=int((~labels).sum()),
        window_seconds=window_seconds,
        method=f"adaptive split-conformal (window {window})",
    )
    return thresholds, summary


def coverage_report(
    calibration_scores: np.ndarray,
    calibration_labels: np.ndarray,
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    *,
    budgets: tuple[float, ...] = (0.005, 0.01, 0.02, 0.05),
    window_seconds: float = 10.0,
) -> list[dict]:
    """Calibrate at several budgets and report what each achieved on held-out data.

    The gap between target and achieved FPR is the honest measure of whether the
    exchangeability assumption held. A large gap is a finding to report, not a bug to hide:
    it quantifies the drift between calibration and deployment.
    """
    rows = []
    for budget in budgets:
        fitted = calibrate(
            calibration_scores, calibration_labels,
            target_fpr=budget, window_seconds=window_seconds,
        )
        alerts = np.asarray(test_scores) >= fitted.threshold
        test_labels = np.asarray(test_labels).astype(bool)

        test_fpr = float(alerts[~test_labels].mean()) if (~test_labels).any() else 0.0
        test_recall = float(alerts[test_labels].mean()) if test_labels.any() else 0.0

        rows.append(
            {
                "target_fpr": budget,
                "threshold": fitted.threshold,
                "calibration_fpr": fitted.achieved_fpr,
                "test_fpr": test_fpr,
                "test_recall": test_recall,
                "test_alerts_per_hour": test_fpr * 3600.0 / window_seconds,
                "coverage_gap": test_fpr - budget,
                "guarantee_held": bool(test_fpr <= budget * 1.5),
            }
        )
    return rows


def save(threshold: ConformalThreshold, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(threshold.to_dict(), indent=2), encoding="utf-8")


def load(path: str | Path) -> ConformalThreshold:
    return ConformalThreshold(**json.loads(Path(path).read_text(encoding="utf-8")))

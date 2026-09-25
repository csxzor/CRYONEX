"""Tabulate result JSONs. Every number in docs/BENCHMARKS.md comes from here, never by hand."""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from .. import config

ORDER = ["logreg", "logreg_stack", "hgb", "lstm", "transformer_clf", "kcwm", "hybrid"]
NAMES = {
    "logreg": "Logistic regression (required baseline)",
    "logreg_stack": "Logistic regression, last 6 windows",
    "hgb": "Gradient boosting",
    "lstm": "LSTM classifier",
    "transformer_clf": "Transformer classifier (same backbone, no dynamics)",
    "kcwm": "KC-WM (world model alone)",
    "hybrid": "**KC-WM hybrid (world model + gradient boosting)**",
}


def load(tag: str) -> list[dict]:
    d = config.results_dir() / tag
    return [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))]


def table(tag: str, budget: float = 0.03) -> str:
    rows = load(tag)
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
    key = f"fpr_{budget}"
    lines = [f"Results `{tag}` (test split; thresholds from calibration at a {budget:.0%} FPR budget; "
             f"mean over seeds, block-bootstrap 95% CI of the first seed)", "",
             "| Model | Seeds | AUPRC | 95% CI | ROC-AUC | Precision | Recall | F1 | FPR | Early-warning AUPRC (base) | Per-capture AUPRC | Onsets warned | Brier |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in sorted(by_model, key=lambda x: ORDER.index(x) if x in ORDER else 99):
        rs = by_model[m]

        def mean(f):
            return float(np.nanmean([f(r) for r in rs]))

        ci = rs[0].get("auprc_ci", (float("nan"), float("nan")))
        op = lambda r: r["operating_points"][key]  # noqa: E731
        warned = sum(op(r)["lead_time"]["warned_before"] for r in rs)
        n_on = sum(op(r)["lead_time"]["n_onsets"] for r in rs)
        lines.append(
            f"| {NAMES.get(m, m)} | {len(rs)} | {mean(lambda r: r['auprc']):.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] | "
            f"{mean(lambda r: r['roc_auc']):.3f} | {mean(lambda r: op(r)['metrics']['precision']):.3f} | "
            f"{mean(lambda r: op(r)['metrics']['recall']):.3f} | {mean(lambda r: op(r)['metrics']['f1']):.3f} | "
            f"{mean(lambda r: op(r)['metrics']['fpr']):.3f} | "
            f"{mean(lambda r: r.get('early_warning', {}).get('auprc', float('nan'))):.3f} "
            f"({mean(lambda r: r.get('early_warning', {}).get('base_rate', float('nan'))):.3f}) | "
            f"{mean(lambda r: r.get('macro_capture_auprc', float('nan'))):.3f} | {warned}/{n_on} | "
            f"{mean(lambda r: op(r)['metrics']['brier']):.3f} |")
    if rows:
        r0 = rows[0]
        lines += ["", f"Test anchors: {r0['n_test']} (positive rate {r0['test_positive_rate']:.3f}); "
                      f"horizon {r0['horizon']} windows; normalisation `{r0['mode']}`; target `{r0['target']}`."]
    return "\n".join(lines)

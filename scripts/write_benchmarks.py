"""Generate docs/BENCHMARKS.md from result JSONs. No number in that file is typed by hand.

Sections: final P1 test (3-seed mean ± std, per model), per-dataset view, early warning,
stage forecasting (G5), synthetic kill-chain campaigns (P4), unseen families (G6),
new network (G7), explanation faithfulness (G8).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
ORDER = ["logreg", "logreg_stack", "hgb", "lstm", "transformer_clf", "kcwm", "hybrid"]
NAMES = {
    "logreg": "Logistic regression (PS-required baseline)",
    "logreg_stack": "Logistic regression, last 6 windows",
    "hgb": "Gradient boosting",
    "lstm": "LSTM classifier",
    "transformer_clf": "Transformer classifier (same backbone, no dynamics)",
    "kcwm": "KC-WM world model alone",
    "hybrid": "**KC-WM hybrid (deployed)**",
}


def load(d: Path) -> dict[str, list[dict]]:
    out = defaultdict(list)
    for f in sorted(d.glob("*.json")):
        r = json.loads(f.read_text())
        if "model" in r:  # skip policy files such as two_level.json
            out[r["model"]].append(r)
    return out


def ms(vals) -> str:
    v = np.asarray([x for x in vals if x == x], dtype=float)
    if v.size == 0:
        return "n/a"
    return f"{v.mean():.3f} ± {v.std():.3f}" if v.size > 1 else f"{v.mean():.3f}"


def op(r, key="fpr_0.03"):
    return r["operating_points"][key]


def final_section(lines: list[str]) -> None:
    res = load(R / "final-p1")
    if not res:
        lines.append("_Final run not found._\n")
        return
    r0 = next(iter(res.values()))[0]
    lines += ["## 1. Final result: P1 test split (read once)", "",
              f"Test anchors: {r0['n_test']:,} windows (positive rate {r0['test_positive_rate']:.3f}), "
              f"horizon {r0['horizon']} windows (5 min), datasets {', '.join(r0['datasets'])}. "
              "Thresholds chosen on calibration at a 3% FPR budget, then applied unchanged to test. "
              "Mean ± std over seeds 17, 23, 29.", "",
              "| Model | AUPRC | ROC-AUC | Precision | Recall | F1 | FPR | Per-network AUPRC | Brier |",
              "|---|---|---|---|---|---|---|---|---|"]
    for m in ORDER:
        rs = res.get(m)
        if not rs:
            continue
        lines.append(
            f"| {NAMES[m]} | {ms(r['auprc'] for r in rs)} | {ms(r['roc_auc'] for r in rs)} | "
            f"{ms(op(r)['metrics']['precision'] for r in rs)} | {ms(op(r)['metrics']['recall'] for r in rs)} | "
            f"{ms(op(r)['metrics']['f1'] for r in rs)} | {ms(op(r)['metrics']['fpr'] for r in rs)} | "
            f"{ms(r.get('macro_capture_auprc', float('nan')) for r in rs)} | "
            f"{ms(op(r)['metrics']['brier'] for r in rs)} |")
    h = res.get("hybrid", [])
    if h:
        ci = h[0].get("auprc_ci")
        lines += ["", f"Block-bootstrap 95% CI (30-min blocks) of the hybrid's AUPRC, seed 17: "
                      f"[{ci[0]:.3f}, {ci[1]:.3f}]."]
    lines += ["", "### Early warning on real data", "",
              "The early-warning subset is windows with no compromise in the previous 5 minutes; "
              "y=1 means a compromise starts within 5 minutes. Lead time is measured on real "
              "compromise onsets with a 2-window sustained alert.", "",
              "| Model | Early-warning AUPRC | Base rate | Onsets warned before start | Median lead (windows) |",
              "|---|---|---|---|---|"]
    for m in ORDER:
        rs = res.get(m)
        if not rs:
            continue
        lt = [op(r)["lead_time"] for r in rs]
        lines.append(f"| {NAMES[m]} | {ms(r['early_warning']['auprc'] for r in rs)} | "
                     f"{rs[0]['early_warning']['base_rate']:.3f} | "
                     f"{sum(x['warned_before'] for x in lt)}/{sum(x['n_onsets'] for x in lt)} | "
                     f"{ms(x['median_lead_windows'] for x in lt)} |")
    tl = R / "final-p1" / "two_level.json"
    if tl.exists():
        t = json.loads(tl.read_text())
        s = t["summary"]
        lines += ["", "### Two-level alerts (deployed policy; chosen after the test read, so post-hoc)", "",
                  "Level 1 **early warning**: the world model's forecast >= a threshold set on *quiet benign "
                  "calibration windows* at a 3% budget, sustained 2 windows. Level 2 **attack in progress**: "
                  "the hybrid score >= its calibrated threshold. Thresholds are from calibration only; the test "
                  "numbers below use the scores saved by the final run (no retraining).", "",
                  "| | Two-level policy | Hybrid alerts only |", "|---|---|---|",
                  f"| Real attacks warned before they started (3 seeds) | **{s['onsets_warned_two_level']}/{s['onsets_total']}** | "
                  f"{s['onsets_warned_hybrid_only']}/{s['onsets_total']} |",
                  f"| Median warning time | {s['median_lead_windows'] * 10 / 60:.1f} min | n/a |",
                  f"| Early warnings on quiet benign windows | {s['early_warning_false_rate']:.1%} "
                  f"(~{s['false_early_warnings_per_hour']:.1f} per hour) | n/a |"]
    k = res.get("kcwm", [])
    if k and k[0].get("stages"):
        lines += ["", "## 2. Stage forecasting (G5)", "",
                  "Next *new* stage at a real stage transition (forecast made one window before), and "
                  "attack-step stage accuracy over the 5-minute horizon, against count-based controls.", "",
                  "| Set | Next stage: KC-WM | Next stage: Markov table | Transitions | Attack-step: KC-WM | Attack-step: carried forward |",
                  "|---|---|---|---|---|---|"]
        for name, label in (("real", "Real test data"), ("p4", "Test campaigns (P4)")):
            reps = [r["stages"][name] for r in k if r.get("stages", {}).get(name)]
            if not reps:
                continue
            ns = [x["next_stage_lead1"] for x in reps]
            at = [x["attack_steps"] for x in reps]
            lines.append(f"| {label} | {ms(x['model'] for x in ns)} | {ms(x['markov'] for x in ns)} | "
                         f"{ns[0]['n_transitions']} | {ms(x['model'] for x in at)} | "
                         f"{ms(x['nowcast_carried_forward'] for x in at)} |")
    if any(r.get("p4") for rs in res.values() for r in rs):
        lines += ["", "## 3. Synthetic kill-chain campaigns (P4)", "",
                  "Real attack snippets from the test span, injected in ATT&CK order into real benign test "
                  "traffic of the same network. Reported separately and never pooled with real data.", "",
                  "| Model | AUPRC | Early-warning AUPRC (base) | Onsets warned | Median lead (windows) | FPR |",
                  "|---|---|---|---|---|---|"]
        for m in ORDER:
            rs = [r for r in res.get(m, []) if r.get("p4")]
            if not rs:
                continue
            lt = [r["p4"]["lead_time"]["fpr_0.03"] for r in rs]
            lines.append(f"| {NAMES[m]} | {ms(r['p4']['auprc'] for r in rs)} | "
                         f"{ms(r['p4']['early_warning_auprc'] for r in rs)} ({rs[0]['p4']['early_warning_base']:.3f}) | "
                         f"{sum(x['warned_before'] for x in lt)}/{sum(x['n_onsets'] for x in lt)} | "
                         f"{ms(x['median_lead_windows'] for x in lt)} | {ms(x['fpr'] for x in lt)} |")


def gate_sections(lines: list[str]) -> None:
    fams = sorted(d.name.removeprefix("g6-") for d in R.glob("g6-*"))
    if fams:
        lines += ["", "## 4. Unseen attack families (G6, leave-one-family-out, seed 17)", "",
                  "Each family is removed from training, including every anchor whose context or horizon "
                  "touches it and every campaign snippet, then tested.", "",
                  "| Held-out family | Hybrid | World model alone | Gradient boosting | Logistic regression |",
                  "|---|---|---|---|---|"]
        acc = defaultdict(list)
        for f in fams:
            res = load(R / f"g6-{f}")
            vals = {m: res[m][0]["auprc"] for m in ("hybrid", "kcwm", "hgb", "logreg") if m in res}
            for m, v in vals.items():
                acc[m].append(v)
            lines.append(f"| {f} | " + " | ".join(f"{vals.get(m, float('nan')):.3f}" for m in ("hybrid", "kcwm", "hgb", "logreg")) + " |")
        lines.append("| **Mean** | " + " | ".join(f"**{np.mean(acc[m]):.3f}**" for m in ("hybrid", "kcwm", "hgb", "logreg")) + " |")
    tg = sorted(R.glob("g7-*"))
    if tg:
        lines += ["", "## 5. New network (G7, seed 17)", "",
                  "Train on the other three datasets; test on every window of the held-out one.", "",
                  "| Target | Normalisation | Hybrid AUPRC / ROC | World model | Gradient boosting | Logistic regression | Hybrid FPR at source threshold |",
                  "|---|---|---|---|---|---|---|"]
        for d in tg:
            tgt, mode = d.name.removeprefix("g7-").rsplit("-", 1)
            res = load(d)
            cell = lambda m: f"{res[m][0]['auprc']:.3f} / {res[m][0]['roc_auc']:.3f}" if m in res else "n/a"  # noqa: E731
            fpr = op(res["hybrid"][0])["metrics"]["fpr"] if "hybrid" in res else float("nan")
            lines.append(f"| {tgt} | {mode} | {cell('hybrid')} | {cell('kcwm')} | {cell('hgb')} | {cell('logreg')} | {fpr:.3f} |")
    g8 = R / "g8_faithfulness.json"
    if g8.exists():
        g = json.loads(g8.read_text())
        lines += ["", "## 6. Explanation faithfulness (G8)", "",
                  f"On the {g['n']} highest-risk calibration windows, deleting the top-{g['k']} features by "
                  f"integrated gradients lowers the forecast risk by {g['mean_drop_top']:.3f} on average; deleting "
                  f"{g['k']} random features lowers it by {g['mean_drop_random']:.3f} ({g['ratio']:.0f}x). "
                  f"Pass bar 2x: **{'pass' if g['passes_G8'] else 'fail'}**."]


def main() -> None:
    lines = ["# Benchmarks", "",
             "Generated by `scripts/write_benchmarks.py` from `results/`. Development history and "
             "every rejected idea: `docs/DEV_HISTORY.md`. Label audit: `docs/LABEL_AUDIT.md`.", ""]
    final_section(lines)
    gate_sections(lines)
    (ROOT / "docs" / "BENCHMARKS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

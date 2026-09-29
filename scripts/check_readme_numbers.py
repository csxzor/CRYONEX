"""Fail if a headline number in README.md differs from the result files in results/.

Parses the README's results tables and headline sentences, recomputes each number from the
JSON in results/ the same way scripts/write_benchmarks.py does, and compares them at the
precision the README prints. Standard library only, so it runs in CI in under a second.

Usage: python scripts/check_readme_numbers.py [README.md]   (exit 1 on any mismatch)
"""

from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
FINAL = R / "final-p1"
SEEDS = (17, 23, 29)

# README detection-table row label (prefix) -> model id in results/
MODELS = {
    "Logistic regression (**required baseline**)": "logreg",
    "Logistic regression, last 6 windows": "logreg_stack",
    "Gradient boosting": "hgb",
    "LSTM classifier": "lstm",
    "Transformer classifier, same backbone, no dynamics": "transformer_clf",
    "CRYONEX world model alone": "kcwm",
    "**CRYONEX hybrid (deployed)**": "hybrid",
}

failures: list[str] = []
checked = 0


def load(model: str, d: Path = FINAL) -> list[dict]:
    return [json.loads(p.read_text()) for s in SEEDS if (p := d / f"{model}-s{s}.json").exists()]


def num(text: str) -> float:
    return float(re.sub(r"[^0-9.\-]", "", text))


def check(what: str, shown: float, actual: float, decimals: int) -> None:
    global checked
    checked += 1
    if round(actual, decimals) != round(shown, decimals):
        failures.append(f"{what}: README shows {shown}, results give {actual:.{decimals + 2}f}")


def table_rows(md: str, header_start: str) -> list[list[str]]:
    """Cells of every data row of the first table whose header row starts with header_start."""
    lines = md.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("| " + header_start):
            rows = []
            for row in lines[i + 2:]:
                if not row.startswith("|"):
                    break
                rows.append([c.strip() for c in row.strip().strip("|").split("|")])
            return rows
    failures.append(f"table not found: {header_start!r}")
    return []


def main(readme: Path) -> int:
    md = readme.read_text()

    # 1. detection table: AUPRC, precision, recall, F1, FPR (%) per model, mean over seeds
    seen = set()
    for cells in table_rows(md, "Model (same features, same split)"):
        model = next((m for label, m in MODELS.items() if cells[0].startswith(label)), None)
        if model is None:
            failures.append(f"detection table: unknown row {cells[0]!r}")
            continue
        seen.add(model)
        rs = load(model)
        ops = [r["operating_points"]["fpr_0.03"]["metrics"] for r in rs]
        check(f"{model} AUPRC", num(cells[1]), mean(r["auprc"] for r in rs), 3)
        for col, key in ((2, "precision"), (3, "recall"), (4, "f1")):
            check(f"{model} {key}", num(cells[col]), mean(o[key] for o in ops), 3)
        check(f"{model} FPR %", num(cells[5]), 100 * mean(o["fpr"] for o in ops), 1)
        if "2 seeds" in cells[0] and len(rs) != 2:
            failures.append(f"{model}: README says 2 seeds, results have {len(rs)}")
    for model in set(MODELS.values()) - seen:
        failures.append(f"detection table: row for {model} missing")

    # 2. early-warning budget sweep
    for cells in table_rows(md, "Early-warning budget"):
        budget = num(cells[0].split("%")[0]) / 100
        s = json.loads((FINAL / f"two_level_sweep_{budget:g}.json").read_text())["summary"]
        warned, total = (int(x) for x in re.findall(r"\d+", cells[1]))
        check(f"EW {budget:.0%} warned", warned, s["onsets_warned_two_level"], 0)
        check(f"EW {budget:.0%} onsets", total, s["onsets_total"], 0)
        check(f"EW {budget:.0%} median lead min", num(cells[2]), s["median_lead_windows"] * 10 / 60, 1)
        check(f"EW {budget:.0%} false per hour", num(cells[3]), s["false_early_warnings_per_hour"], 1)
    deployed = json.loads((FINAL / "two_level.json").read_text())
    m = re.search(r"\*\*(\d+)% \(deployed\)\*\*", md)
    if m is None:
        failures.append("early-warning table: deployed budget row not found")
    else:
        check("deployed early-warning budget %", int(m.group(1)), 100 * deployed["budget"], 0)

    # 3. next stage at real stage changes (lead 1 window), percentages
    ns = json.loads((FINAL / "next_stage.json").read_text())["mean"]["lead_1"]
    markov = mean(r["stages"]["real"]["next_stage_lead1"]["markov"] for r in load("kcwm"))
    expect = {"World-model rollout alone": ns["wm"], "Plain Markov table": markov,
              "Learned transition table": ns["table"], "**Deployed blend**": ns["blend_0.25"]}
    for cells in table_rows(md, "Predictor"):
        key = next((k for k in expect if cells[0].startswith(k)), None)
        if key is None:
            failures.append(f"next-stage table: unknown row {cells[0]!r}")
            continue
        check(f"next stage: {key}", num(cells[1]), 100 * expect.pop(key), 1)
    for key in expect:
        failures.append(f"next-stage table: row {key!r} missing")

    # 4. headline numbers in "Other checks" and prose
    g6h = [json.loads(Path(p).read_text())["auprc"] for p in sorted(glob.glob(str(R / "g6-*/hybrid-s17.json")))]
    g6l = [json.loads(Path(p).read_text())["auprc"] for p in sorted(glob.glob(str(R / "g6-*/logreg-s17.json")))]
    g3 = json.loads((FINAL / "g3_dynamics.json").read_text())["seeds"]
    g3r = json.loads((R / "final-p1-g3" / "g3_dynamics.json").read_text())["seeds"]
    g8 = json.loads((R / "g8_faithfulness.json").read_text())
    dev_ew = lambda m: mean(r["early_warning"]["auprc"] for r in load(m, R / "dev-r6"))  # noqa: E731
    patterns = [
        (r"\*\*([\d.]+)\*\* AUPRC vs ([\d.]+) for the baseline, (\d) of (\d) families",
         [mean(g6h), mean(g6l), sum(a > b for a, b in zip(g6h, g6l)), len(g6h)], [3, 3, 0, 0]),
        (r"released decoder \*\*([\d.]+)\*\*; beats \"nothing changes\" \(([\d.]+)\) and autoregression \(([\d.]+)\), "
         r"but \*\*loses to the recent-history histogram \(([\d.]+)\) on (\d) of (\d) seeds\*\*",
         [mean(g3[s]["1"][k] for s in g3) for k in ("nll_world_model", "nll_persistence", "nll_ridge_ar", "nll_context_hist")]
         + [sum(g3[s]["1"]["gain_ci"][1] < 0 for s in g3), len(g3)], [3, 3, 3, 3, 0, 0]),
        (r"residual-decoder variant beats the histogram \(([\d.]+)\)",
         [mean(g3r[s]["1"]["nll_world_model"] for s in g3r)], [3]),
        (r"cuts the risk \*\*(\d+)×\*\*", [g8["ratio"]], [0]),
        (r"\(world model ([\d.]+), hybrid ([\d.]+), chance ([\d.]+)\), while on test it was ([\d.]+) \(hybrid ([\d.]+)\)",
         [dev_ew("kcwm"), dev_ew("hybrid"),
          mean(r["early_warning"]["base_rate"] for r in load("kcwm", R / "dev-r6")),
          mean(r["early_warning"]["auprc"] for r in load("kcwm")),
          mean(r["early_warning"]["auprc"] for r in load("hybrid"))], [3, 3, 3, 3, 3]),
    ]
    for pattern, actual, decimals in patterns:
        m = re.search(pattern, md)
        if m is None:
            failures.append(f"sentence not found: {pattern[:60]}...")
            continue
        for i, (a, d) in enumerate(zip(actual, decimals)):
            check(f"{pattern[:30]}... #{i + 1}", float(m.group(i + 1)), a, d)

    for f in failures:
        print("MISMATCH", f)
    print(f"{checked} README numbers checked, {len(failures)} problem(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "README.md"))

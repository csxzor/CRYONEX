"""Two-level alert policy: calibrate on calibration data, score on the saved final test scores.

* Level 2, **attack in progress**: the hybrid score >= its calibrated threshold (unchanged).
* Level 1, **early warning**: the world model's rollout risk >= an early-warning threshold,
  sustained for 2 windows, while no level-2 alert is on. The threshold is the (1 - budget)
  quantile of the world model's score on *quiet benign* calibration windows (no compromise in
  the last 5 minutes, none coming within 5 minutes). That is, a false early warning on at most
  `budget` of quiet windows.

This policy was chosen *after* the final test read showed that the world model, not the hybrid,
carries the early-warning signal (docs/BENCHMARKS.md). It uses only calibration data and saved
scores, with no retraining. Its test numbers are reported as post-hoc.

Usage: python scripts/two_level.py [results/final-p1] [budget]
Writes results/<dir>/two_level.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from kcwm.eval.metrics import lead_time, sustained
from kcwm.eval.protocols import compromise_onsets, prepare, quiet_anchors
from kcwm.eval.run import assemble


def main(d: str = "results/final-p1", budget: float = 0.03) -> None:
    rd = Path(d)
    r0 = json.loads((rd / "hybrid-s17.json").read_text())
    K = r0["horizon"]
    win, split, _, _, cfg = assemble(protocol="p1", tag="x", datasets=r0["datasets"], horizon=K,
                                     dev=bool(r0["split_notes"].get("dev")), campaigns=bool(r0.get("campaigns")),
                                     log=lambda *_: None)
    p = prepare(win, split, cfg, mode=r0["mode"])
    arr = p.arr
    y, v = arr.y[K], arr.valid[K]
    q = int(cfg["window"]["onset_quiet"])
    hyst = int(cfg["alert"]["hysteresis"])
    quiet = quiet_anchors(arr.stage, arr.session, q)
    cal = split.calibration[v[split.calibration]]
    test = split.test[v[split.test]]
    test_mask = np.zeros(win.height, bool)
    test_mask[test] = True
    onsets = np.flatnonzero(compromise_onsets(arr.stage, arr.session, q) & test_mask)
    hours = test.size * float(cfg["window"]["seconds"]) / 3600
    out = {"budget": budget, "post_hoc": True, "test_windows": int(test.size), "test_hours": hours,
           "real_onsets": int(onsets.size), "seeds": {}}
    for seed in (17, 23, 29):
        def load(m):
            s = np.full(win.height, np.nan)
            z = np.load(rd / f"{m}-s{seed}.npz")
            s[z["rows"]] = z["scores"]
            return s

        wm, hy = load("kcwm"), load("hybrid")
        hy_r = json.loads((rd / f"hybrid-s{seed}.json").read_text())
        thr2 = hy_r["operating_points"][f"fpr_{budget}"]["metrics"]["threshold"]
        qb = cal[quiet[cal] & ~y[cal]]  # quiet benign calibration windows
        thr1 = float(np.quantile(wm[qb], 1 - budget))
        full_wm = np.nan_to_num(wm, nan=-np.inf)
        full_hy = np.nan_to_num(hy, nan=-np.inf)
        l2 = sustained(full_hy >= thr2, arr.session, hyst)
        l1 = sustained(full_wm >= thr1, arr.session, hyst) & ~l2
        lt1 = lead_time(np.where(l1 | l2, 1.0, 0.0), onsets, arr.session, threshold=0.5, max_lookback=K,
                        hysteresis=1).as_dict()
        lt2 = lead_time(np.where(l2, 1.0, 0.0), onsets, arr.session, threshold=0.5, max_lookback=K,
                        hysteresis=1).as_dict()
        qt = test[quiet[test]]
        qb_t = qt[~y[qt]]
        qp_t = qt[y[qt]]
        out["seeds"][seed] = {
            "early_warning_threshold_wm": thr1, "in_progress_threshold_hybrid": thr2,
            "onsets_warned_before_two_level": lt1["warned_before"],
            "onsets_warned_before_hybrid_only": lt2["warned_before"],
            "median_lead_windows_two_level": lt1["median_lead_windows"],
            "early_warning_false_rate_on_quiet_benign": float(l1[qb_t].mean()) if qb_t.size else float("nan"),
            "early_warning_hit_rate_on_quiet_positive": float(l1[qp_t].mean()) if qp_t.size else float("nan"),
            "false_early_warnings_per_hour": float((l1[qb_t]).sum() / hours),
        }
    s = out["seeds"]
    out["summary"] = {
        "onsets_warned_two_level": sum(x["onsets_warned_before_two_level"] for x in s.values()),
        "onsets_warned_hybrid_only": sum(x["onsets_warned_before_hybrid_only"] for x in s.values()),
        "onsets_total": 3 * int(onsets.size),
        "median_lead_windows": float(np.mean([x["median_lead_windows_two_level"] for x in s.values()])),
        "early_warning_false_rate": float(np.mean([x["early_warning_false_rate_on_quiet_benign"] for x in s.values()])),
        "early_warning_hit_rate": float(np.mean([x["early_warning_hit_rate_on_quiet_positive"] for x in s.values()])),
        "false_early_warnings_per_hour": float(np.mean([x["false_early_warnings_per_hour"] for x in s.values()])),
    }
    (rd / "two_level.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["summary"], indent=2))


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["results/final-p1"]), *(map(float, sys.argv[2:3])))

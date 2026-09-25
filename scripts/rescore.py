"""Re-derive metrics for saved results without retraining.

Rebuilds windows, split and P4 rows with ``eval.run.assemble`` (the same function the run
used, campaigns included), loads the per-row scores saved next to each result
(``<model>-s<seed>.npz``) and re-runs ``score`` / ``score_rows``. Stage metrics are kept from
the original result.

Usage: python scripts/rescore.py results/dev-r2-kcwm-reg-fa/kcwm-s17.json [more.json ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from kcwm.eval.protocols import prepare, score, score_rows, write_result
from kcwm.eval.run import assemble


def rescore(result_path: str) -> dict:
    rp = Path(result_path)
    r = json.loads(rp.read_text())
    K = r["horizon"]
    notes = r.get("split_notes", {})
    tag = rp.parent.name.removeprefix("dev-")
    win, split, p4_rows, _, cfg = assemble(
        protocol=r.get("protocol_name", "p1"), tag=tag, datasets=r["datasets"], horizon=K,
        dev=bool(notes.get("dev")), campaigns=bool(r.get("campaigns")), log=lambda *_: None)
    p = prepare(win, split, cfg, mode=r["mode"], with_bins=False)
    scores = np.full(win.height, np.nan)
    d = np.load(rp.with_suffix(".npz"))
    scores[d["rows"]] = d["scores"]
    res = score(p, scores, horizon=K, budgets=list(cfg["alert"]["fpr_budgets"]),
                onset_quiet=int(cfg["window"]["onset_quiet"]), hysteresis=int(cfg["alert"]["hysteresis"]))
    valid = p.arr.valid[K]
    if p4_rows.size:
        res["p4"] = score_rows(p, scores, p4_rows[valid[p4_rows]], horizon=K, threshold_from=res, cfg=cfg)
    keep = {k: v for k, v in r.items() if k not in res and k not in ("protocol", "name", "written")}
    write_result(rp.parent.name, rp.stem, {**keep, **res, "rescored": True})
    return res


if __name__ == "__main__":
    for path in sys.argv[1:]:
        res = rescore(path)
        sp = res["sustained_policy"]["fpr_0.03"]
        print(f"{path}: AUPRC {res['auprc']:.3f} | sustained@3%: FPR {sp['fpr']:.3f} recall {sp['recall']:.3f} "
              f"F1 {sp['f1']:.3f}")

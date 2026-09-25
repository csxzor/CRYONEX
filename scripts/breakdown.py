"""Per-dataset breakdown of saved results (no retraining): pooled AUPRC and FPR at the
3%-budget threshold, per dataset, for every model in a results directory.

Usage: python scripts/breakdown.py results/dev-r4 [seed]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from kcwm.eval.protocols import prepare
from kcwm.eval.run import assemble


def main(d: str, seed: str = "17") -> None:
    files = sorted(Path(d).glob(f"*-s{seed}.json"))
    r0 = json.loads(files[0].read_text())
    K = r0["horizon"]
    win, split, _, _, cfg = assemble(protocol="p1", tag="x", datasets=r0["datasets"], horizon=K,
                                     dev=bool(r0["split_notes"].get("dev")), campaigns=bool(r0.get("campaigns")),
                                     log=lambda *_: None)
    p = prepare(win, split, cfg, mode=r0["mode"])
    y, v = p.arr.y[K], p.arr.valid[K]
    test = split.test[v[split.test]]
    ds = win["dataset"].to_numpy()[test]
    names = sorted(set(ds))
    print(f"{'model':16}" + "".join(f"{n[:12]:>24}" for n in names))
    for f in files:
        r = json.loads(f.read_text())
        s = np.full(win.height, np.nan)
        z = np.load(f.with_suffix(".npz"))
        s[z["rows"]] = z["scores"]
        thr = r["operating_points"]["fpr_0.03"]["metrics"]["threshold"]
        cells = []
        for n in names:
            m = ds == n
            yy, ss = y[test][m], s[test][m]
            a = average_precision_score(yy, ss) if 0 < yy.sum() < len(yy) else float("nan")
            fpr = ((ss >= thr) & ~yy).sum() / max((~yy).sum(), 1)
            cells.append(f"{a:.3f} fpr{fpr:.2f} n{m.sum()}")
        print(f"{r['model']:16}" + "".join(f"{c:>24}" for c in cells))


if __name__ == "__main__":
    main(*sys.argv[1:])

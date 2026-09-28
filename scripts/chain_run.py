"""Run the host-level chain protocol and write results/<tag>/chain.json.

Usage: python scripts/chain_run.py <tag> [group,group,...]
Dev folds (for design decisions): ctu-A,ctu-B,ctu-C,cic17-*,cicids2018.
Held-out chain tests (scored once, after the design is frozen): dapt2020, darpa-lldos1, darpa-lldos2.
"""

import json
import sys
import warnings

import polars as pl

from kcwm.chain.evaluate import evaluate, with_table
from kcwm.chain.run import run

warnings.filterwarnings("ignore")
tag = sys.argv[1]
groups = sys.argv[2].split(",") if len(sys.argv) > 2 else None
d = run(tag, groups=groups)
p = pl.read_parquet(d / "oof_predictions.parquet")
ep = pl.read_parquet(d / "episodes.parquet")
p = with_table(p, ep, alpha=0.5)  # the Bayesian transition table, read at the tracker's state
res = evaluate(p, ep)
(d / "chain.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res["lead_1"]["chain_steps"], indent=1))

"""Print a compact table of a chain run's results (results/<tag>/chain.json)."""
import json
import sys

r = json.load(open(f"results/{sys.argv[1]}/chain.json"))
for lead, R in r.items():
    print(f"== {lead}")
    for part in ("chain_steps", "first_steps"):
        for ds, v in R[part].items():
            if not v.get("n"):
                continue
            row = "  ".join(f"{k}={v[k]['top1']:.2f}/{v[k]['top2']:.2f}" for k in v if k != "n")
            print(f"  {part:11} {ds:11} n={v['n']:3}  {row}")
    for ds, H in R["hazard"].items():
        for h, v in H.items():
            d, o = v["deployable"], v["oracle"]
            print(f"  hazard {ds:11} {h:3} base={v['base_rate']:.3f} AUPRC dep={d['auprc']:.3f} orc={o['auprc']:.3f}"
                  f"  warned@1%/5% dep={d['steps_warned_at_0.01']},{d['steps_warned_at_0.05']}"
                  f" orc={o['steps_warned_at_0.01']},{o['steps_warned_at_0.05']}")

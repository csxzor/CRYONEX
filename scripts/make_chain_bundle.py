"""Train the released host kill-chain layer: artifacts/release/chain.joblib.

Trained on every trainable group (CIC-IDS2017, CIC-IDS2018, CTU-13, DAPT2020). DARPA 2000 is
never used, so the DARPA demo samples stay a genuinely unseen network. Run after the chain
protocol's design is frozen (docs/DEV_HISTORY.md).
"""

from __future__ import annotations

import hashlib
import json
import warnings

from kcwm import config
from kcwm.chain.infer import train_bundle
from kcwm.chain.run import TEST_ONLY, prepare

warnings.filterwarnings("ignore")


def main() -> None:
    cfg = config.load()
    P = prepare(cfg)
    groups = sorted(g for g in P["hw"]["group"].unique().to_list() if not any(g.startswith(t) for t in TEST_ONLY))
    b = train_bundle(P["hw"], P["anchors"], groups, meta={"protocol": "chain", "held_out": sorted(TEST_ONLY)})
    rel = config.resolve(cfg["paths"]["release"])
    out = rel / "chain.joblib"
    b.save(out)
    man_path = rel / "MANIFEST.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {}
    man["chain"] = {"file": out.name, "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                    "trained_on": groups, "never_seen": sorted(TEST_ONLY)}
    man_path.write_text(json.dumps(man, indent=2))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB), groups {groups}")


if __name__ == "__main__":
    main()

"""Package a trained model into the release bundle the CLI and console load.

Pass a ``hybrid`` result (recommended; the deployed system) or a ``kcwm`` result.

* hybrid: world-model checkpoint + the gradient-boosting model of the same run and seed +
  both Platt calibrations + the alert threshold chosen on calibration for the hybrid score.
* kcwm: world-model checkpoint + its own Platt calibration and threshold.

Nothing is re-estimated here; everything comes from the run's calibration split. Writes
artifacts/release/kcwm.pt and MANIFEST.json (sha256).

Usage: python scripts/make_bundle.py results/r5-final/hybrid-s17.json
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch

from kcwm import config
from kcwm.inference.engine import _apply_calibrator


def main(result_json: str) -> None:
    cfg = config.load()
    rp = Path(result_json)
    r = json.loads(rp.read_text())
    seed = r["seed"]
    budget = cfg["alert"]["primary_fpr_budget"]
    wm_result = r if r["model"] == "kcwm" else json.loads((rp.parent / f"kcwm-s{seed}.json").read_text())
    ckpt_path = wm_result["world_model"]["checkpoint"]
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    wm_cal = wm_result.get("calibrated", {})
    ckpt["calibrator"] = wm_cal.get("platt") or wm_cal.get("knots")
    op = r["operating_points"][f"fpr_{budget}"]["metrics"]
    if r["model"] == "hybrid":
        ckpt["hgb"] = joblib.load(Path(ckpt_path).parent / f"hgb-s{seed}.joblib")
        ckpt["hybrid"] = r["hybrid"]
        ckpt["threshold"] = float(op["threshold"])  # already on the hybrid probability scale
    else:
        ckpt["threshold"] = float(_apply_calibrator(np.array([op["threshold"]]), ckpt["calibrator"])[0])
    ckpt["meta"] = {"model": r["model"], "source_result": str(rp), "source_checkpoint": ckpt_path,
                    "auprc": r["auprc"], "fpr_budget": budget, "datasets": r["datasets"],
                    "mode": r["mode"], "campaigns": r.get("campaigns"),
                    "dev": bool(r.get("split_notes", {}).get("dev"))}
    out = config.resolve(cfg["paths"]["release"]) / "kcwm.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    (out.parent / "MANIFEST.json").write_text(json.dumps(
        {"kcwm.pt": {"sha256": digest, "bytes": out.stat().st_size, "threshold": ckpt["threshold"],
                     **ckpt["meta"]}}, indent=2))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB), model {r['model']}, "
          f"threshold {ckpt['threshold']:.3f}, sha256 {digest[:12]}")


if __name__ == "__main__":
    main(sys.argv[1])

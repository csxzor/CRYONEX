"""Inference engine: a PCAP or flow CSV in, a per-window forecast timeline out. Fully offline.

``analyze`` is the single entry point used by the CLI (``kcwm forecast``) and the UI:

1. Ingest. The input format is detected: PCAP/PCAPNG via NFStream (packet-level block
   populated), a CICFlowMeter CSV (CIC-IDS / DAPT), or an Argus binetflow (CTU-13).
   Ground-truth labels are kept when the file has them, for the truth ribbon only; the
   model never sees them.
2. Windows and features, with exactly the code that built the training data.
3. Scaling with the bundle's training scaler. In ``warmup`` mode, the capture's own first
   ``warmup_minutes`` re-centre it (label-free adaptation to a new network).
4. For every window with a full context: the analytic rollout (P_infil at every k, stage
   marginals, nowcast, progress) and one-step surprise.
5. Alerts: P_infil at the primary horizon at or above the calibrated threshold, sustained
   over ``hysteresis`` windows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import torch

from .. import config
from ..data.sequences import Arrays, context_index, model_input
from ..features.build import build_capture
from ..features.normalize import QuantileBinner, RobustScaler, transform_windows
from ..features.registry import FEATURE_NAMES, MASK_NAMES
from ..model.rollout import analytic, estimate_progress, one_step_nll
from ..model.world_model import KillChainWorldModel
from ..stages import STAGES

RFC1918 = ["10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"]


@dataclass
class Bundle:
    model: KillChainWorldModel
    scaler: RobustScaler
    binner: QuantileBinner
    cfg: dict
    mode: str
    threshold: float
    calibrator: dict | None = None
    meta: dict = field(default_factory=dict)
    # Hybrid alert score (see eval.run.hybrid_scores): gradient boosting on the current
    # window, Platt-calibrated, averaged with the Platt-calibrated world-model rollout risk.
    hgb: object | None = None
    hybrid: dict | None = None

    @property
    def context(self) -> int:
        return int(self.cfg["context"])

    @property
    def horizon(self) -> int:
        return int(self.cfg["primary_horizon"])


def load_bundle(path: str | Path) -> Bundle:
    path = Path(path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    c = ckpt["config"]
    binner = QuantileBinner.from_dict(ckpt["binner"])
    model = KillChainWorldModel(len(FEATURE_NAMES), len(MASK_NAMES), n_bins=np.asarray(ckpt["n_bins"]),
                                d=c["d"], layers=c["layers"], heads=c["heads"], ff=c["ff"],
                                dropout=0.0, max_len=c["context"],
                                n_direct=len(c["horizons"]) if c.get("direct_risk") else 0)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return Bundle(model=model, scaler=RobustScaler.from_dict(ckpt["scaler"]), binner=binner, cfg=c,
                  mode=ckpt.get("mode", "global"), threshold=float(ckpt.get("threshold", 0.5)),
                  calibrator=ckpt.get("calibrator"), meta=ckpt.get("meta", {}),
                  hgb=ckpt.get("hgb"), hybrid=ckpt.get("hybrid"))


def default_bundle() -> str | None:
    """KCWM_BUNDLE, else the released bundle, else the newest trained run."""
    import os

    env = os.environ.get("KCWM_BUNDLE")
    if env:
        return env
    rel = config.resolve("artifacts/release/kcwm.pt")
    if rel.exists():
        return str(rel)
    runs = sorted(config.resolve("runs").glob("*/kcwm-s*.pt"), key=lambda p: p.stat().st_mtime)
    return str(runs[-1]) if runs else None


def detect_format(path: Path) -> str:
    suffix = "".join(path.suffixes).lower()
    if any(s in suffix for s in (".pcap", ".pcapng", ".cap")):
        return "pcap"
    if suffix.endswith(".binetflow") or suffix.endswith(".binetflow.gz"):
        return "ctu13"
    head = pl.read_csv(path, n_rows=0, infer_schema=False).columns
    folded = {"".join(ch for ch in h.lower() if ch.isalnum()) for h in head}
    if "starttime" in folded and "totbytes" in folded:
        return "ctu13"
    if "stage" in folded and "activity" in folded:
        return "dapt2020"
    return "cicids"


def read_any(path: str | Path, *, ctu_scenario: int | None = None) -> tuple[pl.DataFrame, str]:
    from ..ingest.flowcsv import read_cicids, read_ctu13, read_dapt2020

    path = Path(path)
    fmt = detect_format(path)
    if fmt == "pcap":
        from ..ingest.pcap import read_pcap

        return read_pcap(path), fmt
    if fmt == "ctu13":
        import re

        m = re.search(r"s(\d{2})", path.name)  # samples are named ..._s04_...; labels only
        return read_ctu13(path, scenario=ctu_scenario or (int(m.group(1)) if m else 0)), fmt
    if fmt == "dapt2020":
        return read_dapt2020(path), fmt
    return read_cicids(path, dataset="cicids2017"), fmt


def feature_observed(arr: Arrays, rows: np.ndarray) -> np.ndarray:
    """(n, F) float mask: 1 where the feature's group was observed in that window."""
    from ..features.registry import feature_mask_index

    gate = np.asarray(feature_mask_index())
    fm = np.ones((len(rows), arr.x.shape[1]), dtype=np.float32)
    for g in range(arr.m.shape[1]):
        fm[:, gate == g] = arr.m[rows, g : g + 1]
    return fm


def _apply_calibrator(p: np.ndarray, cal: dict | None) -> np.ndarray:
    """Platt (smooth) if available, else isotonic knots, else the raw rollout probability."""
    if not cal:
        return p
    if "a" in cal:
        q = np.clip(p, 1e-4, 1 - 1e-4)
        z = cal["a"] * np.log(q / (1 - q)) + cal["b"]
        return np.where(np.isnan(p), np.nan, 1 / (1 + np.exp(-z)))
    return np.interp(p, np.asarray(cal["x"]), np.asarray(cal["y"]))


@dataclass
class Analysis:
    windows: pl.DataFrame          # per-window timeline incl. forecasts
    flows: pl.DataFrame            # windowed flows (for flagged-flow tables)
    p_infil: np.ndarray            # (n_windows, K) NaN where no full context
    stage_marg: np.ndarray         # (n_windows, K, S)
    nowcast: np.ndarray            # (n_windows, S)
    arrays: Arrays
    fmt: str
    threshold: float
    horizon: int
    has_labels: bool
    meta: dict


@torch.no_grad()
def analyze(path: str | Path, bundle: Bundle, *, internal_cidrs: list[str] | None = None,
            mode: str | None = None, capture: str | None = None, batch: int = 256,
            flows: pl.DataFrame | None = None, fmt: str | None = None) -> Analysis:
    cfg = config.load()
    if flows is None:
        flows, fmt = read_any(path)
    cidrs = internal_cidrs or RFC1918
    name = capture or Path(str(path)).stem
    flows_w, win = build_capture(flows, dataset="upload", capture=name, cfg=cfg, internal_cidrs=cidrs)
    raw = win.select(FEATURE_NAMES).to_numpy().astype(np.float64)
    _, session = np.unique(win["session_key"].to_numpy(), return_inverse=True)
    warm = win["warmup"].to_numpy().astype(bool)
    mode = mode or bundle.mode
    x = transform_windows(bundle.scaler, raw, mode=mode, session=session, warmup=warm)
    pos = win["pos_in_session"].to_numpy().astype(np.int64)
    length = np.bincount(session)[session]
    arr = Arrays(x=x, m=win.select(MASK_NAMES).to_numpy().astype(np.float32),
                 stage=win["stage_now"].to_numpy().astype(np.int64), session=session.astype(np.int64),
                 pos=pos, remaining=length - 1 - pos, y={}, valid={},
                 bins=bundle.binner.transform(x))
    n, K, S = win.height, bundle.horizon, len(STAGES)
    p_infil = np.full((n, K), np.nan)
    stage_marg = np.full((n, K, S), np.nan)
    nowcast = np.full((n, S), np.nan)
    progress = np.full(n, -1)
    surprise = np.full(n, np.nan)
    rows = np.flatnonzero(pos >= bundle.context - 1)
    m = bundle.model
    for i in range(0, len(rows), batch):
        r = rows[i : i + batch]
        H = m.encode(torch.from_numpy(model_input(arr, context_index(r, bundle.context))))
        b0 = torch.softmax(m.nowcast(H[:, -1]), -1)
        p0 = estimate_progress(m, H)
        fc = analytic(m, H[:, -1], b0, p0, K)
        p_infil[r] = fc.p_infil.numpy()
        stage_marg[r] = fc.stage_marg.numpy()
        nowcast[r] = b0.numpy()
        progress[r] = p0.numpy()
        # surprise of window t under the model's one-step prediction from t-1
        prev_stage = torch.softmax(m.nowcast(H[:, -2]), -1).argmax(-1)
        prev_prog = estimate_progress(m, H[:, :-1])
        fm = torch.from_numpy(feature_observed(arr, r))
        surprise[r] = one_step_nll(m, H[:, -2], prev_stage, prev_prog,
                                   torch.from_numpy(arr.bins[r]), fm).numpy()
    wm_risk = _apply_calibrator(p_infil[:, K - 1], bundle.calibrator)
    risk = wm_risk
    hgb_risk = np.full(n, np.nan)
    if bundle.hgb is not None and bundle.hybrid is not None:
        pr = bundle.hybrid["platt"]
        hgb_risk[rows] = _apply_calibrator(bundle.hgb.predict_proba(model_input(arr, rows))[:, 1], pr["hgb"])
        wm_cal = _apply_calibrator(p_infil[:, K - 1], pr["kcwm"])
        risk = 0.5 * hgb_risk + 0.5 * wm_cal
    now_stage = np.where(np.isnan(nowcast).any(1), -1, np.nan_to_num(nowcast).argmax(1))
    horizon_stage = np.nanargmax(np.nan_to_num(stage_marg[:, :, 1:]).sum(1), axis=1) + 1
    alert = (risk >= bundle.threshold) & ~np.isnan(risk)
    from ..eval.metrics import sustained

    sus = sustained(alert, session, int(cfg["alert"]["hysteresis"]))
    timeline = win.select(
        "window", "t", "session", "pos_in_session", "n_flows", "n_attack_flows", "stage_now",
        "family_now", "warmup",
    ).with_columns(
        pl.Series("risk", risk), pl.Series("alert", sus),
        pl.Series("wm_risk", wm_risk), pl.Series("detector_risk", hgb_risk),
        pl.Series("nowcast_stage", now_stage), pl.Series("progress", progress),
        pl.Series("heading_to", np.where(np.isnan(risk), -1, horizon_stage)),
        pl.Series("surprise", surprise),
        pl.from_epoch(pl.col("t").cast(pl.Int64)).alias("time"),
    )
    has_labels = bool((win["n_attack_flows"] > 0).any())
    return Analysis(windows=timeline, flows=flows_w, p_infil=p_infil, stage_marg=stage_marg,
                    nowcast=nowcast, arrays=arr, fmt=fmt or "?", threshold=bundle.threshold,
                    horizon=K, has_labels=has_labels, meta={"mode": mode, "internal_cidrs": cidrs})


def summary(a: Analysis) -> dict:
    t = a.windows
    alerts = t.filter(pl.col("alert"))
    first = alerts["time"].min() if alerts.height else None
    return {
        "format": a.fmt, "windows": t.height, "flows": a.flows.height,
        "scored_windows": int(t["risk"].is_not_nan().sum()),
        "alert_windows": int(alerts.height),
        "first_alert": str(first) if first is not None else None,
        "max_risk": float(np.nanmax(t["risk"].to_numpy())) if t.height else float("nan"),
        "labelled_attack_windows": int((t["stage_now"] > 0).sum()) if a.has_labels else None,
        "mode": a.meta["mode"],
    }


def to_json(a: Analysis) -> str:
    return json.dumps({"summary": summary(a),
                       "timeline": a.windows.with_columns(pl.col("time").cast(pl.Utf8)).to_dicts()},
                      default=float)

"""Evaluation protocols: one harness scores every model on identical rows with one threshold rule.

v1's audits found four leaks, all in hand-built evaluation paths. Here there is one path:

1. ``prepare`` fits the scaler (and bins) on ``split.fit_rows`` only, then builds arrays.
2. Every model produces a score per anchor row.
3. ``score`` picks the alert threshold on **calibration** anchors (conformal, FPR budget)
   and applies it unchanged to test anchors. It reports window-level metrics with
   block-bootstrap CIs, plus lead time on compromise onsets.

Rows scored for horizon K are anchors with ``valid_K`` (the horizon stays inside the session).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from .. import config
from ..calibrate.conformal import calibrate
from ..data.sequences import Arrays, build_arrays
from ..data.splits import Split
from ..features.normalize import QuantileBinner, RobustScaler
from ..features.registry import FEATURE_NAMES
from ..stages import compromise_mask
from .metrics import (
    block_bootstrap,
    classification_metrics,
    lead_time,
    time_blocks,
)


@dataclass
class Prepared:
    win: pl.DataFrame
    split: Split
    arr: Arrays
    scaler: RobustScaler
    binner: QuantileBinner | None
    mode: str


def prepare(win: pl.DataFrame, split: Split, cfg: dict, *, mode: str = "global",
            with_bins: bool = False) -> Prepared:
    raw = win.select(FEATURE_NAMES).to_numpy().astype(np.float64)
    # Statistics describe *active* traffic: an empty window is a trivial all-zero state, and
    # a near-idle network (DAPT2020: 49% empty windows) would otherwise drag every dataset's
    # medians and quantile bins toward zero.
    fit_rows = split.fit_rows
    if cfg.get("normalise", {}).get("skip_empty", True):
        active = win["n_flows"].to_numpy()[fit_rows] > 0
        if active.sum() >= 100:
            fit_rows = fit_rows[active]
    scaler = RobustScaler.fit(raw[fit_rows])
    horizons = list(cfg["window"]["horizons"])
    arr = build_arrays(win, scaler, mode=mode, horizons=horizons)
    binner = None
    if with_bins:
        binner = QuantileBinner.fit(arr.x[fit_rows], n_bins=16)
        arr.bins = binner.transform(arr.x)
    return Prepared(win, split, arr, scaler, binner, mode)


def compromise_onsets(stage: np.ndarray, session: np.ndarray, quiet: int) -> np.ndarray:
    """Rows where a compromise stage appears after ``quiet`` windows without one."""
    comp = compromise_mask()[np.asarray(stage)]
    out = np.zeros(comp.size, dtype=bool)
    for s in np.unique(session):
        idx = np.flatnonzero(session == s)
        c = comp[idx]
        cs = np.concatenate([[0], np.cumsum(c)])
        j = np.arange(idx.size)
        prior = cs[j] - cs[np.clip(j - quiet, 0, None)]
        out[idx] = c & (prior == 0)
    return out


def quiet_anchors(stage: np.ndarray, session: np.ndarray, quiet: int) -> np.ndarray:
    """True where no compromise-stage window occurred in [t - quiet + 1, t] of the session."""
    comp = compromise_mask()[np.asarray(stage)].astype(np.int64)
    out = np.zeros(comp.size, dtype=bool)
    for s in np.unique(session):
        idx = np.flatnonzero(session == s)
        cs = np.concatenate([[0], np.cumsum(comp[idx])])
        j = np.arange(idx.size)
        out[idx] = (cs[j + 1] - cs[np.clip(j + 1 - quiet, 0, None)]) == 0
    return out


def score(
    p: Prepared,
    scores_by_row: np.ndarray,
    *,
    horizon: int,
    budgets: list[float],
    onset_quiet: int,
    hysteresis: int = 2,
    n_boot: int = 300,
    baseline_windows: int = 90,
) -> dict:
    """Calibrate on calibration anchors, report on test anchors. ``scores_by_row`` is full length (NaN = unscored)."""
    arr, split = p.arr, p.split
    y, valid = arr.y[horizon], arr.valid[horizon]
    cal = split.calibration[valid[split.calibration]]
    test = split.test[valid[split.test]]
    s_cal, y_cal = scores_by_row[cal], y[cal]
    s_test, y_test = scores_by_row[test], y[test]
    blocks = time_blocks(arr.session[test], arr.pos[test])
    from sklearn.metrics import average_precision_score, roc_auc_score

    both = len(np.unique(y_test)) > 1
    out: dict = {
        "n_cal": int(cal.size), "n_test": int(test.size),
        "test_positive_rate": float(y_test.mean()) if test.size else float("nan"),
        "auprc": float(average_precision_score(y_test, s_test)) if both else float("nan"),
        "roc_auc": float(roc_auc_score(y_test, s_test)) if both else float("nan"),
    }
    if both:
        out["auprc_ci"] = block_bootstrap(average_precision_score, y_test, s_test, blocks, n_boot=n_boot)
    # Per-capture view: a deployment watches one network, so discrimination *within* a
    # capture matters as much as the pooled number (which also rewards matching risk levels
    # across networks). Macro average over captures holding both classes.
    caps = p.win["capture"].to_numpy()[test]
    per = {}
    for c in np.unique(caps):
        m = caps == c
        yc = y_test[m]
        if yc.sum() >= 5 and (~yc).sum() >= 5:
            per[str(c)] = {"n": int(m.sum()), "pos": float(yc.mean()),
                           "auprc": float(average_precision_score(yc, s_test[m])),
                           "roc_auc": float(roc_auc_score(yc, s_test[m]))}
    out["per_capture"] = per
    out["macro_capture_auprc"] = float(np.mean([v["auprc"] for v in per.values()])) if per else float("nan")
    out["macro_capture_roc"] = float(np.mean([v["roc_auc"] for v in per.values()])) if per else float("nan")
    # Early-warning subset: anchors with no compromise activity in the last `onset_quiet`
    # windows. Here y=1 means "an attack that is not yet happening will start within K". A
    # model that only detects ongoing attacks scores well on all anchors and poorly here.
    quiet = quiet_anchors(arr.stage, arr.session, onset_quiet)
    tq = test[quiet[test]]
    yq, sq = y[tq], scores_by_row[tq]
    bq = len(np.unique(yq)) > 1
    out["early_warning"] = {
        "n": int(tq.size), "positives": int(yq.sum()),
        "auprc": float(average_precision_score(yq, sq)) if bq else float("nan"),
        "roc_auc": float(roc_auc_score(yq, sq)) if bq else float("nan"),
        "base_rate": float(yq.mean()) if tq.size else float("nan"),
    }
    if bq:
        out["early_warning"]["auprc_ci"] = block_bootstrap(
            average_precision_score, yq, sq, time_blocks(arr.session[tq], arr.pos[tq]), n_boot=n_boot)
    comp_on = compromise_onsets(arr.stage, arr.session, onset_quiet)
    test_mask = np.zeros(arr.stage.size, dtype=bool)
    test_mask[split.test] = True
    onset_rows = np.flatnonzero(comp_on & test_mask)
    full = np.where(np.isnan(scores_by_row), -np.inf, scores_by_row)
    # Isotonic calibration fitted on calibration anchors (monotone: ranking kept, levels fixed).
    from sklearn.isotonic import IsotonicRegression

    if len(np.unique(y_cal)) > 1:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(s_cal, y_cal)
        s_iso = iso.predict(s_test)
        from sklearn.metrics import brier_score_loss

        from .metrics import expected_calibration_error

        out["calibrated"] = {
            "brier_raw": float(brier_score_loss(y_test, np.clip(s_test, 0, 1))),
            "brier_iso": float(brier_score_loss(y_test, s_iso)),
            "ece_raw": expected_calibration_error(y_test, s_test),
            "ece_iso": expected_calibration_error(y_test, s_iso),
            "knots": {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()},
        }
        # Smooth alternative for display (Platt scaling on the logit of the raw score):
        # monotone and continuous, where isotonic steps (0.09, 0.33, ...) look arbitrary in a UI.
        from sklearn.linear_model import LogisticRegression

        def logit(v):
            v = np.clip(np.asarray(v, dtype=float), 1e-4, 1 - 1e-4)
            return np.log(v / (1 - v))[:, None]

        pl_ = LogisticRegression(C=1e3).fit(logit(s_cal), y_cal.astype(int))
        s_platt = pl_.predict_proba(logit(s_test))[:, 1]
        out["calibrated"]["platt"] = {"a": float(pl_.coef_[0, 0]), "b": float(pl_.intercept_[0])}
        out["calibrated"]["brier_platt"] = float(brier_score_loss(y_test, s_platt))
        out["calibrated"]["ece_platt"] = expected_calibration_error(y_test, s_platt)

    # Label-free per-network threshold (gate G7 policy): each session's alert threshold is
    # the (1 - budget) quantile of its own first `baseline_windows` scored test windows,
    # which are then excluded from the metrics.
    out["warmup_threshold"] = {}
    sess_t = arr.session[test]
    for b in budgets:
        keep = np.zeros(test.size, dtype=bool)
        thr = np.full(test.size, np.inf)
        for s in np.unique(sess_t):
            idx = np.flatnonzero(sess_t == s)
            if idx.size <= baseline_windows + 10:
                continue
            base, rest = idx[:baseline_windows], idx[baseline_windows:]
            thr[rest] = np.quantile(s_test[base], 1 - b)
            keep[rest] = True
        if keep.any():
            pred = s_test[keep] >= thr[keep]
            yt = y_test[keep].astype(bool)
            tp, fp = (pred & yt).sum(), (pred & ~yt).sum()
            fn, tn = (~pred & yt).sum(), (~pred & ~yt).sum()
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            out["warmup_threshold"][f"fpr_{b}"] = {
                "n": int(keep.sum()), "precision": float(prec), "recall": float(rec),
                "f1": float(2 * prec * rec / (prec + rec)) if prec + rec else 0.0,
                "fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
            }
    # Deployed alert policy: an alert fires only when the score has held above threshold for
    # `hysteresis` consecutive windows. The threshold is chosen on calibration *under that
    # policy* (largest candidate quantile whose sustained FPR stays within budget), then applied
    # unchanged to test. Reported for every model alike.
    from .metrics import sustained as _sustained

    full = np.where(np.isnan(scores_by_row), -np.inf, scores_by_row)

    def sustained_rates(thr: float, rows: np.ndarray) -> tuple[float, float, np.ndarray]:
        al = _sustained(full >= thr, arr.session, hysteresis)[rows]
        yy = y[rows].astype(bool)
        fpr_ = float((al & ~yy).sum() / max((~yy).sum(), 1))
        rec_ = float((al & yy).sum() / max(yy.sum(), 1))
        return fpr_, rec_, al

    out["sustained_policy"] = {}
    cand = np.unique(np.quantile(s_cal, np.linspace(0.5, 0.999, 200)))
    for b in budgets:
        thr = float(cand[-1])
        for c in cand:  # ascending: first threshold meeting the budget = most recall
            if sustained_rates(float(c), cal)[0] <= b:
                thr = float(c)
                break
        fpr_t, rec_t, al = sustained_rates(thr, test)
        yy = y_test.astype(bool)
        prec = float((al & yy).sum() / max(al.sum(), 1))
        out["sustained_policy"][f"fpr_{b}"] = {
            "threshold": thr, "precision": prec, "recall": rec_t, "fpr": fpr_t,
            "f1": float(2 * prec * rec_t / (prec + rec_t)) if prec + rec_t else 0.0,
        }
    out["operating_points"] = {}
    for b in budgets:
        thr = calibrate(s_cal, y_cal.astype(bool), target_fpr=b).threshold
        m = classification_metrics(y_test, s_test, threshold=thr).as_dict()
        lt = lead_time(full, onset_rows, arr.session, threshold=thr, max_lookback=horizon,
                       hysteresis=hysteresis)
        ltd = lt.as_dict()
        ltd.pop("per_onset")
        out["operating_points"][f"fpr_{b}"] = {"metrics": m, "lead_time": ltd}
    return out


def score_rows(p: Prepared, scores_by_row: np.ndarray, rows: np.ndarray, *, horizon: int,
               threshold_from: dict, cfg: dict) -> dict:
    """Score an extra evaluation set (P4 test campaigns) with the thresholds ``score`` chose on
    real calibration data: AUPRC, early-warning AUPRC and lead time on its compromise onsets."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    arr = p.arr
    y = arr.y[horizon][rows]
    s = scores_by_row[rows]
    quiet = quiet_anchors(arr.stage, arr.session, int(cfg["window"]["onset_quiet"]))[rows]
    both = len(np.unique(y)) > 1
    bq = len(np.unique(y[quiet])) > 1
    out = {"n": int(rows.size), "positive_rate": float(y.mean()) if rows.size else float("nan"),
           "auprc": float(average_precision_score(y, s)) if both else float("nan"),
           "roc_auc": float(roc_auc_score(y, s)) if both else float("nan"),
           "early_warning_auprc": float(average_precision_score(y[quiet], s[quiet])) if bq else float("nan"),
           "early_warning_base": float(y[quiet].mean()) if quiet.any() else float("nan")}
    mask = np.zeros(arr.stage.size, dtype=bool)
    mask[rows] = True
    onsets = np.flatnonzero(compromise_onsets(arr.stage, arr.session, int(cfg["window"]["onset_quiet"])) & mask)
    full = np.where(np.isnan(scores_by_row), -np.inf, scores_by_row)
    out["lead_time"] = {}
    for key, op in threshold_from.get("operating_points", {}).items():
        thr = op["metrics"]["threshold"]
        lt = lead_time(full, onsets, arr.session, threshold=thr, max_lookback=horizon,
                       hysteresis=int(cfg["alert"]["hysteresis"])).as_dict()
        lt.pop("per_onset")
        pred = s >= thr
        out["lead_time"][key] = {**lt, "fpr": float((pred & ~y).sum() / max((~y).sum(), 1)),
                                 "recall": float((pred & y).sum() / max(y.sum(), 1))}
    return out


def results_path(protocol: str, name: str) -> Path:
    path = config.results_dir() / protocol / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_result(protocol: str, name: str, payload: dict) -> Path:
    path = results_path(protocol, name)
    payload = {"protocol": protocol, "name": name, "written": time.strftime("%Y-%m-%d %H:%M:%S"),
               **payload}
    path.write_text(json.dumps(payload, indent=2, default=float))
    reg = config.results_dir() / "registry.jsonl"
    with reg.open("a") as fh:
        fh.write(json.dumps({"protocol": protocol, "name": name, "path": str(path),
                             "written": payload["written"]}) + "\n")
    return path

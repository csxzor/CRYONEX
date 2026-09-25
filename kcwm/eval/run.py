"""Run a protocol for a set of models and seeds; write one result JSON per (model, seed)."""

from __future__ import annotations

import time

import numpy as np
import polars as pl

from .. import config
from ..baselines import neural, tabular
from ..data.splits import assert_purged, cross_split, lofo_split, p1_split
from ..pipeline import load_windows
from ..targets.targets import any_attack_targets
from .protocols import prepare, score, score_rows, write_result

TABULAR = {"logreg", "logreg_stack", "hgb"}
SEQUENCE = {"lstm", "transformer_clf"}


def carve_tail(train: np.ndarray, win, horizon: int, frac: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Split training anchors into (fit, early_stop): the last ``frac`` of each session's
    training anchors (purged by the horizon) is used only to pick the epoch. Calibration stays
    reserved for thresholds, so the epoch-selection set never sets an alarm threshold (that
    overlap made thresholds too low and inflated false alarms). Synthetic campaign anchors
    stay in ``fit``."""
    keys = win["session_key"].to_numpy()
    pos = win["pos_in_session"].to_numpy()
    synth = win["synthetic"].fill_null(False).to_numpy() if "synthetic" in win.columns else np.zeros(win.height, bool)
    fit_, stop = [train[synth[train]]], []
    real = train[~synth[train]]
    for s in np.unique(keys[real]):
        r = real[keys[real] == s]
        lo, hi = pos[r].min(), pos[r].max()
        cut = hi - int((hi - lo) * frac)
        fit_.append(r[pos[r] + horizon < cut])
        stop.append(r[pos[r] >= cut])
    return np.concatenate(fit_), np.concatenate(stop)


def train_world_model(p, cfg: dict, *, train_all, cal_all, seed: int, tag: str, log,
                      overrides: dict | None = None, extra_rows=None) -> dict:
    """Fit KC-WM on the split's training anchors; forecast calibration + test anchors; save it."""
    import torch

    from ..model.train import TrainConfig, fit, forecast_rows

    w = cfg["window"]
    tc = TrainConfig(context=int(w["context"]), unroll=int(w["primary_horizon"]),
                     horizons=tuple(w["horizons"]), primary_horizon=int(w["primary_horizon"]),
                     seed=seed, **(overrides or {}))
    arr = p.arr
    # Training anchors need the full unroll inside nothing but their session; the purge
    # already keeps horizons inside the training span.
    if (overrides or {}).get("stop_on_train_tail"):
        # Tried in dev round 3 and rejected: it removed the training data nearest the
        # evaluation period and collapsed CTU-13 (AUPRC 0.963 -> 0.871, FPR 1% -> 24%).
        fit_rows, stop_rows = carve_tail(np.asarray(train_all), p.win, tc.primary_horizon)
    else:
        fit_rows, stop_rows = np.asarray(train_all), np.asarray(cal_all)
    model, info = fit(arr, fit_rows, stop_rows, tc, n_bins=p.binner.n_bins_per_feature(), log=log)
    rows = np.concatenate([cal_all, p.split.test] + ([np.asarray(extra_rows)] if extra_rows is not None and len(extra_rows) else []))
    fc = forecast_rows(model, arr, rows, tc)
    out_dir = config.resolve(cfg["paths"]["runs"]) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f"kcwm-s{seed}.pt"
    torch.save({"state_dict": model.state_dict(), "config": info["config"],
                "scaler": p.scaler.to_dict(), "binner": p.binner.to_dict(), "mode": p.mode,
                "n_bins": p.binner.n_bins_per_feature().tolist()}, ckpt)
    fc["rows"] = rows
    info["forecast"] = fc
    info["checkpoint"] = str(ckpt)
    return info


def dev_split(split, win, horizon: int, tail: float = 0.2):
    """Development split: the test split is untouched while iterating.

    The last ``tail`` of each session's training span (purged by the horizon) becomes the
    early-stopping / threshold set, and the *whole* calibration split becomes the dev test.
    An earlier version cut calibration in half. That left ~30 min per capture, mostly one
    class, and a dev score that measured cross-network risk levels rather than
    discrimination (see scripts/diagnose_dev.py). Test is scored only without ``dev``, once
    per frozen recipe.
    """
    from ..data.splits import Split

    keys = win["session_key"].to_numpy()
    pos = win["pos_in_session"].to_numpy()
    tr = split.train
    keep, stop = [], []
    for s in np.unique(keys[tr]):
        r = tr[keys[tr] == s]
        lo, hi = pos[r].min(), pos[r].max()
        cut = hi - int((hi - lo) * tail)
        keep.append(r[pos[r] + horizon < cut])
        stop.append(r[pos[r] >= cut])
    return Split(split.name + "-dev", np.concatenate(keep), np.concatenate(stop), split.calibration,
                 split.fit_rows, notes={**split.notes, "dev": True, "tail": tail})


def assemble(*, protocol: str, tag: str, datasets, horizon: int, dev: bool, campaigns: bool,
             family=None, train_datasets=None, test_dataset=None, cfg: dict | None = None, log=print):
    """Windows, split and P4 rows exactly as a run sees them (shared with scripts/rescore.py)."""
    cfg = cfg or config.load()
    win = load_windows(cfg, datasets=datasets)
    split = make_split(protocol, win, cfg, horizon=horizon, family=family,
                       train_datasets=train_datasets, test_dataset=test_dataset)
    if dev:
        split = dev_split(split, win, horizon)
        tag = f"dev-{tag}"
    p4_rows = np.zeros(0, dtype=np.int64)
    if campaigns:
        # Training-split campaigns join the *training* anchors of every model alike. They are
        # appended after the real rows, so real indices (and the evaluation) are unchanged,
        # and the scaler still fits on real training rows only.
        from ..data.splits import Split
        from ..synth.generate import load_campaigns

        real_ds = datasets or sorted(win["dataset"].unique().to_list())
        camp = load_campaigns(real_ds, "train", cfg)
        if camp is not None:
            n_real = win.height
            win = pl.concat([win.with_columns(pl.lit(False).alias("synthetic")), camp],
                            how="diagonal_relaxed")
            pos = camp["pos_in_session"].to_numpy()
            extra = n_real + np.flatnonzero(pos >= int(cfg["window"]["warmup_windows"]))
            split = Split(split.name + "+camp", np.concatenate([split.train, extra]), split.calibration,
                          split.test, split.fit_rows, notes={**split.notes, "campaign_train_anchors": int(extra.size)})
            log(f"[{tag}] + {camp['capture'].n_unique()} training campaigns ({extra.size} anchors)")
        # P4: test-split campaigns are scored, never trained on.
        test_camp = load_campaigns(real_ds, "test", cfg)
        if test_camp is not None:
            n0 = win.height
            win = pl.concat([win, test_camp], how="diagonal_relaxed")
            pos = test_camp["pos_in_session"].to_numpy()
            p4_rows = n0 + np.flatnonzero(pos >= int(cfg["window"]["warmup_windows"]))
            log(f"[{tag}] P4: {test_camp['capture'].n_unique()} test campaigns ({p4_rows.size} anchors)")
    # Down-sample purely benign real training anchors of near-idle datasets (config
    # split.benign_keep). Attack-bearing anchors (anything in the horizon or context) and
    # campaigns are always kept, so the rare attack chains stay fully represented.
    keep_cfg = cfg["split"].get("benign_keep", {})
    if keep_cfg:
        from ..data.splits import Split

        ds = win["dataset"].to_numpy()
        stage = win["stage_now"].to_numpy()
        yk = win[f"y_{horizon}"].to_numpy()
        rng = np.random.default_rng(0)
        tr = split.train
        drop = np.zeros(tr.size, dtype=bool)
        for name, frac in keep_cfg.items():
            m = (ds[tr] == name) & ~yk[tr] & (stage[tr] == 0)
            drop |= m & (rng.random(tr.size) > float(frac))
        if drop.any():
            split = Split(split.name, tr[~drop], split.calibration, split.test, split.fit_rows,
                          notes={**split.notes, "benign_downsampled": int(drop.sum())})
            log(f"[{tag}] down-sampled {int(drop.sum())} benign training anchors ({keep_cfg})")
    return win, split, p4_rows, tag, cfg


def hybrid_scores(s_hgb: np.ndarray, s_wm: np.ndarray, y: np.ndarray, cal: np.ndarray) -> tuple[np.ndarray, dict]:
    """The deployed alert score: mean of the two *calibrated* probabilities.

    Each component gets Platt scaling (logistic on its logit) fitted on calibration anchors.
    That puts both on one probability scale and needs nothing but the current window, so it
    works on a live stream (rank averaging would not).
    """
    from sklearn.linear_model import LogisticRegression

    def logit(v):
        v = np.clip(np.asarray(v, dtype=float), 1e-4, 1 - 1e-4)
        return np.log(v / (1 - v))

    params, parts = {}, []
    for name, s in (("hgb", s_hgb), ("kcwm", s_wm)):
        lr = LogisticRegression(C=1e3).fit(logit(s[cal])[:, None], y[cal].astype(int))
        a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        params[name] = {"a": a, "b": b}
        z = a * logit(np.nan_to_num(s, nan=0.5)) + b
        parts.append(np.where(np.isnan(s), np.nan, 1 / (1 + np.exp(-z))))
    return 0.5 * parts[0] + 0.5 * parts[1], {"platt": params, "weights": {"hgb": 0.5, "kcwm": 0.5}}


def make_split(protocol: str, win, cfg: dict, *, horizon: int, family: str | None = None,
               train_datasets: list[str] | None = None, test_dataset: str | None = None):
    w = cfg["window"]
    fr = (cfg["split"]["train"], cfg["split"]["calibration"], cfg["split"]["test"])
    if protocol == "p1":
        sp = p1_split(win, fractions=fr, horizon=horizon, warmup=int(w["warmup_windows"]))
        assert_purged(sp, win, horizon)
        return sp
    if protocol == "lofo":
        return lofo_split(win, family, horizon=horizon, warmup=int(w["warmup_windows"]),
                          context=int(w["context"]), fractions=fr)
    if protocol == "cross":
        return cross_split(win, train_datasets=train_datasets, test_dataset=test_dataset,
                           horizon=horizon, warmup=int(w["warmup_windows"]), fractions=fr)
    raise KeyError(protocol)


def run(
    *,
    protocol: str,
    tag: str,
    datasets: list[str] | None,
    models: list[str],
    seeds: list[int],
    horizon: int,
    mode: str = "global",
    target: str = "compromise",
    family: str | None = None,
    train_datasets: list[str] | None = None,
    test_dataset: str | None = None,
    wm_overrides: dict | None = None,
    dev: bool = False,
    campaigns: bool = False,
    log=print,
) -> list[dict]:
    win, split, p4_rows, tag, cfg = assemble(
        protocol=protocol, tag=tag, datasets=datasets, horizon=horizon, dev=dev, campaigns=campaigns,
        family=family, train_datasets=train_datasets, test_dataset=test_dataset, log=log)
    p = prepare(win, split, cfg, mode=mode, with_bins=bool({"kcwm", "hybrid"} & set(models)))
    arr = p.arr
    if target == "any":  # v1-compatible target for the reproduction check
        y, valid = any_attack_targets(arr.stage, arr.session, horizon)
        arr.y[horizon], arr.valid[horizon] = y, valid
    y, valid = arr.y[horizon], arr.valid[horizon]
    train = split.train[valid[split.train]]
    cal = split.calibration[valid[split.calibration]]
    rows = np.concatenate([cal, split.test[valid[split.test]], p4_rows[valid[p4_rows]] if p4_rows.size else p4_rows])
    ctx = int(cfg["window"]["context"])
    log(f"[{tag}] rows: train {train.size} (pos {y[train].mean():.3f}), cal {cal.size}, "
        f"test {rows.size - cal.size}; features {arr.n_features}")

    outputs = []
    wm_info: dict[int, dict] = {}
    cache: dict[tuple[str, int], np.ndarray] = {}
    hybrid_info: dict[int, dict] = {}
    if "hybrid" in models:  # hybrid needs its components first, in this run
        models = [m for m in ("hgb", "kcwm") if m not in models] + [m for m in models if m != "hybrid"] + ["hybrid"]
    for model in models:
        for seed in seeds:
            t0 = time.time()
            scores = np.full(arr.stage.size, np.nan)
            if model in TABULAR:
                xtr = tabular.features_for(model, arr, train)
                fitted = tabular.fit(model, xtr, y[train], seed=seed)
                scores[rows] = tabular.predict(fitted, tabular.features_for(model, arr, rows))
                if model == "hgb":
                    import joblib

                    out_dir = config.resolve(cfg["paths"]["runs"]) / tag
                    out_dir.mkdir(parents=True, exist_ok=True)
                    joblib.dump(fitted, out_dir / f"hgb-s{seed}.joblib")
            elif model in SEQUENCE:
                fit_rows, stop_rows = train, cal  # same early-stopping rule as the world model
                fitted = neural.fit_sequence_classifier(
                    "lstm" if model == "lstm" else "transformer", arr, fit_rows, y[fit_rows],
                    stop_rows, y[stop_rows], context=ctx, seed=seed, log=log)
                scores[rows] = neural.predict_sequence_classifier(fitted, arr, rows, context=ctx)
            elif model == "kcwm":
                extra = train_world_model(p, cfg, train_all=split.train, cal_all=split.calibration,
                                          seed=seed, tag=tag, log=log, overrides=wm_overrides,
                                          extra_rows=p4_rows)
                fc = extra.pop("forecast")
                scores[fc["rows"]] = fc["p_infil"][:, horizon - 1]
                wm_info[seed] = extra
            elif model == "hybrid":
                scores, hybrid_info[seed] = hybrid_scores(cache[("hgb", seed)], cache[("kcwm", seed)], y, cal)
            else:
                raise KeyError(model)
            cache[(model, seed)] = scores
            res = score(p, scores, horizon=horizon,
                        budgets=list(cfg["alert"]["fpr_budgets"]),
                        onset_quiet=int(cfg["window"]["onset_quiet"]),
                        hysteresis=int(cfg["alert"]["hysteresis"]))
            if model == "kcwm":
                from .stages import stage_report

                so = p.win["is_stage_onset"].to_numpy().astype(bool)
                st = {"real": stage_report(arr, fc, split.test, split.train, so, context=ctx, K=horizon)}
                if p4_rows.size:
                    st["p4"] = stage_report(arr, fc, p4_rows, split.train, so, context=ctx, K=horizon)
                res["stages"] = st
                for name, rep in st.items():
                    if rep:
                        n1 = rep["next_stage_lead1"]
                        log(f"  G5 {name}: nowcast F1 {rep['nowcast_macro_f1']:.3f}; attack-step acc "
                            f"{rep['attack_steps']['model']:.3f} (carried-forward {rep['attack_steps']['nowcast_carried_forward']:.3f}); "
                            f"next stage @lead1 {n1['model']:.3f} vs Markov {n1['markov']:.3f} (n={n1['n_transitions']})")
            if p4_rows.size:
                res["p4"] = score_rows(p, scores, p4_rows[valid[p4_rows]], horizon=horizon,
                                       threshold_from=res, cfg=cfg)
            payload = {"model": model, "seed": seed, "campaigns": campaigns, "horizon": horizon, "mode": mode,
                       "target": target, "datasets": datasets, "split": split.sizes(),
                       "split_notes": split.notes, "seconds": round(time.time() - t0, 1), **res}
            if model == "kcwm":
                payload["world_model"] = wm_info[seed]
            if model == "hybrid":
                payload["hybrid"] = hybrid_info[seed]
            path = write_result(tag, f"{model}-s{seed}", payload)
            # Per-row scores, so metrics can be re-derived without retraining.
            np.savez_compressed(path.with_suffix(".npz"), rows=rows, scores=scores[rows])
            op = res["operating_points"][f"fpr_{cfg['alert']['primary_fpr_budget']}"]
            log(f"  {model:15s} s{seed}  AUPRC {res['auprc']:.3f} {tuple(round(v, 3) for v in res.get('auprc_ci', (np.nan, np.nan)))}"
                f"  ROC {res['roc_auc']:.3f}  F1@3% {op['metrics']['f1']:.3f}  "
                f"R {op['metrics']['recall']:.3f}  FPR {op['metrics']['fpr']:.3f}  "
                f"warned {op['lead_time']['warned_before']}/{op['lead_time']['n_onsets']}  ({time.time() - t0:.0f}s)")
            outputs.append(payload)
    return outputs

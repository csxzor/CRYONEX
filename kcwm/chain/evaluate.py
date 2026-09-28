"""Score out-of-fold chain forecasts on real chain steps.

A **chain step** is a host moving to a different ATT&CK stage after an earlier one (e.g. the
DAPT2020 host 192.168.3.29: Reconnaissance on Tuesday -> Initial Access on Wednesday). For each
step, the forecast is read at the host's last anchor at least ``lead`` windows before the step
starts, and compared with controls that are given the *true* history (so they are upper
bounds on what a lookup can do):

* ``attack_prior``: the ATT&CK-ordered prior (the world model's A0 with progress bonus);
* ``markov``: counts of (previous stage, progress) -> next stage on the training groups' steps;
* ``majority``: the most common next stage in training.

Timing: ``y_<H>`` asks whether the host enters a compromise stage within H. The warning rate is
read at the same anchors, at thresholds that give 1% / 5% alerts on the held-out anchors
without such a transition (an operating point, the way ROC curves are read).
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score

from ..stages import N_STAGES, STAGES
from .targets import HORIZONS
from .tracker import prior_next

VARIANTS = ("deployable", "oracle")


def with_blend(preds: pl.DataFrame, w: float, *, name: str = "blend") -> pl.DataFrame:
    """Geometric blend of the deployable next-stage forecast with the ATT&CK prior read at the
    tracker's own (predicted) latest stage and progress: p ∝ model^w · prior^(1-w)."""
    m = preds.select([f"deployable_next{s}" for s in range(1, N_STAGES)]).to_numpy()
    pri = prior_next(preds["deployable_latest"].to_numpy().astype(int), preds["deployable_progress"].to_numpy().astype(int))
    b = np.exp(w * np.log(np.clip(m, 1e-6, None)) + (1 - w) * np.log(np.clip(pri, 1e-6, None)))
    b = b / b.sum(1, keepdims=True)
    return preds.with_columns([pl.Series(f"{name}_next{s}", b[:, s - 1]) for s in range(1, N_STAGES)])


def transition_table(chain_steps: pl.DataFrame, alpha: float) -> np.ndarray:
    """Bayesian kill-chain transition model: counts of (previous stage, progress) -> next stage
    on training chain steps, plus ``alpha`` pseudo-counts spread by the ATT&CK prior."""
    t = np.zeros((N_STAGES, N_STAGES, N_STAGES))
    for prev, prog, s in chain_steps.select(["prev_stage", "progress_before", "stage"]).iter_rows():
        t[int(prev), int(prog), int(s)] += 1
    pri = np.zeros_like(t)
    for a in range(N_STAGES):
        for b in range(N_STAGES):
            pri[a, b, 1:] = prior_next(np.array([a]), np.array([b]))[0]
    t = t + alpha * pri
    for a in range(1, N_STAGES):
        t[a, :, a] = 0  # "next" means a different stage
    t[:, :, 0] = 0
    return t / np.clip(t.sum(-1, keepdims=True), 1e-12, None)


def with_table(preds: pl.DataFrame, ep: pl.DataFrame, *, alpha: float = 2.0, name: str = "table") -> pl.DataFrame:
    """Next-stage forecast from the transition table fitted on the other groups' chain steps,
    read at the tracker's deployable (predicted) and oracle states."""
    g_of = dict(preds.select(["timeline", "group"]).unique(subset=["timeline"]).iter_rows())
    steps = ep.filter(pl.col("transition") & pl.col("has_history")).with_columns(
        pl.col("timeline").replace_strict(g_of, default=None).alias("_g"))
    groups = preds["group"].to_numpy()
    out = {f"{name}_{v}": np.zeros((preds.height, N_STAGES - 1)) for v in VARIANTS}
    for g in np.unique(groups):
        tr = steps.filter((pl.col("_g") != g) & pl.col("_g").is_not_null() & ~pl.col("_g").str.starts_with("darpa"))
        t = transition_table(tr, alpha)
        i = groups == g
        for v in VARIANTS:
            lat = preds[f"{v}_latest"].to_numpy()[i].astype(int)
            prog = preds[f"{v}_progress"].to_numpy()[i].astype(int)
            out[f"{name}_{v}"][i] = t[lat, np.maximum(prog, lat)][:, 1:]
    cols = []
    for k, m in out.items():
        m = np.where(m.sum(1, keepdims=True) > 0, m, 1.0 / (N_STAGES - 1))
        cols += [pl.Series(f"{k}_next{s}", m[:, s - 1]) for s in range(1, N_STAGES)]
    return preds.with_columns(cols)


def step_anchors(preds: pl.DataFrame, ep: pl.DataFrame, *, lead: int) -> pl.DataFrame:
    """One row per transition: the forecast at the host's last anchor <= start - lead."""
    tr = ep.filter(pl.col("transition")).with_columns((pl.col("start") - lead).alias("_q")).sort("_q")
    p = preds.sort("window").with_columns(pl.col("window").alias("_aw"))
    j = tr.join_asof(p.drop(["next_stage"]), left_on="_q", right_on="window", by=["timeline", "host"],
                     strategy="backward")
    # The forecast must come from the model that held out the step's own recording: an
    # anchor in an earlier group (e.g. the previous day) was scored by a model trained on it.
    cov = preds.select(["timeline", "window", pl.col("group").alias("_sg")]).unique(subset=["timeline", "window"]).sort("window")
    j = j.sort("start").join_asof(cov, left_on="start", right_on="window", by="timeline", strategy="backward",
                                  suffix="_cov")
    return j.filter(pl.col("_aw").is_not_null() & (pl.col("group") == pl.col("_sg")))


def _markov(train_steps: pl.DataFrame) -> np.ndarray:
    t = np.ones((N_STAGES, N_STAGES, N_STAGES)) * 0.1
    for prev, prog, s in train_steps.select(["prev_stage", "progress_before", "stage"]).iter_rows():
        t[int(prev), int(prog), int(s)] += 1
    return t


def _topk(p: np.ndarray, truth: np.ndarray, k: int) -> np.ndarray:
    order = np.argsort(-p, axis=1)[:, :k] + 1
    return (order == truth[:, None]).any(1)


def next_stage_scores(steps: pl.DataFrame, ep_all: pl.DataFrame, group_of_timeline: dict) -> dict:
    """Top-1/top-2 accuracy at chain steps, with controls trained on the other groups."""
    if steps.height == 0:
        return {"n": 0}
    truth = steps["stage"].to_numpy().astype(int)
    prev = steps["prev_stage"].fill_null(0).to_numpy().astype(int)
    prog = steps["progress_before"].fill_null(0).to_numpy().astype(int)
    groups = steps["group"].to_numpy()
    all_chain = ep_all.filter(pl.col("transition") & pl.col("has_history"))
    all_chain = all_chain.with_columns(pl.col("timeline").replace_strict(group_of_timeline, default=None).alias("_g"))
    mk = np.zeros((len(steps), N_STAGES - 1))
    maj = np.zeros((len(steps), N_STAGES - 1))
    for g in np.unique(groups):
        tr = all_chain.filter((pl.col("_g") != g) & pl.col("_g").is_not_null() & ~pl.col("_g").str.starts_with("darpa"))
        t = _markov(tr)
        i = groups == g
        row = t[prev[i], prog[i]].copy()
        row[:, 0] = 0
        row[np.arange(i.sum()), prev[i]] = 0
        mk[i] = row[:, 1:]
        counts = np.bincount(tr["stage"].to_numpy().astype(int), minlength=N_STAGES)[1:].astype(float) + 1e-3
        maj[i] = counts
    out = {"n": int(len(steps))}
    preds = {"attack_prior": prior_next(prev, prog), "markov": mk, "majority": maj}
    if "deployable_latest" in steps.columns:  # the same rule at the tracker's inferred state
        dl = steps["deployable_latest"].fill_null(0).to_numpy().astype(int)
        dp = np.maximum(steps["deployable_progress"].fill_null(0).to_numpy().astype(int), dl)
        preds["attack_prior_deployable"] = prior_next(dl, dp)
    extra = sorted({c[: -len("_next1")] for c in steps.columns if c.endswith("_next1")} - set(VARIANTS))
    for v in (*VARIANTS, *extra):
        if f"{v}_next1" in steps.columns:
            preds[f"kcwm_{v}"] = steps.select([f"{v}_next{s}" for s in range(1, N_STAGES)]).to_numpy()
    for name, p in preds.items():
        out[name] = {"top1": float(_topk(p, truth, 1).mean()), "top2": float(_topk(p, truth, 2).mean())}
    return out


def hazard_scores(preds: pl.DataFrame, steps: pl.DataFrame) -> dict:
    out = {}
    for h in HORIZONS:
        v = preds[f"v_{h}"].to_numpy()
        y = preds[f"y_{h}"].to_numpy()[v]
        r = {"n": int(v.sum()), "base_rate": float(y.mean()) if v.any() else float("nan")}
        for var in VARIANTS:
            s = preds[f"{var}_haz_{h}"].to_numpy()[v]
            r[var] = {"auprc": float(average_precision_score(y, s)) if 0 < y.sum() < y.size else float("nan")}
            neg = s[~y]
            comp_steps = steps.filter(pl.col("compromise") & (pl.col("next_in") <= HORIZONS[h]))
            ss = comp_steps[f"{var}_haz_{h}"].to_numpy()
            for budget in (0.01, 0.05):
                thr = float(np.quantile(neg, 1 - budget)) if neg.size else float("inf")
                r[var][f"steps_warned_at_{budget}"] = f"{int((ss >= thr).sum())}/{ss.size}"
        out[h] = r
    return out


def evaluate(preds: pl.DataFrame, ep: pl.DataFrame, *, leads=(1, 360)) -> dict:
    group_of_timeline = dict(preds.select(["timeline", "group"]).unique(subset=["timeline"]).iter_rows())
    ds_of = {tl: ("ctu13" if g.startswith("ctu") else "cicids2017" if g.startswith("cic17") else
                  "darpa2000" if g.startswith("darpa") else g)
             for tl, g in group_of_timeline.items()}
    res = {}
    for lead in leads:
        st = step_anchors(preds, ep, lead=lead).with_columns(
            pl.col("timeline").replace_strict(ds_of, default=None).alias("dataset"))
        chain = st.filter(pl.col("has_history"))
        first = st.filter(~pl.col("has_history"))
        r = {"chain_steps": {"all": next_stage_scores(chain, ep, group_of_timeline)},
             "first_steps": {"all": next_stage_scores(first, ep, group_of_timeline)}}
        for ds in sorted(chain["dataset"].unique().to_list()):
            r["chain_steps"][ds] = next_stage_scores(chain.filter(pl.col("dataset") == ds), ep, group_of_timeline)
        r["hazard"] = {"all": hazard_scores(preds, st)}
        for ds in sorted(st["dataset"].unique().to_list()):
            tl = [t for t, d in ds_of.items() if d == ds]
            r["hazard"][ds] = hazard_scores(preds.filter(pl.col("timeline").is_in(tl)), st.filter(pl.col("dataset") == ds))
        res[f"lead_{lead}"] = r
    return res


def step_table(preds: pl.DataFrame, ep: pl.DataFrame, timelines: list[str], *, lead: int = 1) -> pl.DataFrame:
    """Readable per-step table (for case studies such as DAPT2020 and DARPA 2000)."""
    st = step_anchors(preds.filter(pl.col("timeline").is_in(timelines)), ep, lead=lead)
    names = dict(enumerate(STAGES))
    rows = []
    for r in st.sort(["timeline", "start"]).iter_rows(named=True):
        p = np.array([r[f"deployable_next{s}"] for s in range(1, N_STAGES)])
        o = np.array([r[f"oracle_next{s}"] for s in range(1, N_STAGES)])
        rows.append({"timeline": r["timeline"], "host": r["host"], "start": r["start"],
                     "from": names.get(r["prev_stage"], "-") if r["prev_stage"] is not None else "-",
                     "to": names[r["stage"]], "predicted": names[int(p.argmax()) + 1], "p_true": float(p[r["stage"] - 1]),
                     "oracle_predicted": names[int(o.argmax()) + 1],
                     "prior_predicted": names[int(prior_next(np.array([r["prev_stage"] or 0]), np.array([r["progress_before"] or 0]))[0].argmax()) + 1],
                     "haz_24h": r["deployable_haz_24h"], "anchor_lead_min": (r["start"] - r["window"]) / 6})
    return pl.DataFrame(rows)

"""Gate G5: kill-chain stage metrics.

v1's tactic head scored 18.7% on attack steps, below "the current tactic continues" (52%),
and 19.1% on next-stage prediction. v2 reports three things, each against controls that use
the same information:

1. **Nowcast**: macro-F1 of the current-stage belief on test anchors.
2. **Attack-step stage accuracy** (v1's metric): for each future window t+k (k <= K) that
   is an attack window, is the most likely *attack* stage in the forecast marginal the true
   one? Control: the nowcast carried forward.
3. **Next new stage at transitions**: at every stage onset r (an attack stage appearing
   that was absent for the previous 5 minutes), take the forecast made at t = r - lead.
   Predict the most likely stage *other than the one active now* over the horizon. Controls:
   the Markov table P(next new stage | current stage, progress) counted from training labels,
   and the most frequent onset stage. The carried-forward nowcast is right by construction
   only when nothing changes, so it cannot score here.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

from ..model.world_model import S


def stage_report(arr, fc: dict, eval_rows: np.ndarray, train_rows: np.ndarray, stage_onset: np.ndarray,
                 *, context: int, K: int) -> dict:
    """All G5 metrics for one evaluation set. ``fc`` is ``forecast_rows`` output with ``rows``."""
    pos = {int(r): i for i, r in enumerate(fc["rows"])}
    rows = np.array([r for r in eval_rows if int(r) in pos])
    idx = np.array([pos[int(r)] for r in rows])
    if rows.size == 0:
        return {}
    sm, now, prog = fc["stage_marg"][idx], fc["nowcast"][idx], fc["progress"][idx]
    table = markov_table(arr.stage, arr.session, stage_onset, train_rows, context)
    out = {"nowcast_macro_f1": nowcast_f1(now, arr.stage[rows]),
           "attack_steps": attack_step_accuracy(sm, now, rows, arr.stage, arr.remaining, K)}
    for lead in (1, 6):
        ns = next_stage_at_transitions(sm, now, prog, rows, arr.stage, arr.session, stage_onset, table, lead=lead)
        ns.pop("detail", None)
        out[f"next_stage_lead{lead}"] = ns
    return out


def nowcast_f1(nowcast: np.ndarray, stage: np.ndarray) -> float:
    return float(f1_score(stage, nowcast.argmax(-1), average="macro", labels=np.unique(stage)))


def attack_step_accuracy(stage_marg: np.ndarray, nowcast: np.ndarray, anchors: np.ndarray,
                         stage: np.ndarray, remaining: np.ndarray, K: int) -> dict:
    """stage_marg (n, K, S) aligned with anchors; returns model vs carried-forward accuracy."""
    hit_m = hit_c = n = 0
    carried = nowcast[:, 1:].argmax(-1) + 1  # most likely attack stage now
    for i, a in enumerate(anchors):
        for k in range(1, min(K, int(remaining[a])) + 1):
            s = stage[a + k]
            if s == 0:
                continue
            n += 1
            hit_m += int(stage_marg[i, k - 1, 1:].argmax() + 1 == s)
            hit_c += int(carried[i] == s)
    return {"n_attack_steps": n, "model": hit_m / n if n else float("nan"),
            "nowcast_carried_forward": hit_c / n if n else float("nan")}


def markov_table(stage: np.ndarray, session: np.ndarray, onset: np.ndarray, rows: np.ndarray,
                 context: int) -> np.ndarray:
    """Counts of (current stage, progress) -> next new stage, from labelled training rows."""
    table = np.ones((S, S, S))  # Laplace
    for r in rows[onset[rows]]:
        a = r - 1
        if a < 0 or session[a] != session[r]:
            continue
        lo = max(0, a - context + 1)
        prog = int(stage[lo : a + 1][session[lo : a + 1] == session[a]].max())
        table[stage[a], prog, stage[r]] += 1
    return table


def next_stage_at_transitions(stage_marg: np.ndarray, nowcast: np.ndarray, progress: np.ndarray,
                              anchors: np.ndarray, stage: np.ndarray, session: np.ndarray,
                              onset: np.ndarray, table: np.ndarray, lead: int = 1) -> dict:
    """Accuracy of the predicted next new stage, for onsets whose anchor (r - lead) was forecast."""
    pos = {int(a): i for i, a in enumerate(anchors)}
    prior = table.sum(axis=(0, 1))
    prior[0] = 0
    hits = {"model": 0, "markov": 0, "marginal": 0}
    n = 0
    detail = []
    for r in np.flatnonzero(onset):
        a = r - lead
        if a not in pos or session[a] != session[r]:
            continue
        i = pos[a]
        cur = int(nowcast[i].argmax())
        mass = stage_marg[i].sum(0).copy()
        mass[0] = 0
        if cur > 0:
            mass[cur] = 0
        pred = int(mass.argmax())
        row = table[stage[a], int(progress[i])].copy()
        row[0] = 0
        if stage[a] > 0:
            row[stage[a]] = 0
        markov = int(row.argmax())
        marg = int(np.where(np.arange(S) == stage[a], 0, prior).argmax())
        truth = int(stage[r])
        n += 1
        hits["model"] += pred == truth
        hits["markov"] += markov == truth
        hits["marginal"] += marg == truth
        detail.append({"row": int(r), "truth": truth, "model": pred, "markov": markov})
    return {"n_transitions": n, **{k: v / n if n else float("nan") for k, v in hits.items()},
            "lead": lead, "detail": detail}

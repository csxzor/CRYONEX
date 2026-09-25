"""Per-window ground truth: current stage, attack episodes and onsets.

v1 counted an "onset" at every benign->attack flip. On CIC-IDS2017 Friday that made 321 of
437 onsets out of Botnet beacon bursts with a median run of two windows. The metric was
mostly rewarding "predict the next beacon", which is not early warning. v2 separates the two:

* An **episode** is a run of attack windows, merged across gaps of at most
  ``merge_gap`` benign windows. Beacons from one bot fall into one episode.
* An **onset** is the first window of an episode that follows at least ``onset_quiet``
  benign windows in the same session. An episode already running at the start of a session
  has no onset, because nothing observable preceded it.
* A **recurrence** is a benign->attack flip *inside* an episode (the next beacon burst).
  It is reported separately and is never counted as early warning.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ..stages import STAGE_INDEX, STAGES


def window_labels(flows: pl.DataFrame) -> pl.DataFrame:
    """Aggregate labelled flows (with ``window``) into per-window stage labels.

    ``stage_now`` is the furthest kill-chain stage present in the window: a window holding
    both a scan and a brute force is labelled by the brute force, since that is how far the
    adversary got. ``family_now`` is the attack family with the most flows.
    """
    attack = flows.filter(pl.col("is_attack"))
    base = flows.group_by("window").agg(pl.len().cast(pl.Int32).alias("n_flows"))
    if attack.height == 0:
        return base.with_columns(
            pl.lit(0, dtype=pl.Int32).alias("n_attack_flows"),
            pl.lit(0, dtype=pl.Int8).alias("stage_now"),
            pl.lit(None, dtype=pl.Utf8).alias("family_now"),
            pl.lit(False).alias("attempted_only"),
        )
    stage_idx = pl.col("stage").replace_strict(STAGE_INDEX, return_dtype=pl.Int8)
    per_window = attack.with_columns(stage_idx.alias("_s")).group_by("window").agg(
        pl.len().cast(pl.Int32).alias("n_attack_flows"),
        pl.col("_s").max().alias("stage_now"),
        pl.col("family").mode().first().alias("family_now"),
        pl.col("attempted").all().alias("attempted_only"),
    )
    return base.join(per_window, on="window", how="left").with_columns(
        pl.col("n_attack_flows").fill_null(0),
        pl.col("stage_now").fill_null(0),
        pl.col("attempted_only").fill_null(False),
    )


def episodes(
    is_attack: np.ndarray,
    session: np.ndarray,
    *,
    merge_gap: int,
    onset_quiet: int,
) -> dict[str, np.ndarray]:
    """Episode ids, onsets and recurrences over a time-ordered window sequence.

    Arrays are aligned with the input (one entry per window, sorted by session then window).
    ``episode`` is -1 outside episodes; a merged gap's benign windows belong to the episode
    (``in_episode``) but are not attack windows.
    """
    is_attack = np.asarray(is_attack, dtype=bool)
    session = np.asarray(session)
    n = is_attack.size
    episode = np.full(n, -1, dtype=np.int32)
    onset = np.zeros(n, dtype=bool)
    recurrence = np.zeros(n, dtype=bool)
    next_id = 0

    for sid in np.unique(session):
        idx = np.flatnonzero(session == sid)
        att = is_attack[idx]
        hits = np.flatnonzero(att)
        if hits.size == 0:
            continue
        # Split attack windows into episodes wherever the benign gap exceeds merge_gap.
        gaps = np.diff(hits) - 1
        cuts = np.flatnonzero(gaps > merge_gap)
        starts = np.concatenate([[hits[0]], hits[cuts + 1]])
        ends = np.concatenate([hits[cuts], [hits[-1]]])
        prev_end = -1
        for s, e in zip(starts, ends):
            episode[idx[s : e + 1]] = next_id
            quiet_before = s if prev_end < 0 else s - prev_end - 1
            if quiet_before >= onset_quiet:
                onset[idx[s]] = True
            flips = s + 1 + np.flatnonzero(att[s + 1 : e + 1] & ~att[s:e])
            recurrence[idx[flips]] = True
            prev_end = e
            next_id += 1

    return {
        "episode": episode,
        "in_episode": episode >= 0,
        "is_onset": onset,
        "is_recurrence": recurrence,
    }


def stage_onsets(stage_now: np.ndarray, session: np.ndarray, *, quiet: int) -> np.ndarray:
    """First window of an attack stage not seen in the previous ``quiet`` windows.

    These are the kill-chain *transitions*: a port scan starting while a botnet is already
    beaconing, or a DDoS following a scan. Episode onsets miss them because the episode was
    already running. They anchor the stage-forecast metrics (G5) and per-stage lead time.
    """
    stage_now = np.asarray(stage_now, dtype=np.int64)
    session = np.asarray(session)
    out = np.zeros(stage_now.size, dtype=bool)
    for sid in np.unique(session):
        idx = np.flatnonzero(session == sid)
        last_seen: dict[int, int] = {}
        for j, i in enumerate(idx):
            s = int(stage_now[i])
            if s > 0:
                if s not in last_seen or j - last_seen[s] > quiet:
                    out[i] = True
                last_seen[s] = j
    return out


def stage_names(indices: np.ndarray) -> list[str]:
    return [STAGES[int(i)] for i in np.asarray(indices).ravel()]

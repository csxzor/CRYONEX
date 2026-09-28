"""Build the per-window table for one capture: features, masks, labels, episodes, targets.

A *capture* is one continuous recording, such as one CIC CSV day or one CTU-13 scenario.
The output has one row per window of every session's contiguous spine:

* ``dataset, capture, session, session_key, window, t``
* every feature in ``registry.FEATURE_NAMES`` (raw units; scaling happens later, per split)
* masks ``m_packet, m_ip, m_lookback`` and ``warmup``
* labels ``n_flows, n_attack_flows, stage_now, family_now, attempted_only``
* episodes ``episode, in_episode, is_onset, is_recurrence``
* targets ``y_k, valid_k`` for every horizon, ``time_to_compromise``, ``next_attack_stage``
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ..ingest.windowing import assign_windows, attach_sessions, session_spine
from ..targets.episodes import episodes, stage_onsets, window_labels
from ..targets.targets import forecast_targets
from .extra import extra_features
from .registry import FEATURE_NAMES, PACKET_FEATURES
from .vector import base_features


def refine_stages(flows: pl.DataFrame, internal_cidrs: list[str]) -> pl.DataFrame:
    """Apply the one ATT&CK rule a per-label map cannot express: lateral movement is
    internal-to-internal by definition. A flow labelled LateralMovement whose source is
    outside the network is reconnaissance (external scanning).

    Found by the G1 audit: CIC-IDS2017's corrected release labels the attacker's external
    scan of 192.168.10.51 (from the NAT address 172.16.0.1, 14:00 local) "Infiltration -
    Portscan", the same label as the compromised victim's internal scan that begins at 14:33,
    after the 14:19 exploit.
    """
    if not internal_cidrs:
        return flows
    from .extra import is_internal

    external_src = ~is_internal(pl.col("src_ip"), internal_cidrs)
    fix = (pl.col("stage") == "LateralMovement") & external_src
    return flows.with_columns(
        pl.when(fix).then(pl.lit("Reconnaissance")).otherwise(pl.col("stage")).alias("stage"),
        pl.when(fix).then(pl.lit("T1595")).otherwise(pl.col("technique")).alias("technique"),
    )


def build_capture(
    flows: pl.DataFrame,
    *,
    dataset: str,
    capture: str,
    cfg: dict,
    internal_cidrs: list[str],
    packet_block: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return ``(windowed_flows, windows)`` for one capture."""
    wcfg = cfg["window"]
    seconds = float(wcfg["seconds"])
    gap = max(1, int(wcfg["session_gap_seconds"] / seconds))

    flows = refine_stages(flows, internal_cidrs)
    flows = assign_windows(flows, seconds)
    spine = session_spine(flows["window"].to_numpy(), gap)
    flows = attach_sessions(flows, spine)

    feats = base_features(flows, seconds).join(
        extra_features(flows, internal_cidrs), on="window", how="left"
    )
    labels = window_labels(flows)
    win = spine.join(feats, on="window", how="left").join(labels, on="window", how="left")

    if packet_block is not None and packet_block.height:
        from ..ingest.packet_merge import merge

        win = merge(win, packet_block)

    has_ips = bool(internal_cidrs) and flows["src_ip"].n_unique() > 1
    win = win.sort("window").with_columns(
        *[pl.col(c).cast(pl.Float64).fill_null(0.0).fill_nan(0.0) for c in FEATURE_NAMES],
        pl.col("m_packet").fill_null(0.0),
        pl.col("n_flows").fill_null(0).cast(pl.Int32),
        pl.col("n_attack_flows").fill_null(0).cast(pl.Int32),
        pl.col("stage_now").fill_null(0).cast(pl.Int8),
        pl.col("attempted_only").fill_null(False),
    )
    # Absent packet measurements stay zero, and the mask says so.
    win = win.with_columns(
        [pl.when(pl.col("m_packet") > 0).then(pl.col(c)).otherwise(0.0).alias(c) for c in PACKET_FEATURES]
    )

    lookback = int(wcfg["context"])
    warmup = int(wcfg["warmup_windows"])
    win = win.with_columns(
        (pl.col("window") - pl.col("window").min().over("session")).alias("_pos"),
    ).with_columns(
        pl.lit(1.0 if has_ips else 0.0).alias("m_ip"),
        (pl.col("_pos") >= lookback).cast(pl.Float64).alias("m_lookback"),
        (pl.col("_pos") < warmup).alias("warmup"),
        pl.lit(dataset).alias("dataset"),
        pl.lit(capture).alias("capture"),
        (pl.lit(f"{capture}#") + pl.col("session").cast(pl.Utf8)).alias("session_key"),
        (pl.col("window").cast(pl.Float64) * seconds).alias("t"),
    ).drop("_pos")

    return flows, add_targets(win, cfg)


TARGET_COLUMNS = ["episode", "in_episode", "is_onset", "is_recurrence", "is_stage_onset",
                  "time_to_compromise", "next_attack_stage", "pos_in_session"]


def add_targets(win: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    """(Re)compute episodes and forecast targets from ``stage_now`` and ``session``.

    They depend on nothing else, so a change to a target definition needs no re-read of the
    raw flows (``kcwm retarget``).
    """
    wcfg = cfg["window"]
    horizons = list(wcfg["horizons"])
    drop = [c for c in win.columns
            if c in TARGET_COLUMNS or c.startswith("y_") or c.startswith("valid_")]
    win = win.drop(drop).sort("window")
    stage_now = win["stage_now"].to_numpy()
    session = win["session"].to_numpy()
    ep = episodes(
        stage_now > 0, session,
        merge_gap=int(wcfg["episode_merge_gap"]), onset_quiet=int(wcfg["onset_quiet"]),
    )
    ep["is_stage_onset"] = stage_onsets(stage_now, session, quiet=int(wcfg["onset_quiet"]))
    tg = forecast_targets(stage_now, session, horizons)
    return win.with_columns(
        [pl.Series(k, v) for k, v in ep.items()] + [pl.Series(k, v) for k, v in tg.items()]
    )


def summarise(win: pl.DataFrame) -> dict:
    stage = win["stage_now"].to_numpy()
    return {
        "windows": int(win.height),
        "sessions": int(win["session"].n_unique()),
        "attack_windows": int((stage > 0).sum()),
        "episodes": int(win.filter(pl.col("episode") >= 0)["episode"].n_unique()),
        "onsets": int(win["is_onset"].sum()),
        "stage_onsets": int(win["is_stage_onset"].sum()),
        "recurrences": int(win["is_recurrence"].sum()),
        "packet_coverage": float(win["m_packet"].mean()),
        "stages": {int(s): int(n) for s, n in zip(*np.unique(stage, return_counts=True))},
    }

"""Campaign synthesizer: real attack episodes, stitched in kill-chain order into real benign traffic.

Why this exists: public captures schedule attacks independently (a DoS at 9:47, a brute
force the next day), so almost nothing in them *precedes* a compromise. Every model's
early-warning AUPRC on real onsets sits at the base rate (docs/BENCHMARKS.md). A world
model can only learn "recon, then access, then C2" if such sequences exist in its data.
This module builds them from genuine traffic:

* **Snippets.** Attack flows cut per (capture, stage) wherever the stage pauses for more
  than 5 minutes, each tagged with the P1 split its start falls in.
* **Backgrounds.** Benign stretches (no attack flows) of the same network and split.
* **Campaigns.** A stage sequence drawn from a kill-chain grammar, with random gaps (1-15
  min), truncation (>= 40% stop early), decoys (recon only) and benign-only runs. Each stage
  takes a snippet of that stage from the same network and split, time-shifted into the
  background. IPs are remapped so that one background host is *the* victim throughout: the
  target of recon and access, the source of C2 beacons and the lateral scan.

Guards against learning artefacts instead of attacks:

* same network only (no cross-dataset scale jumps);
* same split only (test campaigns never reuse training snippets or backgrounds);
* CSV flows only, so no PCAP packet block exists and campaign captures carry
  ``m_packet = 0`` throughout, instead of packet features that fail to react to injected
  flows.

Headline numbers stay on real data. Campaigns are training augmentation, plus the P4
benchmark, reported separately.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from ..features.extra import is_internal
from ..stages import STAGE_INDEX

GRAMMAR_NEXT = {  # stage -> [(next stage, weight)], "END" stops the campaign
    "START": [("Reconnaissance", 0.7), ("InitialAccess", 0.3)],
    "Reconnaissance": [("InitialAccess", 0.6), ("END", 0.4)],
    "InitialAccess": [("CommandAndControl", 0.55), ("Impact", 0.15), ("END", 0.3)],
    "CommandAndControl": [("LateralMovement", 0.45), ("Impact", 0.25), ("Exfiltration", 0.1), ("END", 0.2)],
    "LateralMovement": [("CommandAndControl", 0.3), ("Exfiltration", 0.2), ("Impact", 0.2), ("END", 0.3)],
    "Exfiltration": [("END", 1.0)],
    "Impact": [("END", 1.0)],
}


@dataclass
class Snippet:
    capture: str
    stage: str
    family: str
    split: str
    t0: float
    t1: float
    flows: pl.DataFrame  # attack flows only, original timestamps and IPs


def extract_snippets(flows: pl.DataFrame, capture: str, split_of_window, *, gap: float = 300.0,
                     max_len: float = 900.0) -> list[Snippet]:
    """Cut a capture's attack flows into per-stage snippets (at most ``max_len`` seconds)."""
    att = flows.filter(pl.col("is_attack")).sort("ts")
    out = []
    for stage in att["stage"].unique().to_list():
        s = att.filter(pl.col("stage") == stage)
        ts = s["ts"].to_numpy()
        cuts = np.flatnonzero(np.diff(ts) > gap)
        starts = np.concatenate([[0], cuts + 1])
        ends = np.concatenate([cuts, [len(ts) - 1]])
        for a, b in zip(starts, ends):
            t0 = ts[a]
            t1 = min(ts[b], t0 + max_len)
            seg = s.filter(pl.col("ts").is_between(t0, t1))
            if seg.height < 3:
                continue
            fam = seg["family"].mode().first()
            out.append(Snippet(capture, stage, fam, split_of_window(int(t0 // 10)), float(t0), float(t1), seg))
    return out


def benign_backgrounds(win: pl.DataFrame, split_of_window, *, length: float,
                       seconds: float = 10.0) -> list[tuple[str, float, float]]:
    """(split, t0, t1): non-overlapping runs of ``length`` seconds of attack-free windows,
    contiguous inside one session and one split (from the capture's window spine)."""
    n = int(length // seconds)
    w = win.sort("window")
    ws = w["window"].to_numpy()
    benign = w["n_attack_flows"].to_numpy() == 0  # idle windows are benign states too
    sess = w["session"].to_numpy()
    out, i = [], 0
    while i + n <= ws.size:
        j = i + n
        block = slice(i, j)
        ok = (benign[block].all() and sess[i] == sess[j - 1] and ws[j - 1] - ws[i] == n - 1
              and split_of_window(int(ws[i])) == split_of_window(int(ws[j - 1])))
        if ok:
            out.append((split_of_window(int(ws[i])), float(ws[i] * seconds), float((ws[j - 1] + 1) * seconds)))
            i = j
        else:
            i += max(1, n // 8)
    return out


def sample_stages(rng: np.random.Generator, *, decoy: float = 0.15, benign_only: float = 0.1) -> list[str]:
    u = rng.random()
    if u < benign_only:
        return []
    if u < benign_only + decoy:
        return ["Reconnaissance"]
    seq, cur = [], "START"
    while True:
        nxt, w = zip(*GRAMMAR_NEXT[cur])
        cur = rng.choice(nxt, p=np.asarray(w) / sum(w))
        if cur == "END" or cur in seq:
            return seq
        seq.append(cur)


def _remap(snip: Snippet, victim: str, others: list[str], cidrs: list[str], rng: np.random.Generator) -> pl.DataFrame:
    """Make ``victim`` the internal party of the snippet; other internal hosts map to ``others``."""
    f = snip.flows
    internal = set(
        f.select(pl.concat_list("src_ip", "dst_ip").explode().alias("ip")).unique()
        .filter(is_internal(pl.col("ip"), cidrs))["ip"].to_list()
    )
    src_int = f.filter(is_internal(pl.col("src_ip"), cidrs))
    # the "pivot" internal host: the busiest internal source (C2/lateral), else busiest internal target
    if src_int.height:
        pivot = src_int["src_ip"].mode().first()
    else:
        dst_int = f.filter(is_internal(pl.col("dst_ip"), cidrs))
        pivot = dst_int["dst_ip"].mode().first() if dst_int.height else None
    mapping = {}
    pool = [h for h in others if h != victim]
    for ip in internal:
        if ip == pivot:
            mapping[ip] = victim
        else:
            mapping[ip] = pool[int(rng.integers(len(pool)))] if pool else victim
    return f.with_columns(
        pl.col("src_ip").replace(mapping).alias("src_ip"),
        pl.col("dst_ip").replace(mapping).alias("dst_ip"),
    )


def build_campaign(background: pl.DataFrame, snippets_by_stage: dict[str, list[Snippet]], rng: np.random.Generator,
                   *, cidrs: list[str], lead_in: float = 1800.0) -> tuple[pl.DataFrame, list[dict]]:
    """Inject a sampled kill chain into ``background`` (benign flows). Returns flows and a plan."""
    # Re-draw (up to 5 times) until at least two stages of the chain have snippets in this
    # pool; benign-only and recon-decoy draws are accepted as they come.
    for _ in range(5):
        raw = sample_stages(rng)
        seq = [s for s in raw if snippets_by_stage.get(s)]
        if len(raw) <= 1 or len(seq) >= 2:
            break
    t_start, t_end = float(background["ts"].min()), float(background["ts"].max())
    hosts = (background.filter(is_internal(pl.col("src_ip"), cidrs))
             .group_by("src_ip").len().filter(pl.col("len") > 20)["src_ip"].to_list())
    if not seq or not hosts:
        return background, []
    victim = hosts[int(rng.integers(len(hosts)))]
    t = t_start + lead_in + rng.uniform(0, 300)
    parts, plan = [background], []
    for stage in seq:
        pool = snippets_by_stage[stage]
        snip = pool[int(rng.integers(len(pool)))]
        dur = min(snip.t1 - snip.t0, rng.uniform(120, 480))
        if t + dur > t_end - 60:
            break
        f = snip.flows.filter(pl.col("ts") <= snip.t0 + dur)
        f = _remap(Snippet(snip.capture, snip.stage, snip.family, snip.split, snip.t0, snip.t0 + dur, f),
                   victim, hosts, cidrs, rng)
        shift = t - snip.t0
        f = f.with_columns((pl.col("ts") + shift).alias("ts"), (pl.col("te") + shift).alias("te"))
        parts.append(f)
        plan.append({"stage": stage, "stage_index": STAGE_INDEX[stage], "family": snip.family,
                     "source": snip.capture, "t0": t, "t1": t + dur, "victim": victim})
        t = t + dur + rng.uniform(60, 300)
    cols = background.columns
    return pl.concat([p.select(cols) for p in parts]).sort("ts"), plan

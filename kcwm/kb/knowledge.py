"""Offline ATT&CK / D3FEND knowledge base: stage -> techniques -> response playbook.

Both sources are committed JSON (``scripts/fetch_kb.py``): the ATT&CK subset is extracted
from MITRE's Enterprise STIX bundle, and the countermeasures come from MITRE's D3FEND API.
Nothing here is written from memory.

D3FEND has no mapping for three techniques KC-WM can emit: T1046 (network service
discovery), T1595 (active scanning) and T1496 (resource hijacking). For those,
``CURATED_FALLBACK`` lists countermeasures chosen by us; every ID in it appears verbatim in
D3FEND's mappings for other techniques (checked in ``tests/test_kb.py``). The UI labels them
"curated".
"""

from __future__ import annotations

import functools
import json

from ..config import REPO_ROOT
from ..stages import STAGE_INDEX, STAGES, stage_map_config

CURATED_FALLBACK = {
    "T1046": ["D3-CAA", "D3-NTF", "D3-NTCD"],
    "T1595": ["D3-CAA", "D3-ITF", "D3-NTCD"],
    "T1496": ["D3-PHDURA", "D3-OTF", "D3-NTF"],
}
# Display preference: network-observable responses first (what a SOC can do from traffic).
PREFERENCE = ["D3-NTF", "D3-ITF", "D3-OTF", "D3-CAA", "D3-ISVA", "D3-PHDURA", "D3-NTCD",
              "D3-RTSD", "D3-DNSDL", "D3-DNSTA", "D3-ST", "D3-ANCI", "D3-MFA", "D3-CTS",
              "D3-APCA", "D3-NTSA"]
TACTIC_ORDER = {"Isolate": 0, "Detect": 1, "Evict": 2, "Harden": 3}

# Default technique per stage when a window mixes several (from stage_map defaults).
STAGE_TECHNIQUES = {
    "Reconnaissance": ["T1595", "T1046"],
    "InitialAccess": ["T1110", "T1190", "T1204"],
    "LateralMovement": ["T1021", "T1046"],
    "CommandAndControl": ["T1071", "T1105", "T1571"],
    "Exfiltration": ["T1041", "T1048"],
    "Impact": ["T1498", "T1499", "T1496"],
}


@functools.lru_cache(maxsize=1)
def attack() -> dict:
    return json.loads((REPO_ROOT / "third_party/attack_subset.json").read_text())


@functools.lru_cache(maxsize=1)
def d3fend() -> dict:
    return json.loads((REPO_ROOT / "third_party/d3fend_subset.json").read_text())


@functools.lru_cache(maxsize=1)
def _d3fend_catalogue() -> dict[str, dict]:
    cat = {}
    for items in d3fend()["mappings"].values():
        for it in items:
            cat.setdefault(it["id"], it)
    return cat


def technique(tid: str) -> dict:
    t = attack()["techniques"].get(tid)
    if t is None:
        return {"id": tid, "name": tid, "description": "", "url": f"https://attack.mitre.org/techniques/{tid}/"}
    return {"id": tid, **t}


def countermeasures(tid: str, limit: int = 4) -> list[dict]:
    """Ranked D3FEND countermeasures: at most one per D3FEND tactic first, network-observable first."""
    mapped = d3fend()["mappings"].get(tid, [])
    curated = False
    if not mapped:
        cat = _d3fend_catalogue()
        mapped = [cat[i] for i in CURATED_FALLBACK.get(tid, []) if i in cat]
        curated = True
    rank = {d: i for i, d in enumerate(PREFERENCE)}
    ordered = sorted(mapped, key=lambda m: (rank.get(m["id"], 99), TACTIC_ORDER.get(m["tactic"], 9)))
    picked, tactics = [], set()
    for m in ordered:  # diversity: one per tactic first
        if m["tactic"] not in tactics:
            picked.append(m)
            tactics.add(m["tactic"])
    for m in ordered:
        if len(picked) >= limit:
            break
        if m not in picked:
            picked.append(m)
    return [{**m, "curated": curated} for m in picked[:limit]]


def stage_card(stage: str) -> dict:
    """Everything the Respond panel shows for a stage."""
    meta = stage_map_config()["stages"][stage]
    tactic_id = meta.get("attack_id")
    tactic = attack()["tactics"].get(tactic_id, {}) if tactic_id else {}
    techs = []
    for tid in STAGE_TECHNIQUES.get(stage, []):
        t = technique(tid)
        techs.append({**t, "countermeasures": countermeasures(tid)})
    return {"stage": stage, "index": STAGE_INDEX[stage], "tactic_id": tactic_id,
            "tactic_name": tactic.get("name", stage), "tactic_url": tactic.get("url"),
            "description": meta.get("description", ""), "techniques": techs}


def all_stage_cards() -> dict[str, dict]:
    return {s: stage_card(s) for s in STAGES[1:]}

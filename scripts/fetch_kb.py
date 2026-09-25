"""Build the offline knowledge base (run once, with network; the outputs are committed).

* ``third_party/attack_subset.json``: the ATT&CK tactics and techniques KC-WM can emit,
  with names, descriptions, URLs and mitigations. Extracted from the full Enterprise STIX
  bundle (``enterprise-attack.json``, ~50 MB, not committed).
* ``third_party/d3fend_subset.json``: the D3FEND countermeasures mapped to each of those
  techniques, fetched from MITRE's D3FEND API
  (``https://d3fend.mitre.org/api/offensive-technique/attack/<id>.json``). This is the
  authoritative mapping, not a hand-written one.

Usage: python scripts/fetch_kb.py [path/to/enterprise-attack.json]
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STIX = Path.home() / "Desktop/sih/third_party/enterprise-attack.json"


def techniques_in_use() -> list[str]:
    cfg = yaml.safe_load((ROOT / "configs/stage_map.yaml").read_text())
    ids = set()

    def walk(node):
        if isinstance(node, dict):
            if "technique" in node and node["technique"]:
                ids.add(node["technique"])
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(cfg)
    ids |= {"T1595", "T1046", "T1071", "T1041", "T1021", "T1048", "T1571"}  # stage defaults
    return sorted(ids)


def attack_subset(stix_path: Path, wanted: list[str]) -> dict:
    bundle = json.loads(stix_path.read_text())
    objs = bundle["objects"]
    by_id = {o["id"]: o for o in objs}

    def ext_id(o):
        for ref in o.get("external_references", []):
            if ref.get("source_name") == "mitre-attack":
                return ref.get("external_id"), ref.get("url")
        return None, None

    tactics, techs = {}, {}
    for o in objs:
        if o.get("revoked") or o.get("x_mitre_deprecated"):
            continue
        eid, url = ext_id(o)
        if o["type"] == "x-mitre-tactic":
            tactics[eid] = {"name": o["name"], "shortname": o["x_mitre_shortname"], "url": url,
                            "description": o.get("description", "").split("\n")[0]}
        elif o["type"] == "attack-pattern" and eid in wanted:
            techs[eid] = {"name": o["name"], "url": url,
                          "description": o.get("description", "").split("\n")[0],
                          "tactics": [p["phase_name"] for p in o.get("kill_chain_phases", [])],
                          "stix_id": o["id"], "mitigations": []}
    stix_to_tech = {v["stix_id"]: k for k, v in techs.items()}
    for o in objs:
        if o["type"] == "relationship" and o.get("relationship_type") == "mitigates":
            tgt = stix_to_tech.get(o["target_ref"])
            src = by_id.get(o["source_ref"])
            if tgt and src and src["type"] == "course-of-action" and not src.get("x_mitre_deprecated"):
                mid, murl = ext_id(src)
                if mid and mid.startswith("M"):
                    techs[tgt]["mitigations"].append({"id": mid, "name": src["name"], "url": murl})
    for t in techs.values():
        t.pop("stix_id")
        t["mitigations"] = sorted({m["id"]: m for m in t["mitigations"]}.values(), key=lambda m: m["id"])
    missing = sorted(set(wanted) - set(techs))
    return {"source": str(stix_path.name), "tactics": tactics, "techniques": techs, "missing": missing}


def d3fend_subset(wanted: list[str]) -> dict:
    out = {}
    for tid in wanted:
        url = f"https://d3fend.mitre.org/api/offensive-technique/attack/{tid}.json"
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    data = json.loads(r.read())
                break
            except Exception as exc:  # pragma: no cover - network
                print(f"  {tid}: retry {attempt + 1} ({exc})")
                time.sleep(3)
        else:
            out[tid] = []
            continue
        seen = {}
        for b in data["off_to_def"]["results"]["bindings"]:
            did = b.get("def_tech_id", {}).get("value")
            if not did:
                continue
            seen.setdefault(did, {
                "id": did,
                "name": b["def_tech_label"]["value"],
                "tactic": b.get("def_tactic_label", {}).get("value"),
                "artifact": b.get("def_artifact_label", {}).get("value"),
                "url": f"https://d3fend.mitre.org/technique/d3f:{b['def_tech_label']['value'].title().replace(' ', '').replace('-', '')}/",
            })
        out[tid] = sorted(seen.values(), key=lambda d: (d["tactic"] or "", d["id"]))
        print(f"  {tid}: {len(out[tid])} countermeasures")
    return {"source": "https://d3fend.mitre.org/api/offensive-technique/attack/<id>.json",
            "fetched": time.strftime("%Y-%m-%d"), "mappings": out}


def main() -> None:
    stix = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_STIX
    wanted = techniques_in_use()
    print("techniques:", wanted)
    (ROOT / "third_party").mkdir(exist_ok=True)
    a = attack_subset(stix, wanted)
    (ROOT / "third_party/attack_subset.json").write_text(json.dumps(a, indent=1))
    print(f"ATT&CK subset: {len(a['techniques'])} techniques, {len(a['tactics'])} tactics, missing {a['missing']}")
    d = d3fend_subset(wanted)
    (ROOT / "third_party/d3fend_subset.json").write_text(json.dumps(d, indent=1))


if __name__ == "__main__":
    main()

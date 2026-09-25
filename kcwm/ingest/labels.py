"""Raw dataset label -> (attack family, kill-chain stage, ATT&CK technique).

Driven by ``configs/stage_map.yaml`` so that every mapping decision is auditable as data.
Ported from netforecast v1 ``ingest/labels.py`` (normalisation and the "- Attempted"
handling are unchanged). v2 adds token rules for CTU-13's composite labels and the DAPT2020
stage vocabulary.

An unmapped label raises instead of silently becoming benign. A mislabelled attack flow that
turns into benign poisons both the training targets and the benchmark, which is the class of
dataset fault the corrected CIC releases exist to fix.
"""

from __future__ import annotations

import functools
import re
import unicodedata
from dataclasses import dataclass

from ..stages import STAGE_INDEX, stage_map_config

BENIGN_FAMILY = "Benign"
BENIGN_STAGE = "Benign"
ATTEMPTED_SUFFIX = "-attempted"


def normalise_label(raw: str | None) -> str:
    """Fold a raw label into a comparison key.

    CIC releases spell the same label several ways across files: en-dashes for hyphens,
    non-breaking spaces, doubled spaces and inconsistent case. This function
    Unicode-normalises, maps every dash variant to ``-``, collapses whitespace and lowercases.
    """
    if raw is None:
        return ""
    text = unicodedata.normalize("NFKC", str(raw))
    text = text.replace("–", "-").replace("—", "-").replace("−", "-").replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return re.sub(r"\s*-\s*", "-", text)


def is_attempted(raw: str | None) -> bool:
    """True for the corrected CIC releases' ``"... - Attempted"`` labels.

    These mark attacks that were launched but did not succeed. They are real adversary
    activity, so they keep their attack stage, but they are flagged so that choice can be
    ablated.
    """
    return normalise_label(raw).endswith(ATTEMPTED_SUFFIX)


@dataclass(frozen=True)
class Resolved:
    family: str
    stage: str
    technique: str | None
    is_attack: bool


BENIGN = Resolved(BENIGN_FAMILY, BENIGN_STAGE, None, False)


def _resolved(spec: dict) -> Resolved:
    stage = spec["stage"]
    if stage not in STAGE_INDEX:
        raise ValueError(f"unknown stage {stage!r} in stage_map.yaml")
    return Resolved(spec["family"], stage, spec.get("technique"), stage != BENIGN_STAGE)


class ExactMapper:
    """Exact (normalised) label lookup with aliases and ``- Attempted`` stripping."""

    def __init__(self, dataset: str, section: dict):
        self.dataset = dataset
        self.exact: dict[str, Resolved] = {}
        for label, spec in section.items():
            resolved = _resolved(spec)
            for name in (label, *spec.get("aliases", [])):
                self.exact[normalise_label(name)] = resolved

    def resolve(self, raw: str | None, **_: object) -> Resolved:
        key = normalise_label(raw)
        if key in self.exact:
            return self.exact[key]
        if key.endswith(ATTEMPTED_SUFFIX):
            base = key[: -len(ATTEMPTED_SUFFIX)]
            if base in self.exact:
                return self.exact[base]
        raise KeyError(
            f"{self.dataset}: unmapped label {raw!r} (normalised {key!r}); "
            "add it to configs/stage_map.yaml"
        )


class CtuMapper:
    """CTU-13 composite labels, classified by their ``-``-separated tokens."""

    def __init__(self, section: dict):
        self.benign_patterns = [p.lower() for p in section["benign_patterns"]]
        self.botnet_pattern = section["botnet_pattern"].lower()
        self.rules = section["rules"]
        self.default = _resolved(section["default"])

    @staticmethod
    def tokens(raw: str) -> list[str]:
        text = normalise_label(raw).removeprefix("flow=")
        return [t for t in re.split(r"[-\s]+", text) if t]

    def resolve(self, raw: str | None, *, scenario: int | None = None) -> Resolved:
        key = normalise_label(raw)
        if self.botnet_pattern not in key:
            if any(p in key for p in self.benign_patterns) or key == "":
                return BENIGN
            raise KeyError(f"ctu13: label {raw!r} is neither botnet, normal nor background")
        tokens = self.tokens(raw)
        for rule in self.rules:
            if "scenarios" in rule and scenario not in rule["scenarios"]:
                continue
            hit = any(t in rule.get("tokens", []) for t in tokens)
            for prefix in rule.get("token_prefixes", []):
                hit = hit or any(
                    t.startswith(prefix) and t[len(prefix):].isdigit() and len(t) > len(prefix)
                    for t in tokens
                )
            if hit:
                return _resolved(rule)
        return self.default


@functools.lru_cache(maxsize=8)
def mapper(dataset: str) -> ExactMapper | CtuMapper:
    """The label mapper for a dataset family: ``cicids``, ``ctu13`` or ``dapt2020``."""
    cfg = stage_map_config()
    if dataset.startswith("cicids"):
        return ExactMapper("cicids", cfg["cicids"])
    if dataset == "ctu13":
        return CtuMapper(cfg["ctu13"])
    if dataset == "dapt2020":
        return ExactMapper("dapt2020", cfg["dapt2020"])
    raise KeyError(f"no label mapping for dataset {dataset!r}")


def lofo_families() -> list[str]:
    return list(stage_map_config()["lofo_families"])

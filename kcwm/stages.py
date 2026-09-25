"""Kill-chain stage vocabulary.

The index of a stage is its position in the ATT&CK matrix order (Reconnaissance first,
Impact last). That ordering is what the world model's kill-chain *progress* variable
``p_t = max stage reached`` means, so it is part of every saved model's contract: append
only, never reorder. ``configs/stage_map.yaml`` is checked against this list on load.
"""

from __future__ import annotations

import functools

import numpy as np
import yaml

from .config import CONFIG_DIR

STAGES: list[str] = [
    "Benign",
    "Reconnaissance",
    "InitialAccess",
    "LateralMovement",
    "CommandAndControl",
    "Exfiltration",
    "Impact",
]
N_STAGES = len(STAGES)
STAGE_INDEX: dict[str, int] = {s: i for i, s in enumerate(STAGES)}
BENIGN = 0

# Short labels for UI and plots.
STAGE_SHORT: dict[str, str] = {
    "Benign": "Benign",
    "Reconnaissance": "Recon",
    "InitialAccess": "Initial Access",
    "LateralMovement": "Lateral Movement",
    "CommandAndControl": "C2",
    "Exfiltration": "Exfiltration",
    "Impact": "Impact",
}


@functools.lru_cache(maxsize=1)
def stage_map_config() -> dict:
    with open(CONFIG_DIR / "stage_map.yaml", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    declared = sorted(cfg["stages"], key=lambda s: cfg["stages"][s]["index"])
    if declared != STAGES:
        raise ValueError(f"stage_map.yaml stage order {declared} != code order {STAGES}")
    return cfg


def compromise_mask() -> np.ndarray:
    """Boolean mask over stages that count as infiltration / compromise progression."""
    names = set(stage_map_config()["compromise_stages"])
    return np.array([s in names for s in STAGES], dtype=bool)


def stage_metadata(stage: str) -> dict:
    return dict(stage_map_config()["stages"][stage])

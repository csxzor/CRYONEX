"""Project configuration: one YAML file, loaded once, with path resolution.

Every tunable number lives in ``configs/default.yaml`` so that a reviewer can see the whole
experimental setup in one place, and so that an experiment config can override a subset of
it without code changes.
"""

from __future__ import annotations

import copy
import functools
import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "default.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


@functools.lru_cache(maxsize=8)
def _read_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load(experiment: str | Path | None = None) -> dict[str, Any]:
    """The default config, optionally overlaid with an experiment YAML."""
    cfg = copy.deepcopy(_read_yaml(str(DEFAULT_CONFIG)))
    if experiment is not None:
        cfg = _deep_merge(cfg, _read_yaml(str(Path(experiment))))
    return cfg


def resolve(path: str | Path) -> Path:
    """Repo-relative paths become absolute; ``~`` is expanded."""
    p = Path(os.path.expanduser(str(path)))
    return p if p.is_absolute() else REPO_ROOT / p


def data_root(cfg: dict | None = None) -> Path:
    """Where raw datasets live. ``KCWM_DATA_ROOT`` overrides the config."""
    cfg = cfg or load()
    env = os.environ.get("KCWM_DATA_ROOT")
    return resolve(env if env else cfg["paths"]["data_root"])


def dataset_dir(name: str, cfg: dict | None = None) -> Path:
    """Raw directory for one dataset, honouring its ``root: repo`` override."""
    cfg = cfg or load()
    spec = cfg["datasets"][name]
    if spec.get("root") == "repo":
        return resolve(spec["raw"])
    return data_root(cfg) / spec["raw"]


def processed_dir(cfg: dict | None = None) -> Path:
    cfg = cfg or load()
    return resolve(cfg["paths"]["processed"])


def results_dir(cfg: dict | None = None) -> Path:
    cfg = cfg or load()
    return resolve(cfg["paths"]["results"])

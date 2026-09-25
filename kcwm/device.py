"""Compute device for training: ``cpu`` (default) or ``cuda``.

Set with ``kcwm evaluate --device cuda`` or the ``KCWM_DEVICE`` environment variable. Training
and batch forecasting move to the device; inference bundles are always loaded on CPU, since
the product runs offline on a laptop.
"""

from __future__ import annotations

import os

import torch

_DEVICE: torch.device | None = None


def get() -> torch.device:
    global _DEVICE
    if _DEVICE is None:
        name = os.environ.get("KCWM_DEVICE", "cpu")
        if name.startswith("cuda") and not torch.cuda.is_available():
            print("KCWM_DEVICE=cuda requested but no GPU is visible; using CPU")
            name = "cpu"
        _DEVICE = torch.device(name)
    return _DEVICE


def set(name: str) -> torch.device:  # noqa: A001 - mirrors get()
    global _DEVICE
    os.environ["KCWM_DEVICE"] = name
    _DEVICE = None
    return get()

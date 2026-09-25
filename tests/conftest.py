"""Shared fixtures: small synthetic canonical flow tables with known structure."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from kcwm.ingest.schema import conform


def make_flows(rows: list[dict]) -> pl.DataFrame:
    """Canonical flows from minimal dicts; unspecified flow-level fields default sensibly."""
    base = {
        "te": None, "src_port": 40000, "protocol": 6, "duration": 0.5,
        "fwd_bytes": 500.0, "bwd_bytes": 1500.0, "fwd_packets": 5.0, "bwd_packets": 5.0,
        "syn_count": 1.0, "ack_count": 1.0, "fin_count": 1.0, "rst_count": 0.0,
        "psh_count": 1.0, "urg_count": 0.0, "iat_mean": 0.05, "iat_std": 0.01,
        "iat_max": 0.1, "iat_min": 0.01, "pkt_len_mean": 200.0, "pkt_len_std": 50.0,
        "pkt_len_max": 400.0, "pkt_len_min": 60.0,
        "label": "BENIGN", "family": "Benign", "stage": "Benign", "technique": None,
        "is_attack": False, "attempted": False,
    }
    full = []
    for r in rows:
        d = {**base, **r}
        if d["te"] is None:
            d["te"] = d["ts"] + d["duration"]
        full.append(d)
    return conform(pl.DataFrame(full), source="synthetic")


def background(t0: float, t1: float, rng: np.random.Generator, rate: float = 2.0) -> list[dict]:
    """Benign internal->external web traffic at roughly ``rate`` flows/s."""
    n = int((t1 - t0) * rate)
    ts = np.sort(rng.uniform(t0, t1, n))
    hosts = [f"192.168.1.{i}" for i in range(10, 20)]
    servers = [f"8.8.{i}.{j}" for i in range(4) for j in range(1, 4)]
    return [{"ts": float(t), "src_ip": hosts[rng.integers(len(hosts))],
             "dst_ip": servers[rng.integers(len(servers))], "dst_port": 443} for t in ts]


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def cfg():
    from kcwm import config

    c = config.load()
    c["window"]["warmup_windows"] = 12
    c["window"]["context"] = 8
    c["window"]["horizons"] = [3, 6]
    c["window"]["episode_merge_gap"] = 3
    c["window"]["onset_quiet"] = 3
    return c

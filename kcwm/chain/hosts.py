"""Per-host activity table: one row per (internal host, 10-second window) with any traffic.

Every flow counts for each internal endpoint: as *outgoing* activity of an internal source and
as *incoming* activity of an internal destination. The features are structural (distinct
peers and ports, failed-connection shares, auth-service hits, first-contact counts, bytes
sent out of the network), because counts of raw volume do not transfer between networks.

Labels: ``stage_now`` is the highest ATT&CK stage index among the attack flows the host took
part in during the window, in either role. They are used for training and scoring only.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ..features.extra import AUTH_PORTS, MAIL_PORTS, is_internal
from ..stages import STAGE_INDEX

HOST_FEATURES: list[str] = [
    # outgoing (host is the source)
    "o_n", "o_hosts", "o_int_hosts", "o_ext_hosts", "o_ports", "o_fail", "o_fail_frac",
    "o_auth_int", "o_icmp", "o_dns", "o_mail", "o_new_int", "o_new_ext", "o_long_ext",
    # incoming (host is the destination)
    "i_n", "i_hosts", "i_ext_hosts", "i_int_hosts", "i_ports", "i_fail", "i_fail_frac",
    "i_auth", "i_auth_ext", "i_icmp", "i_new_ext", "i_new_int",
    # volume leaving the network from this host (log bytes)
    "sent_ext", "recv_ext",
]

_COLS = ["ts", "window", "src_ip", "dst_ip", "dst_port", "protocol", "duration", "fwd_bytes", "bwd_bytes",
         "fwd_packets", "bwd_packets", "syn_count", "ack_count", "rst_count", "stage", "attempted"]


def _prepare(flows: pl.DataFrame, cidrs: list[str], seconds: float) -> pl.DataFrame:
    f = flows.select([c for c in _COLS if c in flows.columns])
    if "window" not in f.columns:
        f = f.with_columns((pl.col("ts") // seconds).cast(pl.Int64).alias("window"))
    pk = pl.col("fwd_packets").fill_null(0) + pl.col("bwd_packets").fill_null(0)
    f = f.with_columns(
        is_internal(pl.col("src_ip"), cidrs).alias("_si"),
        is_internal(pl.col("dst_ip"), cidrs).alias("_di"),
        ((pl.col("rst_count").fill_null(0) > 0)
         | ((pl.col("syn_count").fill_null(0) > 0) & (pl.col("ack_count").fill_null(0) == 0))
         | (pk <= 2)).alias("_fail"),
        pl.col("dst_port").is_in(AUTH_PORTS).alias("_auth"),
        (pl.col("protocol") == 1).alias("_icmp"),
        (pl.col("dst_port") == 53).alias("_dns"),
        pl.col("dst_port").is_in(MAIL_PORTS).alias("_mail"),
        pl.col("stage").fill_null("Benign").replace_strict(STAGE_INDEX, default=0).cast(pl.Int8).alias("_stage"),
        pl.col("attempted").fill_null(False).alias("_att"),
    )
    # first contact of a (src, dst) pair within this capture: causal, uses only earlier flows
    first = f.group_by(["src_ip", "dst_ip"]).agg(pl.col("ts").min().alias("_first"))
    return f.join(first, on=["src_ip", "dst_ip"], how="left").with_columns(
        (pl.col("ts") <= pl.col("_first")).alias("_new")
    ).drop("_first")


def host_windows(flows: pl.DataFrame, cidrs: list[str], *, seconds: float = 10.0) -> pl.DataFrame:
    """``(host, window)`` rows with ``HOST_FEATURES``, ``stage_now`` and ``attempted``."""
    f = _prepare(flows, cidrs, seconds)
    out = f.filter(pl.col("_si")).group_by(["src_ip", "window"]).agg(
        pl.len().alias("o_n"),
        pl.col("dst_ip").n_unique().alias("o_hosts"),
        pl.col("dst_ip").filter(pl.col("_di")).n_unique().alias("o_int_hosts"),
        pl.col("dst_ip").filter(~pl.col("_di")).n_unique().alias("o_ext_hosts"),
        pl.col("dst_port").n_unique().alias("o_ports"),
        pl.col("_fail").sum().alias("o_fail"),
        (pl.col("_auth") & pl.col("_di")).sum().alias("o_auth_int"),
        pl.col("_icmp").sum().alias("o_icmp"),
        pl.col("_dns").sum().alias("o_dns"),
        pl.col("_mail").sum().alias("o_mail"),
        pl.col("dst_ip").filter(pl.col("_new") & pl.col("_di")).n_unique().alias("o_new_int"),
        pl.col("dst_ip").filter(pl.col("_new") & ~pl.col("_di")).n_unique().alias("o_new_ext"),
        ((pl.col("duration") > 60) & ~pl.col("_di")).sum().alias("o_long_ext"),
        pl.col("fwd_bytes").filter(~pl.col("_di")).sum().alias("_sent_o"),
        pl.col("bwd_bytes").filter(~pl.col("_di")).sum().alias("_recv_o"),
        pl.col("_stage").max().alias("_st_o"),
        pl.col("_att").any().alias("_att_o"),
    ).rename({"src_ip": "host"})
    inc = f.filter(pl.col("_di")).group_by(["dst_ip", "window"]).agg(
        pl.len().alias("i_n"),
        pl.col("src_ip").n_unique().alias("i_hosts"),
        pl.col("src_ip").filter(~pl.col("_si")).n_unique().alias("i_ext_hosts"),
        pl.col("src_ip").filter(pl.col("_si")).n_unique().alias("i_int_hosts"),
        pl.col("dst_port").n_unique().alias("i_ports"),
        pl.col("_fail").sum().alias("i_fail"),
        pl.col("_auth").sum().alias("i_auth"),
        (pl.col("_auth") & ~pl.col("_si")).sum().alias("i_auth_ext"),
        pl.col("_icmp").sum().alias("i_icmp"),
        pl.col("src_ip").filter(pl.col("_new") & ~pl.col("_si")).n_unique().alias("i_new_ext"),
        pl.col("src_ip").filter(pl.col("_new") & pl.col("_si")).n_unique().alias("i_new_int"),
        pl.col("bwd_bytes").filter(~pl.col("_si")).sum().alias("_sent_i"),
        pl.col("fwd_bytes").filter(~pl.col("_si")).sum().alias("_recv_i"),
        pl.col("_stage").max().alias("_st_i"),
        pl.col("_att").any().alias("_att_i"),
    ).rename({"dst_ip": "host"})
    hw = out.join(inc, on=["host", "window"], how="full", coalesce=True)
    counts = [c for c in HOST_FEATURES if c not in ("o_fail_frac", "i_fail_frac", "sent_ext", "recv_ext")]
    hw = hw.with_columns(
        [pl.col(c).fill_null(0).cast(pl.Float64) for c in counts],
    ).with_columns(
        (pl.col("o_fail") / pl.col("o_n").clip(lower_bound=1)).alias("o_fail_frac"),
        (pl.col("i_fail") / pl.col("i_n").clip(lower_bound=1)).alias("i_fail_frac"),
        (pl.col("_sent_o").fill_null(0) + pl.col("_sent_i").fill_null(0)).log1p().alias("sent_ext"),
        (pl.col("_recv_o").fill_null(0) + pl.col("_recv_i").fill_null(0)).log1p().alias("recv_ext"),
        pl.max_horizontal(pl.col("_st_o").fill_null(0), pl.col("_st_i").fill_null(0)).cast(pl.Int8).alias("stage_now"),
        (pl.col("_att_o").fill_null(False) | pl.col("_att_i").fill_null(False)).alias("attempted"),
        (pl.col("window").cast(pl.Float64) * seconds).alias("t"),
    )
    return hw.select(["host", "window", "t", *HOST_FEATURES, "stage_now", "attempted"]).sort(["host", "window"])


def merge_host_tables(tables: list[pl.DataFrame]) -> pl.DataFrame:
    """Combine tables of one network (e.g. DAPT's public and private interfaces on one day):
    counts add, stage takes the maximum. Distinct counts may double-count a peer seen on two
    sensors; that is accepted (the same happens to a live deployment with two taps)."""
    if len(tables) == 1:
        return tables[0]
    allt = pl.concat(tables, how="vertical_relaxed")
    counts = [c for c in HOST_FEATURES if c not in ("o_fail_frac", "i_fail_frac", "sent_ext", "recv_ext")]
    m = allt.group_by(["host", "window"]).agg(
        pl.col("t").first(),
        *[pl.col(c).sum() for c in counts],
        pl.col("sent_ext").exp().sub(1).sum().log1p(),
        pl.col("recv_ext").exp().sub(1).sum().log1p(),
        pl.col("stage_now").max(),
        pl.col("attempted").any(),
    )
    return m.with_columns(
        (pl.col("o_fail") / pl.col("o_n").clip(lower_bound=1)).alias("o_fail_frac"),
        (pl.col("i_fail") / pl.col("i_n").clip(lower_bound=1)).alias("i_fail_frac"),
    ).select(allt.columns).sort(["host", "window"])


# Scale-free shares and sustained activity: a brute force of 10 logins per 10 s (CIC-IDS2017)
# and one of 320 (CIC-IDS2018) have the same *share* of failed auth-service flows and both
# persist for minutes. Raw counts alone let a tree learn one network's volume.
SHARES = {"i_auth_frac": ("i_auth", "i_n"), "i_auth_ext_frac": ("i_auth_ext", "i_n"),
          "i_ext_frac": ("i_ext_hosts", "i_hosts"), "i_port_frac": ("i_ports", "i_n"),
          "o_int_frac": ("o_int_hosts", "o_hosts"), "o_port_frac": ("o_ports", "o_n"),
          "o_new_int_frac": ("o_new_int", "o_hosts")}
ROLLING = ["i_auth_ext", "i_fail", "i_ports", "i_new_ext", "o_int_hosts", "o_new_int", "o_ports", "o_fail",
           "o_auth_int", "o_long_ext"]
ROLLING_SPANS = {"1m": 6, "5m": 30}


def nowcast_feature_names() -> list[str]:
    return ([*HOST_FEATURES, *SHARES] + [f"{c}_{t}" for t in ROLLING_SPANS for c in ROLLING]
            + [f"active_{t}" for t in ROLLING_SPANS])


def feature_matrix(hw: pl.DataFrame) -> np.ndarray:
    """Model input for the per-window stage nowcast: log counts, shares, and each host's
    activity over the last 1 and 5 minutes (only its own rows at or before the window)."""
    keys = ["timeline", "host"] if "timeline" in hw.columns else ["host"]
    order = hw.with_row_index("_r").sort([*keys, "window"])
    exprs = [(pl.col(a) / pl.col(b).clip(lower_bound=1)).alias(n) for n, (a, b) in SHARES.items()]
    for tag, n in ROLLING_SPANS.items():
        exprs += [pl.col(c).rolling_sum_by("window", window_size=f"{n}i").over(keys).alias(f"{c}_{tag}")
                  for c in ROLLING]
        exprs.append(pl.col("window").is_not_null().cast(pl.Float64).rolling_sum_by("window", window_size=f"{n}i")
                     .over(keys).alias(f"active_{tag}"))
    order = order.with_columns(exprs).sort("_r")
    cols = []
    for c in nowcast_feature_names():
        x = order[c].to_numpy().astype(np.float64)
        cols.append(x if c.endswith("_frac") or c in ("sent_ext", "recv_ext") else np.log1p(x))
    return np.stack(cols, axis=1).astype(np.float32)

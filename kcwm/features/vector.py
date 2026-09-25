"""Per-window base state features (v1's 75-dimensional vector, ported).

Ported from netforecast v1 ``state/vector.py`` with unchanged definitions. The only change
is that v1's ``packet_level_valid`` feature became the ``m_packet`` input mask (see
``registry.MASK_NAMES``), so validity is signalled explicitly rather than as one more input
the model might misread.

Entropy is normalised by ``log2(n distinct)`` so that it measures dispersion, not volume.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from ..ingest.schema import PAYLOAD_HIST_COLUMNS

EPS = 1e-9
FLAGS = ["syn", "ack", "fin", "rst", "psh", "urg"]
WELL_KNOWN = [80, 443, 22, 21, 25, 53, 445, 3389]


def _normalised_entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0 or counts.size <= 1:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum()) / math.log2(counts.size)


def _entropy_by_window(df: pl.DataFrame, column: str, name: str) -> pl.DataFrame:
    grouped = (
        df.group_by(["window", column]).agg(pl.len().alias("n"))
        .group_by("window").agg(pl.col("n"))
    )
    if grouped.height == 0:
        return pl.DataFrame(schema={"window": pl.Int64, name: pl.Float64})
    values = [_normalised_entropy(np.asarray(r, dtype=np.float64)) for r in grouped["n"].to_list()]
    return pl.DataFrame({"window": grouped["window"], name: pl.Series(values, dtype=pl.Float64)})


def _cardinality(df: pl.DataFrame, group: str, count: str, prefix: str) -> pl.DataFrame:
    per_group = df.group_by(["window", group]).agg(pl.col(count).n_unique().alias("n"))
    return per_group.group_by("window").agg(
        pl.col("n").max().cast(pl.Float64).alias(f"{prefix}_max"),
        pl.col("n").mean().cast(pl.Float64).alias(f"{prefix}_mean"),
    )


def base_features(flows: pl.DataFrame, window_seconds: float) -> pl.DataFrame:
    """v1 state features plus the ``m_packet`` mask, one row per occupied window."""
    w = float(window_seconds)
    prepared = flows.with_columns(
        (pl.col("fwd_packets") + pl.col("bwd_packets")).alias("_pkts"),
        (pl.col("fwd_bytes") + pl.col("bwd_bytes")).alias("_bytes"),
        (pl.col("bwd_bytes").fill_null(0.0) <= 0).alias("_unidirectional"),
        (pl.col("bwd_packets").fill_null(0.0) <= 0).alias("_empty_response"),
        ((pl.col("syn_count") > 0) & (pl.col("ack_count") <= 0)).alias("_unanswered_syn"),
        (pl.col("rst_count") > 0).alias("_has_rst"),
        ((pl.col("dst_port") >= 0) & (pl.col("dst_port") < 1024)).alias("_low_port"),
        (pl.col("dst_port") >= 49152).alias("_ephemeral_port"),
        pl.col("dst_port").is_in(WELL_KNOWN).alias("_well_known"),
    )
    # The packet block counts as observed only when PCAP-derived headers are present (TTL is
    # populated exactly then). The corrected CIC CSVs carry a TCP initial window but no TTL,
    # fragments or retransmissions. Marking those windows "observed" would let the model read
    # zeros for the missing fields as measurements, and differently per dataset.
    valid_packet = (pl.col("ttl_mean").is_not_null() & pl.col("ttl_mean").is_not_nan()).any()
    agg = prepared.group_by("window").agg(
        pl.len().cast(pl.Float64).alias("flow_count"),
        pl.col("fwd_bytes").sum().alias("fwd_bytes_total"),
        pl.col("bwd_bytes").sum().alias("bwd_bytes_total"),
        pl.col("fwd_packets").sum().alias("fwd_packets_total"),
        pl.col("bwd_packets").sum().alias("bwd_packets_total"),
        pl.col("_bytes").mean().alias("bytes_per_flow_mean"),
        pl.col("_bytes").std().alias("bytes_per_flow_std"),
        pl.col("_bytes").max().alias("bytes_per_flow_max"),
        pl.col("_pkts").mean().alias("packets_per_flow_mean"),
        pl.col("_pkts").std().alias("packets_per_flow_std"),
        pl.col("_pkts").max().alias("packets_per_flow_max"),
        pl.col("duration").mean().alias("duration_mean"),
        pl.col("duration").std().alias("duration_std"),
        pl.col("duration").max().alias("duration_max"),
        *[pl.col(f"{f}_count").sum().alias(f"_{f}_sum") for f in FLAGS],
        pl.col("_unanswered_syn").mean().alias("unanswered_syn_frac"),
        pl.col("_has_rst").mean().alias("rst_frac"),
        (pl.col("protocol") == 6).mean().alias("tcp_frac"),
        (pl.col("protocol") == 17).mean().alias("udp_frac"),
        (pl.col("protocol") == 1).mean().alias("icmp_frac"),
        (~pl.col("protocol").is_in([1, 6, 17])).mean().alias("other_proto_frac"),
        pl.col("iat_mean").mean().alias("iat_mean_mean"),
        pl.col("iat_mean").std().alias("iat_mean_std"),
        pl.col("iat_mean").min().alias("iat_mean_min"),
        pl.col("iat_max").max().alias("iat_max_max"),
        pl.col("iat_std").mean().alias("iat_std_mean"),
        pl.col("src_ip").n_unique().cast(pl.Float64).alias("unique_src_hosts"),
        pl.col("dst_ip").n_unique().cast(pl.Float64).alias("unique_dst_hosts"),
        pl.col("dst_port").n_unique().cast(pl.Float64).alias("unique_dst_ports"),
        pl.col("src_port").n_unique().cast(pl.Float64).alias("unique_src_ports"),
        pl.col("_low_port").mean().alias("low_port_frac"),
        pl.col("_ephemeral_port").mean().alias("ephemeral_port_frac"),
        pl.col("_well_known").mean().alias("well_known_service_frac"),
        (pl.col("fwd_bytes") / (pl.col("_bytes") + EPS)).mean().alias("bidirectional_ratio_mean"),
        pl.col("_unidirectional").mean().alias("unidirectional_flow_frac"),
        pl.col("_empty_response").mean().alias("empty_response_frac"),
        pl.col("ttl_mean").mean().alias("ttl_mean_mean"),
        pl.col("ttl_var").mean().alias("ttl_var_mean"),
        pl.col("ttl_unique").mean().alias("ttl_unique_mean"),
        (pl.col("ttl_mean").max() - pl.col("ttl_mean").min()).alias("ttl_spread"),
        pl.col("win_mean").mean().alias("win_mean_mean"),
        pl.col("win_std").mean().alias("win_std_mean"),
        pl.col("win_zero_count").sum().alias("_win_zero_sum"),
        pl.col("frag_rate").mean().alias("frag_rate_mean"),
        pl.col("retrans_count").sum().alias("_retrans_sum"),
        *[pl.col(c).mean().alias(f"{c}_mean") for c in PAYLOAD_HIST_COLUMNS],
        valid_packet.cast(pl.Float64).alias("m_packet"),
    )
    pkts = pl.col("fwd_packets_total") + pl.col("bwd_packets_total")
    agg = agg.with_columns(
        (pl.col("flow_count") / w).alias("flow_rate"),
        ((pl.col("fwd_bytes_total") + pl.col("bwd_bytes_total")) / w).alias("bytes_rate"),
        (pkts / w).alias("packets_rate"),
        *[(pl.col(f"_{f}_sum") / (pl.col("flow_count") + EPS)).alias(f"{f}_rate") for f in FLAGS],
        (pl.col("_syn_sum") / (pl.col("_ack_sum") + 1.0)).alias("syn_ack_ratio"),
        (pl.col("_win_zero_sum") / (pkts + EPS)).alias("win_zero_rate"),
        (pl.col("_retrans_sum") / (pkts + EPS)).alias("retrans_rate"),
        (1.0 / (1.0 + pl.col("iat_mean_std") / (pl.col("iat_mean_mean") + EPS))).alias("iat_regularity"),
    )
    sums = [pl.col(f"_{f}_sum") for f in FLAGS]
    total = sum(sums) + EPS
    agg = agg.with_columns(
        (-sum((s / total) * ((s / total) + EPS).log(base=2) for s in sums) / math.log2(len(FLAGS)))
        .alias("flag_entropy")
    )
    for frame in (
        _entropy_by_window(prepared, "dst_port", "dst_port_entropy"),
        _entropy_by_window(prepared, "src_port", "src_port_entropy"),
        _entropy_by_window(prepared, "dst_ip", "dst_ip_entropy"),
        _entropy_by_window(prepared, "src_ip", "src_ip_entropy"),
        _cardinality(prepared, "src_ip", "dst_ip", "fanout"),
        _cardinality(prepared, "src_ip", "dst_port", "ports_per_src"),
        prepared.group_by(["window", "src_ip"]).agg(pl.len().alias("n"))
        .group_by("window").agg(pl.col("n").max().cast(pl.Float64).alias("flows_per_host_max")),
    ):
        agg = agg.join(frame, on="window", how="left")

    # Share of destination ports not seen in the previous occupied window.
    ports = prepared.group_by("window").agg(pl.col("dst_port").unique().alias("_p")).sort("window")
    prev, cur = ports["_p"].shift(1).to_list(), ports["_p"].to_list()
    novelty = []
    for p, c in zip(prev, cur):
        cs = set(c or [])
        novelty.append(len(cs - set(p or [])) / len(cs) if cs else 0.0)
    agg = agg.join(
        pl.DataFrame({"window": ports["window"], "new_dst_port_frac": pl.Series(novelty, dtype=pl.Float64)}),
        on="window", how="left",
    )
    return agg

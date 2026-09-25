"""v2 state features: kill-chain signatures that v1's aggregate vector could not see.

v1's cross-dataset audit found that volume features invert between networks, while
structural ones (flag ratios, port fan-out, timing regularity) keep their direction. These
features are structural and per-source: a single brute-forcer, beaconing bot or internal
scanner stays visible even when its flows are a tiny share of a busy window.

* **access**: login-service hammering, web request bursts, half-open SYNs, sequential port
  access, ICMP sweeps (Reconnaissance / Initial Access).
* **lateral**: internal-to-internal fan-out and scanning (Lateral Movement). Needs
  internal/external addressing, so it is gated by the ``m_ip`` mask.
* **c2**: DNS query bursts and mail-port traffic (bot tasking, spam).
* **exfil**: an internal host's outbound volume and upload/download asymmetry.
* **lookback**: causal history features. New host pairs are pairs unseen in the previous 64
  windows; beaconing is periodic connection starts on one (src, dst, port) channel. Both use
  only flows at or before the window they describe (see ``tests/test_features.py``).
"""

from __future__ import annotations

import ipaddress

import polars as pl

AUTH_PORTS = [21, 22, 23, 445, 3306, 3389, 5900]
WEB_PORTS = [80, 443, 8080, 8443]
MAIL_PORTS = [25, 465, 587]
NEW_PAIR_LOOKBACK = 64
BEACON_WINDOW = 6       # connections per rolling estimate
BEACON_MIN = 4          # minimum gaps before a channel can be scored
BEACON_GAP = (2.0, 900.0)  # mean gap between connection starts, seconds
SEQ_MIN_FLOWS = 8


def ipv4_to_int(expr: pl.Expr) -> pl.Expr:
    parts = expr.cast(pl.Utf8).str.split(".")
    octets = [parts.list.get(i, null_on_oob=True).cast(pl.Int64, strict=False) for i in range(4)]
    value = octets[0] * 16_777_216 + octets[1] * 65_536 + octets[2] * 256 + octets[3]
    return pl.when(parts.list.len() == 4).then(value).otherwise(None)


def is_internal(expr: pl.Expr, cidrs: list[str]) -> pl.Expr:
    """True for IPv4 addresses inside any of ``cidrs``. IPv6 and malformed values are external."""
    if not cidrs:
        return pl.lit(False)
    ip = ipv4_to_int(expr)
    conds = []
    for cidr in cidrs:
        net = ipaddress.ip_network(cidr, strict=False)
        conds.append(ip.is_between(int(net.network_address), int(net.broadcast_address)))
    return pl.any_horizontal(conds).fill_null(False)


def _max_count(df: pl.DataFrame, keys: list[str], name: str) -> pl.DataFrame:
    return (
        df.group_by(["window", *keys]).agg(pl.len().alias("_n"))
        .group_by("window").agg(pl.col("_n").max().cast(pl.Float64).alias(name))
    )


def _max_distinct(df: pl.DataFrame, keys: list[str], col: str, name: str) -> pl.DataFrame:
    return (
        df.group_by(["window", *keys]).agg(pl.col(col).n_unique().alias("_n"))
        .group_by("window").agg(pl.col("_n").max().cast(pl.Float64).alias(name))
    )


def access_features(f: pl.DataFrame) -> list[pl.DataFrame]:
    auth = f.filter(pl.col("dst_port").is_in(AUTH_PORTS))
    web = f.filter(pl.col("dst_port").is_in(WEB_PORTS))
    half_open = f.filter((pl.col("syn_count") > 0) & (pl.col("ack_count") <= 0))
    icmp = f.filter(pl.col("protocol") == 1)
    seq = (
        f.filter(pl.col("dst_port") >= 0)
        .sort("ts")
        .group_by(["window", "src_ip"])
        .agg(pl.col("dst_port").alias("_ports"), pl.len().alias("_n"))
        .filter(pl.col("_n") >= SEQ_MIN_FLOWS)
        .with_columns(
            pl.col("_ports").list.diff(null_behavior="drop")
            .list.eval(pl.element().abs() == 1).list.mean().alias("_seq")
        )
        .group_by("window").agg(pl.col("_seq").max().cast(pl.Float64).alias("seq_port_score"))
    )
    return [
        _max_count(auth, ["src_ip", "dst_ip"], "auth_conn_max"),
        auth.group_by("window").agg((pl.col("duration") < 3.0).mean().cast(pl.Float64).alias("auth_short_frac")),
        _max_count(web, ["src_ip", "dst_ip"], "web_conn_max"),
        _max_count(half_open, ["src_ip"], "half_open_max"),
        seq,
        _max_distinct(icmp, ["src_ip"], "dst_ip", "icmp_sweep_max"),
    ]


def lateral_features(f: pl.DataFrame) -> list[pl.DataFrame]:
    ii = f.filter(pl.col("_src_int") & pl.col("_dst_int"))
    return [
        f.group_by("window").agg((pl.col("_src_int") & pl.col("_dst_int")).mean().cast(pl.Float64).alias("int_int_frac")),
        _max_distinct(ii, ["src_ip"], "dst_ip", "int_fanout_max"),
        _max_distinct(ii, ["src_ip", "dst_port"], "dst_ip", "int_scan_max"),
    ]


def c2_features(f: pl.DataFrame) -> list[pl.DataFrame]:
    dns = f.filter(pl.col("dst_port") == 53)
    return [
        _max_count(dns, ["src_ip"], "dns_max"),
        f.group_by("window").agg(pl.col("dst_port").is_in(MAIL_PORTS).mean().cast(pl.Float64).alias("smtp_frac")),
    ]


def exfil_features(f: pl.DataFrame) -> list[pl.DataFrame]:
    # Per internal host, bytes it sent to and received from external peers, whichever side
    # opened the connection.
    outbound = f.filter(pl.col("_src_int") & ~pl.col("_dst_int")).select(
        "window", pl.col("src_ip").alias("host"),
        pl.col("fwd_bytes").alias("_out"), pl.col("bwd_bytes").alias("_in"), "duration",
    )
    inbound = f.filter(~pl.col("_src_int") & pl.col("_dst_int")).select(
        "window", pl.col("dst_ip").alias("host"),
        pl.col("bwd_bytes").alias("_out"), pl.col("fwd_bytes").alias("_in"), "duration",
    )
    both = pl.concat([outbound, inbound])
    per_host = both.group_by(["window", "host"]).agg(pl.col("_out").sum(), pl.col("_in").sum())
    return [
        per_host.group_by("window").agg(pl.col("_out").max().cast(pl.Float64).alias("out_bytes_max")),
        per_host.filter((pl.col("_out") + pl.col("_in")) >= 10_000)
        .group_by("window")
        .agg(((pl.col("_out") - pl.col("_in")) / (pl.col("_out") + pl.col("_in") + 1.0)).max().cast(pl.Float64).alias("out_asym_max")),
        outbound.filter(pl.col("duration") >= 60.0).group_by("window")
        .agg(pl.col("_out").sum().cast(pl.Float64).alias("long_out_bytes")),
    ]


def lookback_features(f: pl.DataFrame) -> list[pl.DataFrame]:
    """Causal features: each window's value depends only on flows starting at or before it."""
    pairs = (
        f.select("window", "src_ip", "dst_ip", "_src_int", "_dst_int")
        .unique(["window", "src_ip", "dst_ip"])
        .sort(["src_ip", "dst_ip", "window"])
        .with_columns(pl.col("window").shift(1).over(["src_ip", "dst_ip"]).alias("_prev"))
        .with_columns(
            (pl.col("_prev").is_null() | ((pl.col("window") - pl.col("_prev")) > NEW_PAIR_LOOKBACK)).alias("_new")
        )
    )
    new_pairs = pairs.group_by("window").agg(
        pl.col("_new").mean().cast(pl.Float64).alias("new_pair_frac"),
        (pl.col("_new") & pl.col("_src_int") & pl.col("_dst_int")).sum().cast(pl.Float64).alias("new_int_pairs"),
    )

    key = ["src_ip", "dst_ip", "dst_port"]
    beacons = (
        f.select("ts", "te", "window", *key)
        .sort("ts")
        .with_columns(
            (pl.col("ts") - pl.col("ts").shift(1).over(key)).alias("_gap"),
            (pl.col("ts") - pl.col("te").shift(1).over(key)).alias("_idle"),
        )
        .with_columns(
            pl.col("_gap").rolling_mean(BEACON_WINDOW, min_samples=BEACON_MIN).over(key).alias("_m"),
            pl.col("_gap").rolling_std(BEACON_WINDOW, min_samples=BEACON_MIN).over(key).alias("_s"),
            pl.col("_idle").rolling_mean(BEACON_WINDOW, min_samples=BEACON_MIN).over(key).alias("_mi"),
        )
        .with_columns(
            pl.when(pl.col("_m").is_between(*BEACON_GAP) & (pl.col("_mi") >= 1.0))
            .then(1.0 / (1.0 + pl.col("_s") / pl.col("_m")))
            .otherwise(0.0)
            .fill_null(0.0)
            .fill_nan(0.0)
            .alias("_score")
        )
    )
    beacon = beacons.group_by("window").agg(
        pl.col("_score").max().cast(pl.Float64).alias("beacon_max"),
        pl.struct(key).filter(pl.col("_score") >= 0.8).n_unique().cast(pl.Float64).alias("beacon_pairs"),
    )
    return [new_pairs, beacon]


def extra_features(flows: pl.DataFrame, internal_cidrs: list[str]) -> pl.DataFrame:
    """All v2 features for the occupied windows of one capture."""
    f = flows.with_columns(
        is_internal(pl.col("src_ip"), internal_cidrs).alias("_src_int"),
        is_internal(pl.col("dst_ip"), internal_cidrs).alias("_dst_int"),
    )
    out = f.select("window").unique()
    for frame in (*access_features(f), *lateral_features(f), *c2_features(f),
                  *exfil_features(f), *lookback_features(f)):
        out = out.join(frame, on="window", how="left")
    return out

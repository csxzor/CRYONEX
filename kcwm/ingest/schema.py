"""Canonical flow schema.

Every ingest path funnels into this one schema: CIC-IDS CSV, CTU-13 Argus binetflow,
DAPT2020, or a PCAP parsed by NFStream. Windowing, features and the host-attribution replay
never need to know where a flow came from. Ported from netforecast v1 ``ingest/schema.py``.
v2 renames ``attack_family``/``tactic`` to ``family``/``stage`` and adds ``technique``.

Two feature levels live side by side, as the problem statement requires:

* Flow-level: always populated.
* Packet-level (``PACKET_LEVEL_COLUMNS``): populated only by the PCAP path, plus the TCP
  initial window size that CIC's corrected CSVs carry. CSV-sourced flows keep **null** in
  the rest of the block. The absence must stay visible so that no model reads an imputed
  zero as a measurement; the window builder turns it into an explicit mask.
"""

from __future__ import annotations

import polars as pl

KEY_COLUMNS: list[str] = ["ts", "te", "src_ip", "dst_ip", "src_port", "dst_port", "protocol"]

FLOW_LEVEL_COLUMNS: list[str] = [
    "duration",
    "fwd_bytes", "bwd_bytes", "fwd_packets", "bwd_packets",
    "syn_count", "ack_count", "fin_count", "rst_count", "psh_count", "urg_count",
    "iat_mean", "iat_std", "iat_max", "iat_min",
    "pkt_len_mean", "pkt_len_std", "pkt_len_max", "pkt_len_min",
]

PAYLOAD_HIST_BINS = 8
PAYLOAD_HIST_COLUMNS: list[str] = [f"payload_hist_{i}" for i in range(PAYLOAD_HIST_BINS)]

PACKET_LEVEL_COLUMNS: list[str] = [
    "ttl_mean", "ttl_var", "ttl_unique",
    "win_mean", "win_std", "win_zero_count",
    "frag_rate", "retrans_count",
    *PAYLOAD_HIST_COLUMNS,
]

LABEL_COLUMNS: list[str] = ["label", "family", "stage", "technique", "is_attack", "attempted"]

ALL_COLUMNS: list[str] = [*KEY_COLUMNS, *FLOW_LEVEL_COLUMNS, *PACKET_LEVEL_COLUMNS, *LABEL_COLUMNS]

BENIGN_LABEL = "BENIGN"

SCHEMA: dict[str, pl.DataType] = {
    "ts": pl.Float64,
    "te": pl.Float64,
    "src_ip": pl.Utf8,
    "dst_ip": pl.Utf8,
    "src_port": pl.Int32,
    "dst_port": pl.Int32,
    "protocol": pl.Int16,
    **{c: pl.Float64 for c in FLOW_LEVEL_COLUMNS},
    **{c: pl.Float64 for c in PACKET_LEVEL_COLUMNS},
    "label": pl.Utf8,
    "family": pl.Utf8,
    "stage": pl.Utf8,
    "technique": pl.Utf8,
    "is_attack": pl.Boolean,
    "attempted": pl.Boolean,
}


class SchemaError(ValueError):
    """Raised when a frame cannot be conformed to the canonical flow schema."""


def empty_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=SCHEMA)


def conform(df: pl.DataFrame, *, source: str) -> pl.DataFrame:
    """Coerce ``df`` to the canonical schema.

    Missing packet-level columns become null (not zero). Missing key or flow-level columns
    are an error: silently zero-filling an aggregate would corrupt every state vector.
    """
    missing = [c for c in (*KEY_COLUMNS, *FLOW_LEVEL_COLUMNS) if c not in df.columns]
    if missing:
        raise SchemaError(f"{source}: missing required columns {sorted(missing)}")

    defaults = {
        "label": pl.lit(BENIGN_LABEL, dtype=pl.Utf8),
        "family": pl.lit("Benign", dtype=pl.Utf8),
        "stage": pl.lit("Benign", dtype=pl.Utf8),
        "technique": pl.lit(None, dtype=pl.Utf8),
        "is_attack": pl.lit(False, dtype=pl.Boolean),
        "attempted": pl.lit(False, dtype=pl.Boolean),
    }
    exprs: list[pl.Expr] = []
    for name in ALL_COLUMNS:
        dtype = SCHEMA[name]
        if name in df.columns:
            exprs.append(pl.col(name).cast(dtype, strict=False).alias(name))
        elif name in PACKET_LEVEL_COLUMNS:
            exprs.append(pl.lit(None, dtype=dtype).alias(name))
        else:
            exprs.append(defaults[name].alias(name))
    return df.select(exprs)


def has_packet_level(df: pl.DataFrame) -> bool:
    """True when any representative packet-level column carries a real measurement.

    NaN counts as missing alongside null (a NumPy round trip turns null into NaN).
    """
    if df.height == 0:
        return False
    for name in ("ttl_mean", "win_mean", "frag_rate", "retrans_count"):
        if name in df.columns:
            col = df[name]
            if bool((~(col.is_null() | col.is_nan().fill_null(True))).any()):
                return True
    return False

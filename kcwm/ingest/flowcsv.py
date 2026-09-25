"""Flow-record ingest for CIC-IDS2017/2018 (corrected), CTU-13 and DAPT2020.

Ported from netforecast v1 ``ingest/flowcsv.py``: the column maps, timestamp parsing, port
and protocol normalisation, and CTU-13 flag recovery are unchanged. v2 changes:

* **Projected reads by default.** v1's full ``read_csv`` OOM-killed on a single
  CSE-CIC-IDS2018 day on this 15 GB machine. Only the columns the map uses are parsed, via a
  lazy ``scan_csv().select()`` (v1 needed a monkeypatch script for this).
* **Labels resolve to (family, stage, technique)** through ``labels.mapper``. CTU-13
  labels resolve with their scenario number, because some activities (DDoS, scans) are only
  documented for certain captures.
* **DAPT2020** is CICFlowMeter output whose ``Stage`` column is the label.

Column-name matching uses a folded key (lowercase alphanumerics), because CIC headers vary
in leading spaces, capitalisation and singular/plural across releases.
"""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl

from . import schema
from .labels import is_attempted, mapper

_CIC_MICROSECOND_COLUMNS = {"duration", "iat_mean", "iat_std", "iat_max", "iat_min"}
_NULLS = ["", "NaN", "nan", "Infinity", "-Infinity", "inf", "-"]


def fold(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


CIC_COLUMN_MAP: dict[str, str] = {
    "srcip": "src_ip", "sourceip": "src_ip", "flowid": "_flow_id",
    "dstip": "dst_ip", "destinationip": "dst_ip",
    "srcport": "src_port", "sourceport": "src_port",
    "dstport": "dst_port", "destinationport": "dst_port",
    "protocol": "protocol",
    "timestamp": "_timestamp",
    "flowduration": "duration",
    "totalfwdpackets": "fwd_packets", "totalfwdpacket": "fwd_packets", "totfwdpkts": "fwd_packets",
    "totalbackwardpackets": "bwd_packets", "totalbwdpackets": "bwd_packets",
    "totbwdpkts": "bwd_packets",
    "totallengthoffwdpackets": "fwd_bytes", "totallengthoffwdpacket": "fwd_bytes",
    "totlenfwdpkts": "fwd_bytes",
    "totallengthofbwdpackets": "bwd_bytes", "totallengthofbwdpacket": "bwd_bytes",
    "totlenbwdpkts": "bwd_bytes",
    "synflagcount": "syn_count", "synflagcnt": "syn_count",
    "ackflagcount": "ack_count", "ackflagcnt": "ack_count",
    "finflagcount": "fin_count", "finflagcnt": "fin_count",
    "rstflagcount": "rst_count", "rstflagcnt": "rst_count",
    "pshflagcount": "psh_count", "pshflagcnt": "psh_count",
    "urgflagcount": "urg_count", "urgflagcnt": "urg_count",
    "flowiatmean": "iat_mean", "flowiatstd": "iat_std",
    "flowiatmax": "iat_max", "flowiatmin": "iat_min",
    "packetlengthmean": "pkt_len_mean", "pktlenmean": "pkt_len_mean",
    "packetlengthstd": "pkt_len_std", "pktlenstd": "pkt_len_std",
    "maxpacketlength": "pkt_len_max", "pktlenmax": "pkt_len_max", "packetlengthmax": "pkt_len_max",
    "minpacketlength": "pkt_len_min", "pktlenmin": "pkt_len_min", "packetlengthmin": "pkt_len_min",
    # Initial TCP window: a genuine packet-level measurement that the corrected CSVs expose.
    "fwdinitwinbytes": "win_mean", "initwinbytesforward": "win_mean", "initfwdwinbyts": "win_mean",
    "label": "label",
    # DAPT2020's phase column. Mapped only when a reader asks for it (see read_dapt2020).
    "stage": "_stage",
    "activity": "_activity",
}

CTU_COLUMN_MAP: dict[str, str] = {
    "starttime": "_timestamp", "dur": "duration", "proto": "protocol",
    "srcaddr": "src_ip", "sport": "src_port", "dstaddr": "dst_ip", "dport": "dst_port",
    "state": "_state", "totpkts": "_tot_packets", "totbytes": "_tot_bytes",
    "srcbytes": "fwd_bytes", "label": "label",
}

PROTOCOL_NAMES: dict[str, int] = {
    "tcp": 6, "udp": 17, "icmp": 1, "igmp": 2, "ipv6icmp": 58, "rtp": 17,
    "arp": 0, "ipv6": 41, "gre": 47, "esp": 50,
}


_NUMERIC_CANONICAL = {
    *schema.FLOW_LEVEL_COLUMNS, "win_mean", "_tot_packets", "_tot_bytes",
}


def _scan_projected(path: Path, column_map: dict[str, str], *, header: list[str] | None = None) -> pl.DataFrame:
    """Read only the columns ``column_map`` knows, renamed to canonical names.

    Everything is read as text and numeric columns are cast inside the lazy plan, so a
    malformed cell becomes null instead of aborting a 6M-row file and no unused column is
    ever materialised. When several source headers fold to one canonical name, the first
    is kept.
    """
    has_header = header is None
    if header is None:
        header = pl.read_csv(path, n_rows=0, infer_schema=False).columns
    renames: dict[str, str] = {}
    claimed: set[str] = set()
    for original in header:
        canonical = column_map.get(fold(original))
        if canonical is None or canonical in claimed:
            continue
        renames[original] = canonical
        claimed.add(canonical)
    if str(path).endswith(".gz"):  # scan_csv cannot stream compressed input; read_csv can
        lf = pl.read_csv(path, infer_schema=False, null_values=_NULLS,
                         columns=list(renames)).lazy().rename(renames)
    else:
        lf = (
            pl.scan_csv(path, infer_schema=False, null_values=_NULLS, has_header=has_header,
                        new_columns=None if has_header else header)
            .select(list(renames))
            .rename(renames)
        )
    casts = [
        pl.col(c).str.strip_chars().cast(pl.Float64, strict=False)
        for c in renames.values() if c in _NUMERIC_CANONICAL
    ]
    return lf.with_columns(casts).collect(engine="streaming")


def _parse_timestamp(column: str = "_timestamp") -> pl.Expr:
    """Parse the CIC / CTU / DAPT timestamp layouts to epoch seconds (first match wins)."""
    text = pl.col(column).cast(pl.Utf8).str.strip_chars()
    formats = [
        "%Y-%m-%d %H:%M:%S%.f",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S%.f",
        "%Y/%m/%d %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %I:%M:%S %p",
        "%d/%m/%Y %H:%M",
        "%d/%m/%y %H:%M",
        "%d-%m-%Y %H:%M:%S",
    ]
    parsed = pl.coalesce(
        [text.str.to_datetime(fmt, strict=False, time_unit="us") for fmt in formats]
    )
    numeric = text.cast(pl.Float64, strict=False)
    return pl.coalesce(
        [parsed.dt.epoch(time_unit="us").cast(pl.Float64) / 1e6, numeric]
    )


def _protocol_to_number(expr: pl.Expr) -> pl.Expr:
    as_number = expr.cast(pl.Int16, strict=False)
    as_name = (
        expr.cast(pl.Utf8).str.strip_chars().str.to_lowercase().str.replace_all(r"[^a-z0-9]", "")
        .replace_strict(PROTOCOL_NAMES, default=0, return_dtype=pl.Int16)
    )
    return pl.coalesce([as_number, as_name]).fill_null(0)


def _port_to_number(expr: pl.Expr) -> pl.Expr:
    """Parse a port column. Argus writes some ports in hex (``0x0303``)."""
    text = expr.cast(pl.Utf8).str.strip_chars()
    decimal = text.cast(pl.Int32, strict=False)
    hexadecimal = (
        pl.when(text.str.starts_with("0x"))
        .then(text.str.slice(2).str.to_integer(base=16, strict=False))
        .otherwise(None)
        .cast(pl.Int32)
    )
    port = pl.coalesce([decimal, hexadecimal]).fill_null(-1)
    # Argus encodes ICMP type/code in the port field; anything outside 0-65535 is not a port.
    return pl.when((port < 0) | (port > 65535)).then(-1).otherwise(port).cast(pl.Int32)


def attach_labels(df: pl.DataFrame, dataset: str, *, scenario: int | None = None) -> pl.DataFrame:
    """Resolve raw ``label`` into family / stage / technique / is_attack / attempted.

    Resolution runs over distinct labels only (tens, against millions of rows) and joins back.
    """
    if "label" not in df.columns:
        return df.with_columns(pl.lit(schema.BENIGN_LABEL).alias("label"))
    m = mapper(dataset)
    rows = []
    for raw in df.select(pl.col("label").cast(pl.Utf8)).unique().to_series().to_list():
        r = m.resolve(raw, scenario=scenario)
        rows.append({
            "label": raw, "family": r.family, "stage": r.stage, "technique": r.technique,
            "is_attack": r.is_attack, "attempted": is_attempted(raw),
        })
    lookup = pl.DataFrame(rows, schema={
        "label": pl.Utf8, "family": pl.Utf8, "stage": pl.Utf8, "technique": pl.Utf8,
        "is_attack": pl.Boolean, "attempted": pl.Boolean,
    })
    return df.with_columns(pl.col("label").cast(pl.Utf8)).join(lookup, on="label", how="left")


def _finalise(df: pl.DataFrame, dataset: str, source: str, *, scenario: int | None = None) -> pl.DataFrame:
    if "te" not in df.columns:
        df = df.with_columns((pl.col("ts") + pl.col("duration").fill_null(0.0)).alias("te"))
    df = attach_labels(df, dataset, scenario=scenario)
    out = schema.conform(df, source=source)
    return out.filter(pl.col("ts").is_not_null()).sort("ts")


def read_cicids(path: str | Path, *, dataset: str = "cicids2017", label_column: str = "label",
                header: list[str] | None = None) -> pl.DataFrame:
    """One CICFlowMeter CSV (CIC-IDS2017/2018 corrected, or DAPT2020) -> canonical flows.

    Duration and inter-arrival statistics are microseconds in CIC exports; they are converted
    to seconds so every dataset shares one time unit.
    """
    path = Path(path)
    df = _scan_projected(path, CIC_COLUMN_MAP, header=header)
    if "_timestamp" not in df.columns:
        raise schema.SchemaError(f"{path}: no timestamp column")
    if label_column != "label":
        if label_column not in df.columns:
            raise schema.SchemaError(f"{path}: no {label_column!r} column for labels")
        df = df.drop("label", strict=False).rename({label_column: "label"})

    if "src_ip" not in df.columns and "_flow_id" in df.columns:
        parts = pl.col("_flow_id").cast(pl.Utf8).str.split("-")
        df = df.with_columns(
            parts.list.get(0, null_on_oob=True).alias("src_ip"),
            parts.list.get(1, null_on_oob=True).alias("dst_ip"),
        )

    exprs = [_parse_timestamp().alias("ts")]
    for column in _CIC_MICROSECOND_COLUMNS:
        if column in df.columns:
            exprs.append((pl.col(column) / 1e6).alias(column))
    if "protocol" in df.columns:
        exprs.append(_protocol_to_number(pl.col("protocol")).alias("protocol"))
    for port in ("src_port", "dst_port"):
        if port in df.columns:
            exprs.append(_port_to_number(pl.col(port)).alias(port))
    df = df.with_columns(exprs)
    # CICFlowMeter writes -1 (original) or 0 (corrected, for UDP/ICMP) where a flow has no TCP
    # initial window. That is absence, not a measurement.
    if "win_mean" in df.columns:
        df = df.with_columns(
            pl.when((pl.col("win_mean") < 0) | (pl.col("protocol") != 6))
            .then(None)
            .otherwise(pl.col("win_mean"))
            .alias("win_mean")
        )

    for column in (*schema.KEY_COLUMNS, *schema.FLOW_LEVEL_COLUMNS):
        if column not in df.columns:
            df = df.with_columns(pl.lit("0.0.0.0" if column.endswith("_ip") else 0.0).alias(column))
    return _finalise(df, dataset, str(path))


def read_dapt2020(path: str | Path) -> pl.DataFrame:
    """DAPT2020 CSV: CICFlowMeter columns, labelled by the ``Stage`` column.

    One published file (``enp0s3-pvt-thursday``) has no header row; it takes the header of a
    sibling file (all DAPT flow files share the same 85 columns).
    """
    path = Path(path)
    first = pl.read_csv(path, n_rows=0, infer_schema=False).columns
    header = None
    if fold(first[0]) != "flowid":
        for sib in sorted(path.parent.glob("*.csv")):
            cols = pl.read_csv(sib, n_rows=0, infer_schema=False).columns
            if fold(cols[0]) == "flowid" and len(cols) == len(first):
                header = cols
                break
        if header is None:
            raise schema.SchemaError(f"{path}: headerless and no sibling header found")
    return read_cicids(path, dataset="dapt2020", label_column="_stage", header=header)


def read_ctu13(path: str | Path, *, scenario: int) -> pl.DataFrame:
    """One CTU-13 Argus bidirectional netflow (``.binetflow``) file.

    Argus reports totals, not a forward/backward split, so backward counts are
    ``total - forward``. Flag presence is recovered from the ``State`` string
    (e.g. ``"SRPA_SPA"``). Presence is coarser than a count but keeps the SYN/ACK/RST
    structure.
    """
    path = Path(path)
    df = _scan_projected(path, CTU_COLUMN_MAP)
    df = df.with_columns(
        _parse_timestamp().alias("ts"),
        _protocol_to_number(pl.col("protocol")).alias("protocol"),
        _port_to_number(pl.col("src_port")).alias("src_port"),
        _port_to_number(pl.col("dst_port")).alias("dst_port"),
        pl.col("duration").fill_null(0.0),
        pl.col("_tot_packets").fill_null(0.0),
        pl.col("_tot_bytes").fill_null(0.0),
        pl.col("fwd_bytes").fill_null(0.0),
    )
    state = pl.col("_state").cast(pl.Utf8).fill_null("").str.to_uppercase()
    letters = {"syn": "S", "ack": "A", "fin": "F", "rst": "R", "psh": "P", "urg": "U"}
    df = df.with_columns(
        [state.str.contains(letter, literal=True).cast(pl.Float64).alias(f"{name}_count")
         for name, letter in letters.items()]
    )
    total_packets, total_bytes = pl.col("_tot_packets"), pl.col("_tot_bytes")
    fwd_share = (pl.col("fwd_bytes") / total_bytes.clip(lower_bound=1.0)).clip(0.0, 1.0)
    df = df.with_columns(
        (total_bytes - pl.col("fwd_bytes")).clip(lower_bound=0.0).alias("bwd_bytes"),
        (total_packets * fwd_share).round().alias("fwd_packets"),
        (total_packets * (1.0 - fwd_share)).round().alias("bwd_packets"),
    )
    mean_iat = pl.col("duration") / total_packets.clip(lower_bound=1.0)
    mean_len = total_bytes / total_packets.clip(lower_bound=1.0)
    df = df.with_columns(
        mean_iat.alias("iat_mean"), pl.lit(0.0).alias("iat_std"),
        mean_iat.alias("iat_max"), mean_iat.alias("iat_min"),
        mean_len.alias("pkt_len_mean"), pl.lit(0.0).alias("pkt_len_std"),
        mean_len.alias("pkt_len_max"), mean_len.alias("pkt_len_min"),
    )
    return _finalise(df, "ctu13", str(path), scenario=scenario)

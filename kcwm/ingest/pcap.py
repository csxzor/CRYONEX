"""PCAP ingest: packet-level features that flow records cannot provide.

Ported from netforecast v1 ``ingest/pcap.py``. The NFStream plugin, raw-header parsing and
canonical mapping are unchanged; only the return type changed (a canonical DataFrame).

Flow aggregates expose volume: a SYN flood is obvious in bytes per second. They miss a
reconnaissance scan paced below every flow threshold. Header and timing detail exposes that
scan: TTL variance within a session, an unnaturally uniform TCP window, retransmissions to
closed ports.

**Header fields are parsed from raw bytes, not NFStream attributes.** NFStream's packet
object has no TTL or TCP window. v1 once read ``packet.ip_ttl``, a field that does not
exist, so every TTL statistic was silently zero on real captures. The fields are decoded
from ``packet.ip_packet`` instead; see ``tests/test_pcap.py``.

Import-safe without NFStream (the ``pcap`` extra). Only the PCAP path needs it.
"""

from __future__ import annotations

import math
from pathlib import Path

import polars as pl

from .schema import PAYLOAD_HIST_BINS, PAYLOAD_HIST_COLUMNS, conform

# Payload-size histogram edges (bytes): empty control packets, probes and beacon headers,
# small, typical and full-MTU payloads.
PAYLOAD_BINS = [0, 1, 64, 128, 256, 512, 1024, 1400, 65536]
assert len(PAYLOAD_BINS) - 1 == PAYLOAD_HIST_BINS


def parse_ip_header(buf: bytes | None) -> tuple[int | None, int, int | None, int | None]:
    """Decode ``(ttl, fragmented, tcp_window, tcp_seq)`` from raw IP bytes.

    ``None`` marks a field the packet does not carry: no window on UDP, nothing on a
    malformed frame. That keeps "not applicable" distinct from zero.
    """
    if not buf or len(buf) < 20:
        return None, 0, None, None
    version = buf[0] >> 4
    if version == 4:
        header_length = (buf[0] & 0x0F) * 4
        ttl, protocol = buf[8], buf[9]
        flags_and_offset = int.from_bytes(buf[6:8], "big")
        fragmented = int(bool((flags_and_offset >> 13) & 0x01 or flags_and_offset & 0x1FFF))
    elif version == 6:
        header_length, ttl, protocol, fragmented = 40, buf[7], buf[6], 0
    else:
        return None, 0, None, None
    window = sequence = None
    if protocol == 6 and len(buf) >= header_length + 20:
        tcp = buf[header_length:]
        sequence = int.from_bytes(tcp[4:8], "big")
        window = int.from_bytes(tcp[14:16], "big")
    return ttl, fragmented, window, sequence


def _require_nfstream():
    try:
        import nfstream
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError('PCAP ingest needs NFStream: uv sync --extra pcap') from exc
    return nfstream


def build_plugin():
    """NFPlugin accumulating per-flow TTL, window, fragmentation, payload and retransmits.

    Welford's method keeps variance in constant memory per flow, so a multi-gigabyte
    capture fits in RAM.
    """
    nfstream = _require_nfstream()

    class PacketLevelFeatures(nfstream.NFPlugin):
        def on_init(self, packet, flow):
            u = flow.udps
            u.ttl_n, u.ttl_mean, u.ttl_m2, u.ttl_values = 0, 0.0, 0.0, set()
            u.win_n, u.win_mean, u.win_m2, u.win_zero_count = 0, 0.0, 0.0, 0
            u.frag_count = u.packet_count = u.retrans_count = 0
            # Scalar counters, not a list: NFStream serialises list-valued udps to strings.
            for i in range(PAYLOAD_HIST_BINS):
                setattr(u, f"payload_hist_{i}", 0)
            u.seen_seq = set()
            self.on_update(packet, flow)

        def on_update(self, packet, flow):
            u = flow.udps
            u.packet_count += 1
            ttl, fragmented, window, sequence = parse_ip_header(getattr(packet, "ip_packet", None))
            if ttl is not None:
                u.ttl_n += 1
                d = ttl - u.ttl_mean
                u.ttl_mean += d / u.ttl_n
                u.ttl_m2 += d * (ttl - u.ttl_mean)
                if len(u.ttl_values) < 16:
                    u.ttl_values.add(ttl)
            if window is not None:
                u.win_n += 1
                d = window - u.win_mean
                u.win_mean += d / u.win_n
                u.win_m2 += d * (window - u.win_mean)
                if window == 0:
                    u.win_zero_count += 1
            if fragmented:
                u.frag_count += 1
            payload = int(getattr(packet, "payload_size", 0) or 0)
            for i in range(PAYLOAD_HIST_BINS):
                if PAYLOAD_BINS[i] <= payload < PAYLOAD_BINS[i + 1]:
                    setattr(u, f"payload_hist_{i}", getattr(u, f"payload_hist_{i}") + 1)
                    break
            # Retransmission proxy: a payload-carrying sequence number seen before.
            if sequence is not None and payload > 0:
                if sequence in u.seen_seq:
                    u.retrans_count += 1
                elif len(u.seen_seq) < 4096:
                    u.seen_seq.add(sequence)

        def on_expire(self, flow):
            u = flow.udps
            u.ttl_unique = float(len(u.ttl_values))
            u.ttl_values = None
            u.seen_seq = None
            u.ttl_var = u.ttl_m2 / u.ttl_n if u.ttl_n > 1 else 0.0
            u.win_std = math.sqrt(u.win_m2 / u.win_n) if u.win_n > 1 else 0.0
            u.frag_rate = u.frag_count / u.packet_count if u.packet_count else 0.0

    return PacketLevelFeatures()


def read_pcap(path: str | Path, *, bpf_filter: str | None = None) -> pl.DataFrame:
    """Parse a PCAP/PCAPNG into canonical (unlabelled) flows with the packet block populated.

    nDPI dissection is disabled (``n_dissections=0``). That roughly doubles throughput, and
    application identity is not part of the state vector.
    """
    nfstream = _require_nfstream()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    streamer = nfstream.NFStreamer(
        source=str(path), statistical_analysis=True, splt_analysis=0,
        n_dissections=0, bpf_filter=bpf_filter, udps=build_plugin(),
    )
    df = pl.from_pandas(streamer.to_pandas())
    if df.height == 0:
        raise ValueError(f"{path}: no flows extracted")
    return nfstream_to_canonical(df, source=str(path))


def nfstream_to_canonical(df: pl.DataFrame, *, source: str) -> pl.DataFrame:
    """Map NFStream's columns (milliseconds) onto the canonical schema (seconds)."""

    def column(name: str, default=0.0) -> pl.Expr:
        return pl.col(name) if name in df.columns else pl.lit(default)

    payload_total = sum(column(f"udps.payload_hist_{i}", 0.0) for i in range(PAYLOAD_HIST_BINS))
    canonical = df.select(
        (column("bidirectional_first_seen_ms") / 1000.0).alias("ts"),
        (column("bidirectional_last_seen_ms") / 1000.0).alias("te"),
        column("src_ip", "0.0.0.0").alias("src_ip"),
        column("dst_ip", "0.0.0.0").alias("dst_ip"),
        column("src_port", 0).cast(pl.Int32).alias("src_port"),
        column("dst_port", 0).cast(pl.Int32).alias("dst_port"),
        column("protocol", 0).cast(pl.Int16).alias("protocol"),
        (column("bidirectional_duration_ms") / 1000.0).alias("duration"),
        column("src2dst_bytes").alias("fwd_bytes"),
        column("dst2src_bytes").alias("bwd_bytes"),
        column("src2dst_packets").alias("fwd_packets"),
        column("dst2src_packets").alias("bwd_packets"),
        column("bidirectional_syn_packets").alias("syn_count"),
        column("bidirectional_ack_packets").alias("ack_count"),
        column("bidirectional_fin_packets").alias("fin_count"),
        column("bidirectional_rst_packets").alias("rst_count"),
        column("bidirectional_psh_packets").alias("psh_count"),
        column("bidirectional_urg_packets").alias("urg_count"),
        (column("bidirectional_mean_piat_ms") / 1000.0).alias("iat_mean"),
        (column("bidirectional_stddev_piat_ms") / 1000.0).alias("iat_std"),
        (column("bidirectional_max_piat_ms") / 1000.0).alias("iat_max"),
        (column("bidirectional_min_piat_ms") / 1000.0).alias("iat_min"),
        column("bidirectional_mean_ps").alias("pkt_len_mean"),
        column("bidirectional_stddev_ps").alias("pkt_len_std"),
        column("bidirectional_max_ps").alias("pkt_len_max"),
        column("bidirectional_min_ps").alias("pkt_len_min"),
        column("udps.ttl_mean").alias("ttl_mean"),
        column("udps.ttl_var").alias("ttl_var"),
        column("udps.ttl_unique").alias("ttl_unique"),
        column("udps.win_mean").alias("win_mean"),
        column("udps.win_std").alias("win_std"),
        column("udps.win_zero_count").alias("win_zero_count"),
        column("udps.frag_rate").alias("frag_rate"),
        column("udps.retrans_count").alias("retrans_count"),
        *[(column(f"udps.payload_hist_{i}") / (payload_total + 1e-9)).alias(name)
          for i, name in enumerate(PAYLOAD_HIST_COLUMNS)],
    )
    return conform(canonical, source=source).sort("ts")

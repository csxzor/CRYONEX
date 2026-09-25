"""Feature registry: every state dimension's name, group, scaling and meaning.

The world model's input vector ``x_t`` is defined here, once. The order of ``FEATURES`` is
part of the saved-model contract. Each entry carries what the explanation panel needs to
say something an analyst can act on ("half-open SYNs per source: 41 vs baseline 3"):

* ``group`` drives grouped attributions and the input masks;
* ``log`` marks heavy-tailed counts and volumes that get ``log1p`` before scaling;
* ``desc`` is the human-readable meaning.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Feature:
    name: str
    group: str
    log: bool
    desc: str
    unit: str = ""


# Groups. Masks are per group: "packet" is absent on CSV input, "ip" when a source carries
# no addresses, "lookback" until the context holds enough history.
GROUPS = ["volume", "flags", "protocol", "timing", "ports", "hosts", "packet",
          "access", "lateral", "c2", "exfil", "lookback", "history"]
MASKED_GROUPS = {"packet": "packet", "lateral": "ip", "exfil": "ip", "lookback": "lookback"}

_F = Feature
FEATURES: list[Feature] = [
    # --- volume (v1) ---
    _F("flow_count", "volume", True, "flows started in the window", "flows"),
    _F("flow_rate", "volume", True, "flow start rate", "flows/s"),
    _F("fwd_bytes_total", "volume", True, "bytes sent by initiators", "bytes"),
    _F("bwd_bytes_total", "volume", True, "bytes sent by responders", "bytes"),
    _F("bytes_rate", "volume", True, "total byte rate", "bytes/s"),
    _F("fwd_packets_total", "volume", True, "packets sent by initiators", "pkts"),
    _F("bwd_packets_total", "volume", True, "packets sent by responders", "pkts"),
    _F("packets_rate", "volume", True, "total packet rate", "pkts/s"),
    _F("bytes_per_flow_mean", "volume", True, "mean bytes per flow", "bytes"),
    _F("bytes_per_flow_std", "volume", True, "spread of bytes per flow", "bytes"),
    _F("bytes_per_flow_max", "volume", True, "largest flow", "bytes"),
    _F("packets_per_flow_mean", "volume", True, "mean packets per flow", "pkts"),
    _F("packets_per_flow_std", "volume", True, "spread of packets per flow", "pkts"),
    _F("packets_per_flow_max", "volume", True, "most packets in one flow", "pkts"),
    _F("duration_mean", "volume", True, "mean flow duration", "s"),
    _F("duration_std", "volume", True, "spread of flow duration", "s"),
    _F("duration_max", "volume", True, "longest flow", "s"),
    # --- TCP flags (v1) ---
    _F("syn_rate", "flags", False, "SYN flags per flow"),
    _F("ack_rate", "flags", False, "ACK flags per flow"),
    _F("fin_rate", "flags", False, "FIN flags per flow"),
    _F("rst_rate", "flags", False, "RST flags per flow"),
    _F("psh_rate", "flags", False, "PSH flags per flow"),
    _F("urg_rate", "flags", False, "URG flags per flow"),
    _F("syn_ack_ratio", "flags", True, "SYN to ACK ratio"),
    _F("unanswered_syn_frac", "flags", False, "share of flows with SYN but no ACK (half-open)"),
    _F("rst_frac", "flags", False, "share of flows reset"),
    _F("flag_entropy", "flags", False, "evenness of the TCP flag mix"),
    # --- protocol mix (v1) ---
    _F("tcp_frac", "protocol", False, "share of TCP flows"),
    _F("udp_frac", "protocol", False, "share of UDP flows"),
    _F("icmp_frac", "protocol", False, "share of ICMP flows"),
    _F("other_proto_frac", "protocol", False, "share of other protocols"),
    # --- inter-arrival timing (v1) ---
    _F("iat_mean_mean", "timing", True, "mean packet inter-arrival time", "s"),
    _F("iat_mean_std", "timing", True, "spread of packet inter-arrival times", "s"),
    _F("iat_mean_min", "timing", True, "fastest flow's inter-arrival time", "s"),
    _F("iat_max_max", "timing", True, "longest silence inside a flow", "s"),
    _F("iat_std_mean", "timing", True, "within-flow timing jitter", "s"),
    _F("iat_regularity", "timing", False, "timing regularity (automation signature)"),
    # --- ports and hosts (v1) ---
    _F("unique_src_hosts", "hosts", True, "distinct source hosts", "hosts"),
    _F("unique_dst_hosts", "hosts", True, "distinct destination hosts", "hosts"),
    _F("unique_dst_ports", "ports", True, "distinct destination ports", "ports"),
    _F("unique_src_ports", "ports", True, "distinct source ports", "ports"),
    _F("dst_port_entropy", "ports", False, "dispersion of destination ports"),
    _F("src_port_entropy", "ports", False, "dispersion of source ports"),
    _F("dst_ip_entropy", "hosts", False, "dispersion of destination hosts"),
    _F("src_ip_entropy", "hosts", False, "dispersion of source hosts"),
    _F("fanout_max", "hosts", True, "most destinations contacted by one source", "hosts"),
    _F("fanout_mean", "hosts", True, "mean destinations per source", "hosts"),
    _F("ports_per_src_max", "ports", True, "most destination ports touched by one source", "ports"),
    _F("ports_per_src_mean", "ports", True, "mean destination ports per source", "ports"),
    _F("low_port_frac", "ports", False, "share of flows to ports below 1024"),
    _F("ephemeral_port_frac", "ports", False, "share of flows to ephemeral ports"),
    _F("well_known_service_frac", "ports", False, "share of flows to common services"),
    _F("bidirectional_ratio_mean", "hosts", False, "mean initiator share of bytes"),
    _F("unidirectional_flow_frac", "hosts", False, "share of flows with no reply bytes"),
    _F("empty_response_frac", "hosts", False, "share of flows with no reply packets"),
    _F("flows_per_host_max", "hosts", True, "most flows started by one source", "flows"),
    _F("new_dst_port_frac", "ports", False, "share of destination ports unseen in the previous window"),
    # --- packet level (v1; PCAP, or TCP initial window from corrected CIC CSVs) ---
    _F("ttl_mean_mean", "packet", False, "mean IP TTL"),
    _F("ttl_var_mean", "packet", True, "TTL variance within sessions"),
    _F("ttl_unique_mean", "packet", False, "distinct TTLs per session"),
    _F("ttl_spread", "packet", False, "TTL range across flows"),
    _F("win_mean_mean", "packet", True, "mean TCP window size", "bytes"),
    _F("win_std_mean", "packet", True, "TCP window size variation", "bytes"),
    _F("win_zero_rate", "packet", False, "zero-window events per packet"),
    _F("frag_rate_mean", "packet", False, "IP fragment rate"),
    _F("retrans_rate", "packet", False, "TCP retransmissions per packet"),
    _F("payload_hist_0_mean", "packet", False, "share of empty-payload packets"),
    _F("payload_hist_1_mean", "packet", False, "share of 1-63 B payloads"),
    _F("payload_hist_2_mean", "packet", False, "share of 64-127 B payloads"),
    _F("payload_hist_3_mean", "packet", False, "share of 128-255 B payloads"),
    _F("payload_hist_4_mean", "packet", False, "share of 256-511 B payloads"),
    _F("payload_hist_5_mean", "packet", False, "share of 512-1023 B payloads"),
    _F("payload_hist_6_mean", "packet", False, "share of 1024-1399 B payloads"),
    _F("payload_hist_7_mean", "packet", False, "share of full-MTU payloads"),
    # --- v2: initial access / reconnaissance signatures ---
    _F("auth_conn_max", "access", True, "most connections from one source to one host's login service (21/22/23/445/3306/3389/5900)", "flows"),
    _F("auth_short_frac", "access", False, "share of login-service flows shorter than 3 s"),
    _F("web_conn_max", "access", True, "most web connections between one source and one server", "flows"),
    _F("half_open_max", "access", True, "most half-open SYNs from one source", "flows"),
    _F("seq_port_score", "access", False, "sequential port-access score of the busiest scanner"),
    _F("icmp_sweep_max", "access", True, "most hosts pinged by one source", "hosts"),
    # --- v2: lateral movement (needs internal/external addressing) ---
    _F("int_int_frac", "lateral", False, "share of flows between internal hosts"),
    _F("int_fanout_max", "lateral", True, "most internal hosts contacted by one internal host", "hosts"),
    _F("int_scan_max", "lateral", True, "most internal hosts probed on one port by one internal host", "hosts"),
    # --- v2: command and control ---
    _F("dns_max", "c2", True, "most DNS queries from one source", "flows"),
    _F("smtp_frac", "c2", False, "share of flows to mail ports (25/465/587)"),
    # --- v2: exfiltration (needs internal/external addressing) ---
    _F("out_bytes_max", "exfil", True, "most bytes one internal host sent outside", "bytes"),
    _F("out_asym_max", "exfil", False, "strongest upload/download imbalance of an internal host"),
    _F("long_out_bytes", "exfil", True, "bytes on outbound flows longer than 60 s", "bytes"),
    # --- v2: causal lookback (history before t) ---
    _F("new_pair_frac", "lookback", False, "share of host pairs unseen in the previous 64 windows"),
    _F("new_int_pairs", "lookback", True, "new internal-to-internal host pairs", "pairs"),
    _F("beacon_max", "lookback", False, "strongest periodic (beacon-like) connection pattern"),
    _F("beacon_pairs", "lookback", True, "connection pairs with beacon-like periodicity", "pairs"),
]

# --- v2.1: long memory. Precursors (a scan, a login burst, a beacon) often sit 15-30+ min
# before the compromise, beyond the model's 64-window (~11 min) context. For each attack
# signature, the strongest value in the last 30 min and 2 h (causal rolling max, per session).
HISTORY_SOURCES = ["half_open_max", "seq_port_score", "auth_conn_max", "web_conn_max",
                   "icmp_sweep_max", "int_scan_max", "int_fanout_max", "beacon_max", "dns_max",
                   "unanswered_syn_frac", "out_asym_max", "new_int_pairs"]
HISTORY_SPANS = {"h30": 180, "h120": 720}  # windows of 10 s
_by = {f.name: f for f in FEATURES}
for _tag, _n in HISTORY_SPANS.items():
    for _src in HISTORY_SOURCES:
        FEATURES.append(_F(f"{_tag}_{_src}", "history", _by[_src].log,
                           f"max over the last {_n * 10 // 60} min of: {_by[_src].desc}", _by[_src].unit))

FEATURE_NAMES: list[str] = [f.name for f in FEATURES]
N_FEATURES = len(FEATURES)
BY_NAME: dict[str, Feature] = {f.name: f for f in FEATURES}
LOG_FEATURES: frozenset[str] = frozenset(f.name for f in FEATURES if f.log)
PACKET_FEATURES: list[str] = [f.name for f in FEATURES if f.group == "packet"]
MASK_NAMES: list[str] = ["m_packet", "m_ip", "m_lookback"]


def names_in_group(group: str) -> list[str]:
    return [f.name for f in FEATURES if f.group == group]


def feature_mask_index(names: list[str] | None = None) -> list[int]:
    """For each feature, the index into ``MASK_NAMES`` gating it, or -1 if always observed."""
    lookup = {"packet": 0, "ip": 1, "lookback": 2}
    feats = FEATURES if names is None else [BY_NAME[n] for n in names]
    return [lookup[MASKED_GROUPS[f.group]] if f.group in MASKED_GROUPS else -1 for f in feats]

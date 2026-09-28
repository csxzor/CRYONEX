"""DARPA 2000 (MIT Lincoln Laboratory) LLDOS 1.0 and 2.0.2: real multi-stage intrusions
recorded as packets, with per-session labels for each attack phase.

Source: https://archive.ll.mit.edu/ideval/data/2000/ (``LLS_DDOS_1.0.tar.gz``,
``LLS_DDOS_2.0.2.tar.gz``). We use the *inside* tcpdump sensor and its ``phase-N.list``
session lists (1999 list format; times are US Eastern local time: EST in March, EDT in April;
IPs zero-padded).

Phase to ATT&CK stage, from the scenario documentation:

LLDOS 1.0 (novice attacker, 7 March 2000)
  1 ICMP sweep of four subnets              Reconnaissance      T1595.001
  2 sadmind "ping" probes of live hosts     Reconnaissance      T1046
  3 sadmind exploit, root on three hosts    InitialAccess       T1190
  4 mstream DDoS tool installed (telnet,    CommandAndControl   T1105 ingress tool transfer;
    rcp/rsh; servers register with master)                      internal-to-internal: LateralMovement
  5 DDoS launched                           Impact (flood)      T1498; control sessions: C2

LLDOS 2.0.2 (stealthier attacker, 16 April 2000)
  1 DNS HINFO query to the DNS server       Reconnaissance      T1590
  2 sadmind exploit of mill                 InitialAccess       T1190
  3 FTP upload of mstream to mill           CommandAndControl   T1105
  4 from mill: exploit and install on       LateralMovement     T1210 (internal-to-internal)
    pascal                                                      attacker's session to mill: C2
  5 DDoS launched                           Impact / C2 as in LLDOS 1.0

This dataset is **held out**: never used to train or tune anything (``holdout: true``).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from .schema import conform

EASTERN = ZoneInfo("America/New_York")
DDOS_TARGET = "131.84.1.31"
SCENARIOS = {
    "lldos1": {"dir": "lldos1/data_and_labeling/tcpdump_inside", "pcap": "LLS_DDOS_1.0-inside.dump",
               "phases": {1: "Reconnaissance", 2: "Reconnaissance", 3: "InitialAccess", 4: "CommandAndControl",
                          5: "Impact"}},
    "lldos2": {"dir": "lldos2/data_and_labeling/tcpdump_inside", "pcap": "LLS_DDOS_2.0.2-inside.dump",
               "phases": {1: "Reconnaissance", 2: "InitialAccess", 3: "CommandAndControl", 4: "LateralMovement",
                          5: "Impact"}},
}
TECHNIQUE = {"Reconnaissance": "T1595", "InitialAccess": "T1190", "CommandAndControl": "T1105",
             "LateralMovement": "T1210", "Impact": "T1498"}


def _ip(s: str) -> str:
    return ".".join(str(int(o)) for o in s.split(".")) if s and s[0].isdigit() else s


def _port(s: str) -> int | None:
    return int(s) if s.isdigit() else None


def read_phase_list(path: Path, phase: int) -> pl.DataFrame:
    rows = []
    for line in path.read_text().splitlines():
        f = line.split()
        if len(f) < 10:
            continue
        date, clock, dur = f[1], f[2], f[3]
        start = dt.datetime.strptime(f"{date} {clock}", "%m/%d/%Y %H:%M:%S").replace(tzinfo=EASTERN).timestamp()
        h, m, s = (int(x) for x in dur.split(":"))
        rows.append({"phase": phase, "t0": start, "t1": start + h * 3600 + m * 60 + s,
                     "sport": _port(f[5]), "dport": _port(f[6]), "a": _ip(f[7]), "b": _ip(f[8])})
    return pl.DataFrame(rows, schema={"phase": pl.Int8, "t0": pl.Float64, "t1": pl.Float64, "sport": pl.Int32,
                                      "dport": pl.Int32, "a": pl.Utf8, "b": pl.Utf8})


def label_flows(flows: pl.DataFrame, sessions: pl.DataFrame, phases: dict[int, str], *, slack: float = 5.0,
                internal_prefix: str = "172.16.") -> pl.DataFrame:
    """Label each flow with the phase of a listed session between the same two hosts (either
    direction) whose time span overlaps the flow's, and whose ports agree when listed."""
    f = flows.with_row_index("_fid")
    s = sessions.with_row_index("_sid")
    pairs = []
    for a, b in (("src_ip", "dst_ip"), ("dst_ip", "src_ip")):
        j = f.select(["_fid", "ts", "te", "src_ip", "dst_ip", "src_port", "dst_port"]).join(
            s, left_on=[a, b], right_on=["a", "b"], how="inner")
        same_dir = a == "src_ip"
        sp, dp = ("src_port", "dst_port") if same_dir else ("dst_port", "src_port")
        port_ok = ((pl.col("dport").is_null() | (pl.col(dp) == pl.col("dport")))
                   & (pl.col("sport").is_null() | (pl.col(sp) == pl.col("sport"))))
        # the DDoS flood carries spoofed sources: match it by destination alone below
        pairs.append(j.filter(port_ok & (pl.col("ts") <= pl.col("t1") + slack)
                              & (pl.col("te").fill_null(pl.col("ts")) >= pl.col("t0") - slack)).select(["_fid", "phase"]))
    hit = pl.concat(pairs).group_by("_fid").agg(pl.col("phase").max())
    f = f.join(hit, on="_fid", how="left")
    # phase-5 flood packets: spoofed random sources towards the DDoS target
    p5 = sessions.filter(pl.col("phase") == 5)
    if p5.height:
        lo, hi = p5["t0"].min() - slack, p5["t1"].max() + slack
        flood = (pl.col("dst_ip") == DDOS_TARGET) & pl.col("ts").is_between(lo, hi)
        f = f.with_columns(pl.when(flood).then(pl.lit(5, dtype=pl.Int8)).otherwise(pl.col("phase")).alias("phase"))
    stage = pl.col("phase").replace_strict(phases, default=None, return_dtype=pl.Utf8)
    int_src = pl.col("src_ip").str.starts_with(internal_prefix)
    int_dst = pl.col("dst_ip").str.starts_with(internal_prefix)
    # Refinements the phase lists cannot express: tool transfer or exploitation between two
    # internal hosts is lateral movement; phase-5 sessions not aimed at the target are C2.
    stage = (pl.when(pl.col("phase").is_in([3, 4]) & int_src & int_dst).then(pl.lit("LateralMovement"))
             .when((pl.col("phase") == 5) & (pl.col("dst_ip") != DDOS_TARGET)).then(pl.lit("CommandAndControl"))
             .when((pl.col("phase") == 4) & ~(int_src & int_dst) & (pl.lit(phases.get(4)) == "LateralMovement"))
             .then(pl.lit("CommandAndControl"))
             .otherwise(stage))
    f = f.with_columns(stage.fill_null("Benign").alias("stage"))
    return f.with_columns(
        pl.when(pl.col("stage") == "Benign").then(pl.lit("BENIGN"))
        .otherwise(pl.concat_str([pl.lit("LLDOS phase "), pl.col("phase").cast(pl.Utf8)])).alias("label"),
        pl.when(pl.col("stage") == "Benign").then(pl.lit("Benign")).otherwise(pl.lit("DDoSCampaign")).alias("family"),
        pl.col("stage").replace_strict(TECHNIQUE, default=None, return_dtype=pl.Utf8).alias("technique"),
        (pl.col("stage") != "Benign").alias("is_attack"),
        pl.lit(False).alias("attempted"),
    ).drop(["_fid", "phase"])


def read_darpa(pcap: str | Path, *, scenario: str) -> pl.DataFrame:
    from .pcap import read_pcap

    pcap = Path(pcap)
    spec = SCENARIOS[scenario]
    flows = read_pcap(pcap)
    d = pcap.parent
    sessions = pl.concat([read_phase_list(d / f"phase-{p}.list", p) for p in range(1, 6)
                          if (d / f"phase-{p}.list").exists()])
    return conform(label_flows(flows, sessions, spec["phases"]), source=str(pcap))


def write_sidecar(scenario: str, root: Path, out: Path) -> None:
    """Session list of all phases as one CSV next to a sample PCAP (``<pcap>.labels.csv``)."""
    spec = SCENARIOS[scenario]
    d = root / spec["dir"]
    s = pl.concat([read_phase_list(d / f"phase-{p}.list", p) for p in range(1, 6) if (d / f"phase-{p}.list").exists()])
    s.with_columns(pl.lit(scenario).alias("scenario")).write_csv(out)


def label_sidecar(flows: pl.DataFrame, path: Path) -> pl.DataFrame:
    s = pl.read_csv(path, schema_overrides={"sport": pl.Int32, "dport": pl.Int32, "phase": pl.Int8})
    scenario = s["scenario"][0]
    return conform(label_flows(flows, s.drop("scenario"), SCENARIOS[scenario]["phases"]), source=str(path))


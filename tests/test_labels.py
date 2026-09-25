"""Label mapping: the auditable decisions in configs/stage_map.yaml behave as documented."""

import pytest

from kcwm.ingest.labels import is_attempted, mapper, normalise_label
from kcwm.stages import STAGES, compromise_mask, stage_map_config


def test_stage_order_matches_config():
    stage_map_config()  # raises on mismatch
    assert STAGES[0] == "Benign" and STAGES[-1] == "Impact"
    assert not compromise_mask()[0] and not compromise_mask()[1]  # benign, recon
    assert compromise_mask()[2:].all()


@pytest.mark.parametrize("raw,stage,family", [
    ("BENIGN", "Benign", "Benign"),
    ("Portscan", "Reconnaissance", "PortScan"),
    ("FTP-Patator - Attempted", "InitialAccess", "BruteForce"),
    ("Web Attack – XSS", "InitialAccess", "WebAttack"),       # en-dash
    ("Web  Attack - Sql\xa0Injection", "InitialAccess", "WebAttack"),
    ("Infiltration - Portscan", "LateralMovement", "Infiltration"),
    ("Infiltration", "CommandAndControl", "Infiltration"),
    ("Infiltration - Attempted", "InitialAccess", "Infiltration"),  # exact key beats stripping
    ("Botnet Ares - Attempted", "CommandAndControl", "Botnet"),
    ("DDoS-LOIC-HTTP", "Impact", "DDoS"),
    ("DoS Hulk", "Impact", "DoS"),
])
def test_cicids_labels(raw, stage, family):
    r = mapper("cicids2017").resolve(raw)
    assert (r.stage, r.family) == (stage, family)
    assert r.is_attack == (stage != "Benign")


def test_unmapped_cicids_label_fails_loudly():
    with pytest.raises(KeyError):
        mapper("cicids2017").resolve("Totally New Attack")


def test_attempted_flag():
    assert is_attempted("DoS Hulk - Attempted")
    assert not is_attempted("DoS Hulk")
    assert normalise_label(" Web Attack – Brute Force ") == "web attack-brute force"


@pytest.mark.parametrize("raw,scenario,stage,family", [
    ("flow=From-Botnet-V42-TCP-CC73-Not-Encrypted", 4, "CommandAndControl", "Botnet"),
    ("flow=From-Botnet-V42-UDP-DNS", 4, "CommandAndControl", "Botnet"),   # DNS before DDoS rule
    ("flow=From-Botnet-V42-UDP-Attempt", 4, "Impact", "DDoS"),            # UDP flood scenario
    ("flow=From-Botnet-V51-6-ICMP", 10, "Impact", "DDoS"),
    ("flow=From-Botnet-V44-ICMP", 3, "CommandAndControl", "Botnet"),      # not a DDoS scenario
    ("flow=From-Botnet-V44-TCP-Attempt", 3, "Reconnaissance", "PortScan"),  # scan scenario
    ("flow=From-Botnet-V42-TCP-Attempt", 1, "CommandAndControl", "Botnet"),  # no scans in 1
    ("flow=From-Botnet-V42-TCP-Attempt-SPAM", 1, "Impact", "Spam"),
    ("flow=From-Botnet-V42-TCP-Established-HTTP-Ad-63", 1, "Impact", "ClickFraud"),
    ("flow=From-Botnet-V47-TCP-Established-HTTP-Binary-Download-12", 7, "CommandAndControl", "Botnet"),
    ("flow=To-Background-UDP-CVUT-DNS-Server", 1, "Benign", "Benign"),
    ("flow=From-Normal-V42-Stribrek", 1, "Benign", "Benign"),
])
def test_ctu13_token_rules(raw, scenario, stage, family):
    r = mapper("ctu13").resolve(raw, scenario=scenario)
    assert (r.stage, r.family) == (stage, family)


def test_ctu_cc_prefix_needs_digits():
    # "CC" followed by digits is a C&C channel; a token merely starting with "cc" is not.
    r = mapper("ctu13").resolve("flow=From-Botnet-V42-TCP-ccitt-Established", scenario=1)
    assert r.stage == "CommandAndControl"  # the default, not the c2 rule, and no error


@pytest.mark.parametrize("raw,stage", [
    ("Benign", "Benign"), ("Reconnaissance", "Reconnaissance"),
    ("Establish Foothold", "InitialAccess"), ("Lateral Movement", "LateralMovement"),
    ("Data Exfiltration", "Exfiltration"),
])
def test_dapt_stages(raw, stage):
    assert mapper("dapt2020").resolve(raw).stage == stage

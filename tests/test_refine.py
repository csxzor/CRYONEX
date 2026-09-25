"""The ATT&CK rule found by the G1 audit: lateral movement must originate inside the network."""

import polars as pl

from kcwm.features.build import refine_stages

from .conftest import make_flows


def test_external_portscan_is_recon_internal_is_lateral():
    flows = make_flows([
        {"ts": 1.0, "src_ip": "172.16.0.1", "dst_ip": "192.168.10.51", "dst_port": 80,
         "label": "Infiltration - Portscan", "family": "Infiltration", "stage": "LateralMovement",
         "is_attack": True},
        {"ts": 2.0, "src_ip": "192.168.10.8", "dst_ip": "192.168.10.5", "dst_port": 445,
         "label": "Infiltration - Portscan", "family": "Infiltration", "stage": "LateralMovement",
         "is_attack": True},
    ])
    out = refine_stages(flows, ["192.168.10.0/24"])
    assert out["stage"].to_list() == ["Reconnaissance", "LateralMovement"]
    assert out.filter(pl.col("stage") == "Reconnaissance")["technique"].item() == "T1595"

# Datasets

KC-WM trains and evaluates on public datasets only. None are redistributed. Raw data is read
from `paths.data_root` (`configs/default.yaml`, or the `KCWM_DATA_ROOT` environment variable),
laid out as `<data_root>/raw/<dataset>/...`.

| Dataset | Files used | Size | Role | Stages it covers |
|---|---|---|---|---|
| CIC-IDS2017, corrected release (Engelen et al. 2021; Liu et al. 2022) | `raw/cicids2017/{monday..friday}.csv` | 1.1 GB | primary training and evaluation | Recon, Initial Access, Lateral Movement, C2, Impact |
| CIC-IDS2017 raw captures | `raw/cicids2017/pcaps/*-WorkingHours.pcap` | 50 GB | packet-level features (TTL, TCP window, fragments, retransmissions, payload sizes) | n/a |
| CSE-CIC-IDS2018, corrected release | `raw/cicids2018/<6 days>.csv` | 21 GB | training breadth; cross-network target | Initial Access, C2, Impact |
| CTU-13 (Garcia et al. 2014) | `raw/ctu13/CTU-13-Dataset/<1..13>/*.binetflow` | 2.5 GB | real botnet C2 → DDoS/spam chains; cross-network target | Recon, C2, Impact |
| DAPT2020 (Myneni et al. 2020) | `data/raw/dapt2020/*.csv` (in this repo, gitignored) | 460 MB | real APT chain; the only real Exfiltration data | Recon, Initial Access, Lateral Movement, Exfiltration |

## Sources

* CIC-IDS2017 corrected: https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/CICIDS2017_improved.zip
* CSE-CIC-IDS2018 corrected: https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/CSECICIDS2018_improved.zip
  (4 of its 10 days have CRC errors in the published archive and are excluded: 15-02, 21-02,
  28-02 and 01-03.)
* CIC-IDS2017 PCAPs: https://www.unb.ca/cic/datasets/ids-2017.html
* CTU-13: https://mcfp.felk.cvut.cz/publicDatasets/CTU-13-Dataset/CTU-13-Dataset.tar.bz2
* DAPT2020: https://www.kaggle.com/datasets/sowmyamyneni/dapt2020 (free Kaggle account)
* MITRE ATT&CK Enterprise STIX: https://github.com/mitre-attack/attack-stix-data. Only the
  subset we use is committed (`third_party/attack_subset.json`).
* MITRE D3FEND mappings: https://d3fend.mitre.org/api/. Cached in `third_party/d3fend_subset.json`.

## Ground truth

Labels map to (family, kill-chain stage, ATT&CK technique) in `configs/stage_map.yaml`. One
further rule is applied in code: lateral movement must originate inside the network, so an
external source's scan is Reconnaissance. `docs/LABEL_AUDIT.md` (gate G1) checks every
CIC-IDS2017 attack against UNB's published schedule.

## Building

```bash
kcwm build cicids2017 cicids2018 ctu13   # ~10 min on 12 cores; 2018 needs ~9 GB RAM peak
kcwm build dapt2020                      # after downloading DAPT2020
python scripts/label_audit.py            # gate G1
```

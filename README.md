# KC-WM: Kill-Chain World Model for network attack forecasting

**Smart India Hackathon 2026 · Problem Statement 26153 · National Technical Research Organisation (NTRO)**

KC-WM learns how a network's state evolves from flow- and packet-level traffic, and
simulates its next 5 minutes. For every 10-second window it reports:

* two alert levels: ⚠️ **early warning** (a compromise is forecast within 5 minutes) and 🔴 **attack in progress**;
* the **probability of an infiltration** within the next 5 minutes;
* the **MITRE ATT&CK stage** the network is in and heading into;
* **why**: the driving flags, ports and timing features, the moments that mattered, and the
  responsible hosts;
* **what to do**: ATT&CK techniques and D3FEND countermeasures.

It runs **fully offline on a CPU laptop**, from a PCAP or a flow CSV.

* How it works: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
* Every result: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md), generated from `results/`
* How we got here, including what failed: [`docs/DEV_HISTORY.md`](docs/DEV_HISTORY.md)

## Quick start (no datasets needed)

```bash
uv sync --all-extras                     # Python 3.12, CPU-only PyTorch (https://docs.astral.sh/uv/)
uv run kcwm forecast samples/cic17_wednesday_heartbleed.csv.gz --internal 192.168.10.0/24
uv run kcwm serve                        # operator console at http://localhost:8501
```

The trained model ships in `artifacts/release/kcwm.pt`; its sha256 is in `MANIFEST.json`.
The samples in `samples/` are real traffic that the model was **not trained on**:

| Sample | What is in it |
|---|---|
| `cic17_wednesday_heartbleed.csv.gz` | CIC-IDS2017: Heartbleed exploit at 18:12 UTC; early warning at 18:09 |
| `cic17_friday_scan_ddos.csv.gz` | CIC-IDS2017: botnet C2, a port scan (18:21 UTC) and a DDoS (18:56) |
| `ctu13_s04_c2_ddos.binetflow.gz` | CTU-13 scenario 4: botnet C2, then UDP/ICMP DDoS |
| `dapt_friday_exfiltration.csv.gz` | DAPT2020: data exfiltration (20:33, 20:40) |
| `dapt_wednesday_foothold.csv.gz` | DAPT2020: repeated foothold attempts |

`kcwm forecast` accepts any PCAP/PCAPNG (parsed with NFStream), CICFlowMeter CSV, or Argus
binetflow. Pass `--internal` with your network's CIDRs.

## Reproducing everything

1. Get the datasets. See [`docs/DATASETS.md`](docs/DATASETS.md) for links and layout.
2. Run the pipeline:

```bash
make test                                # 70+ tests, no datasets needed
kcwm build cicids2017 cicids2018 ctu13 dapt2020     # windows + features (~15 min)
python scripts/label_audit.py            # gate G1: labels vs published attack schedule
kcwm campaigns cicids2017 cicids2018 ctu13 dapt2020 # kill-chain campaigns from real traffic
scripts/run_final.sh                     # final 3-seed run: every model, one test read
scripts/queue_dev_g6g7.sh                # unseen attack families (G6), new network (G7)
python scripts/faithfulness.py runs/final-p1/kcwm-s17.pt   # gate G8
python scripts/write_benchmarks.py       # regenerates docs/BENCHMARKS.md
python scripts/make_bundle.py results/final-p1/hybrid-s17.json   # release bundle
```

Model selection used `kcwm evaluate --dev`, which never reads the test split. The test
split was read once, by `scripts/run_final.sh`.

## Repository

| Path | Contents |
|---|---|
| `kcwm/ingest` | CSV, CTU-13, DAPT and PCAP readers; labels; windowing; packet-level merge |
| `kcwm/features` | 92-feature state vector, causal look-back features, scaling, quantile bins |
| `kcwm/targets` | kill-chain stages, episodes, onsets, forecast targets |
| `kcwm/synth` | kill-chain campaign synthesizer (real snippets in real benign traffic) |
| `kcwm/model` | world model, exact and Monte Carlo rollouts, training |
| `kcwm/baselines` | logistic regression, gradient boosting, LSTM, Transformer classifier |
| `kcwm/eval` | purged splits, protocols (P1, G6, G7, P4), metrics, stage metrics, reports |
| `kcwm/explain` | integrated gradients, attention, counterfactuals, host attribution, faithfulness |
| `kcwm/kb` | offline ATT&CK and D3FEND knowledge base |
| `kcwm/inference`, `kcwm/ui` | offline engine and Streamlit console |
| `configs/` | all settings; `stage_map.yaml` is the auditable label → stage → technique map |

## Licence

Apache-2.0. Datasets are not redistributed; see `docs/DATASETS.md`.

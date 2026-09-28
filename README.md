# CRYONEX: Kill-Chain World Model for network attack forecasting

**Smart India Hackathon 2026 · Problem Statement 26153 · National Technical Research Organisation (NTRO)**

CRYONEX learns how a network's state evolves from flow- and packet-level traffic, and simulates
its next 5 minutes. For every 10-second window it reports:

* two alert levels: ⚠️ **early warning** (the world model forecasts a compromise within
  5 minutes) and 🔴 **attack in progress**;
* the **probability of an infiltration** within the next 5 minutes;
* the **MITRE ATT&CK stage** the network is in, and the next stage it is heading into;
* **why**: the driving features, the moments that mattered, a what-if, and the responsible hosts;
* **what to do**: ATT&CK techniques and D3FEND countermeasures.

An optional host layer remembers each internal host's kill chain across hours or days and
forecasts its next stage. Everything runs **fully offline on a CPU laptop**, from a PCAP or a
flow CSV.

| Document | Contents |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | how it works: pipeline, world model, alerts, host layer, evaluation design |
| [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) | every result, generated from `results/` |
| [`docs/DEV_HISTORY.md`](docs/DEV_HISTORY.md) | how we got here, including what failed |
| [`docs/DATASETS.md`](docs/DATASETS.md) | where to get each dataset and how to lay it out |
| [`docs/LABEL_AUDIT.md`](docs/LABEL_AUDIT.md) | gate G1: labels checked against the published attack schedule |

## Results at a glance

P1 test split, read once, mean of 3 seeds (full tables in `docs/BENCHMARKS.md`):

| | CRYONEX hybrid | Gradient boosting | Logistic regression (required baseline) |
|---|---|---|---|
| AUPRC, 5-minute infiltration forecast | **0.847** | 0.767 | 0.650 |
| AUPRC, unseen attack families (G6) | **0.782** | 0.725 | 0.569 |
| Real attacks warned before they started | **69/207** (two-level alerts, median 5.0 min) | ≈ 0 | ≈ 0 |
| Next stage right at a real stage change | **50%** | | |

The two-level alert policy and the next-stage table were chosen after the test read, and are
reported as post-hoc.

## Quick start (no datasets needed)

```bash
uv sync --all-extras                     # Python 3.12, CPU-only PyTorch (https://docs.astral.sh/uv/)
uv run cryonex forecast samples/cic17_wednesday_heartbleed.csv.gz --internal 192.168.10.0/24
uv run cryonex serve                        # operator console at http://localhost:8501
```

Or with Docker, which needs no network at run time:

```bash
docker build -t cryonex . && docker run --rm -p 8501:8501 cryonex
```

The trained model ships in `artifacts/release/kcwm.pt`; its sha256 is in `MANIFEST.json`.
The samples in `samples/` are real traffic that the model was **not trained on**:

| Sample | `--internal` | What is in it |
|---|---|---|
| `cic17_wednesday_heartbleed.csv.gz` | `192.168.10.0/24` | CIC-IDS2017: Heartbleed exploit at 18:12 UTC; attack-in-progress alert at 18:11 |
| `cic17_friday_scan_ddos.csv.gz` | `192.168.10.0/24` | CIC-IDS2017: botnet C2, a port scan (18:21 UTC) and a DDoS (18:56) |
| `ctu13_s04_c2_ddos.binetflow.gz` | `147.32.0.0/16` | CTU-13 scenario 4: botnet C2, then UDP/ICMP DDoS |
| `dapt_friday_exfiltration.csv.gz` | `192.168.3.0/24` | DAPT2020: data exfiltration (20:33, 20:40) |
| `dapt_wednesday_foothold.csv.gz` | `192.168.3.0/24` | DAPT2020: repeated foothold attempts |
| `darpa2000_lldos{1,2}_inside.pcap` * | `172.16.0.0/16` | DARPA 2000, never trained or tuned on: sweep, break-in, tool upload, lateral move, DDoS |

\* Not committed (PCAPs are large). Generate them after downloading DARPA 2000 with
`python scripts/make_samples.py darpa`; a `.labels.csv` sidecar supplies the ground-truth ribbon.

The model raises no alerts on the CTU-13, DAPT2020 and DARPA samples; see the limits below.

`cryonex forecast` accepts any PCAP/PCAPNG (parsed with NFStream), CICFlowMeter CSV, or Argus
binetflow. Pass `--internal` with your network's CIDRs (default: RFC 1918) and `--out` to save
the full per-window timeline as JSON.

## How it works

```
PCAP / flow CSV ─► canonical labelled flows ─► 10-s windows ─► 92 structural features
   ─► CRYONEX world model (causal Transformer + kill-chain stage dynamics, 64-window context)
        └─ simulates 30 steps ahead ─► P(infiltration ≤ 5 min), stage path, next stage
   ─► + gradient-boosting detector ─► hybrid score
   ─► ⚠️ early warning (world-model forecast) · 🔴 attack in progress (hybrid)
   ─► explanations (integrated gradients, attention, what-if, host re-simulation) + ATT&CK/D3FEND
```

The world state is a latent summary of the recent traffic, the current ATT&CK stage, and the
furthest stage reached. Forecasting is simulation: an exact rollout of the belief over stage and
progress, which a 256-sample Monte Carlo rollout matches. Details in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Reproducing everything

1. Get the datasets: CIC-IDS2017, CSE-CIC-IDS2018, CTU-13, DAPT2020, and (held out) DARPA 2000.
   See [`docs/DATASETS.md`](docs/DATASETS.md) for links and layout. Set `KCWM_DATA_ROOT` or
   `paths.data_root` in `configs/default.yaml`.
2. Run the pipeline:

```bash
make test                                            # 75 tests, no datasets needed
cryonex build cicids2017 cicids2018 ctu13 dapt2020      # windows + features (~15 min)
python scripts/label_audit.py                        # gate G1: labels vs published schedule
cryonex campaigns cicids2017 cicids2018 ctu13 dapt2020  # kill-chain campaigns from real traffic
scripts/run_final.sh                                 # final 3-seed run: every model, one test read
scripts/queue_dev_g6g7.sh                            # unseen attack families (G6), new network (G7)
python scripts/faithfulness.py runs/final-p1/kcwm-s17.pt   # gate G8
python scripts/g3_dynamics.py                        # gate G3: learned next-state dynamics
python scripts/two_level.py results/final-p1 0.05    # two-level alert thresholds
python scripts/next_stage_eval.py results/final-p1   # next-stage forecast (G5)
python scripts/write_benchmarks.py                   # regenerates docs/BENCHMARKS.md
python scripts/make_bundle.py results/final-p1/hybrid-s17.json   # release bundle
```

3. Optional host kill-chain layer:

```bash
cryonex build darpa2000                                 # held out: never trained on
python -c "from kcwm.chain.build import build_hosts; build_hosts()"   # per-host tables
python scripts/chain_run.py chain-dev \
  ctu-A,ctu-B,ctu-C,cic17-monday,cic17-tuesday,cic17-wednesday,cic17-thursday,cic17-friday,cicids2018
python scripts/chain_run.py chain-heldout dapt2020,darpa2000   # held-out chains, scored once
python scripts/make_chain_bundle.py                  # artifacts/release/chain.joblib
```

Model selection used `cryonex evaluate --dev`, which never reads the test split. The test split
was read once, by `scripts/run_final.sh`. GPU training is supported; `kaggle/` packages the
dev experiments for a Kaggle GPU notebook.

## Repository

| Path | Contents |
|---|---|
| `kcwm/ingest` | CSV, CTU-13, DAPT, DARPA and PCAP readers; labels; windowing; packet-level merge |
| `kcwm/features` | feature registry and state vector, causal look-back features, scaling, quantile bins |
| `kcwm/targets`, `kcwm/data` | kill-chain episodes, onsets and forecast targets; model arrays and purged splits |
| `kcwm/synth` | kill-chain campaign synthesizer (real snippets in real benign traffic) |
| `kcwm/model` | world model, exact and Monte Carlo rollouts, training |
| `kcwm/baselines` | logistic regression, gradient boosting, LSTM, Transformer classifier |
| `kcwm/calibrate`, `kcwm/eval` | conformal thresholds; protocols (P1, P4, G6, G7), metrics, dynamics, reports |
| `kcwm/chain` | host kill-chain layer: per-host memory, next-stage and hazard forecasts |
| `kcwm/explain` | integrated gradients, attention, counterfactuals, host attribution |
| `kcwm/kb` | offline ATT&CK and D3FEND knowledge base (`third_party/`) |
| `kcwm/inference`, `kcwm/ui` | offline engine and Streamlit console |
| `configs/` | all settings; `stage_map.yaml` is the auditable label → stage → technique map |
| `scripts/` | final run, gates, alert policy, bundles, samples, benchmark writer |
| `artifacts/release/` | released model bundle and manifest |
| `results/` | every run's metrics; `docs/BENCHMARKS.md` is generated from them |

## Limits

* **Early warning is partial**: about one real attack in three is warned before it starts, at
  about 7 false early-warning windows per hour of quiet traffic. Most attacks in public
  datasets have no precursors in the traffic.
* **New networks need a short calibration**: ranking transfers partially; alert thresholds do not.
* **Near-idle networks** (DAPT2020) and CTU-13 remain hard for every model.
* **The host layer** does not yet generalise to unseen networks (DAPT2020, DARPA 2000).

## Licence

Apache-2.0. Datasets are not redistributed; see `docs/DATASETS.md`.

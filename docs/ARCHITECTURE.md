# CRYONEX architecture

**SIH 2026 · PS SIH26153 (NTRO) · Team CRYOBLUE.** CRYONEX is a *kill-chain world model*: it learns
how a network's state evolves, `P(S_t+1 | S_t)`, from flow- and packet-level traffic, simulates the
next 5 minutes, and warns defenders, in MITRE ATT&CK terms, before a compromise completes. Every
number below is generated from `results/` and listed in `docs/BENCHMARKS.md`.

## 1. Pipeline

```
 PCAP / PCAPNG ──► NFStream + raw IP-header decoder (TTL, TCP window, fragments, retransmits)
 Flow CSV ───────► CICFlowMeter (CIC-IDS2017/2018, DAPT2020) and CTU-13 Argus readers
                         │ canonical flows + label → ATT&CK stage (configs/stage_map.yaml)
                         ▼
 Network state S_t: every 10 s, 92 features (flow level + packet level) + 3 "was it measured" masks
                         │ robust scaling, re-centred on each network's first 15 min (warm-up)
                         ▼
 WORLD MODEL (PyTorch)   causal Transformer over the last 64 states (~11 min)
   belief h_t · ATT&CK stage z_t · kill-chain progress p_t · learned transitions · next-state decoder
                         │ forward simulation: 30 steps = 5 minutes
                         ▼
 P(infiltration by each minute) · stage now · next stage · early warning / attack in progress
                         │
                         ▼
 Explanations (integrated gradients, attention, what-if, host attribution) + ATT&CK / D3FEND
                         ▼
 Offline dashboard (Streamlit) and CLI (`cryonex forecast`, `cryonex serve`)
```

## 2. Components

| Part | What it does | Code |
|---|---|---|
| **Ingest** | Reads PCAP (NFStream plus our own header decoder) and CSV flow records into one canonical schema; maps every dataset label to a family, an ATT&CK stage and a technique | `kcwm/ingest`, `configs/stage_map.yaml` |
| **State** | 92 causal features per 10-s window in 12 groups: volume, TCP flags, protocol, timing (inter-arrival statistics), hosts, ports, packet level (TTL and its variance, TCP window, fragments, payload sizes, retransmissions), access (login hammering, scans), lateral, C2 (beaconing), exfiltration, look-back | `kcwm/features` |
| **World model** | State `s_t = (h_t, z_t, p_t)`: a Transformer belief, one of 7 ATT&CK stages, and progress (furthest stage reached). Transitions `P(z' \| z, p, h) = softmax(log A[z] + log B[p] + W_z h)`, with A and B starting from the ATT&CK kill-chain order. Latent dynamics `h' = LN(h + MLP[h, z', p'])`; the decoder gives a full next-state distribution (16 bins per feature) | `kcwm/model` |
| **Simulation** | An exact rollout over the 7 × 7 (stage, progress) states, 30 steps ahead, gives the infiltration probability at every step and the stage outlook; a Monte Carlo rollout agrees with it (tested) | `kcwm/model/rollout.py` |
| **Alerts** | *Attack in progress*: the calibrated average of the world model and a gradient-boosting detector (the "hybrid"), thresholded at a 3% false-positive budget. *Early warning*: the world model's own forecast, thresholded on quiet benign traffic. Both must hold for 20 s | `kcwm/inference/engine.py` |
| **Next stage** | The simulation's stage outlook blended with a kill-chain transition table learned from training data | `engine.next_stage_probs` |
| **Explain** | Integrated gradients on the forecast (which features, in real units), attention (when), group counterfactuals (what if), host attribution by re-simulating without each host (who), flagged flows | `kcwm/explain` |
| **Respond** | Offline MITRE ATT&CK techniques and D3FEND countermeasures for the stage to prepare for | `kcwm/kb`, `third_party/` |

**Training** (supervised dynamics learning on the attack timelines) minimises, jointly: current-stage
and stage-transition cross-entropy; the likelihood of the next network state, one step and 30
imagined steps ahead; the infiltration risk computed *through* the free-running simulation (it never
sees future labels); and a penalty that keeps the learned transitions close to the ATT&CK order.
About 645k parameters; one model trains in about 15 minutes on a laptop CPU.

## 3. Data and honest evaluation

* **Data:** CIC-IDS2017 (with raw PCAPs), CSE-CIC-IDS2018, CTU-13 and DAPT2020, about 60 million
  labelled flows. Synthetic kill-chain campaigns (real attack snippets injected in ATT&CK order into
  real benign traffic) augment training; they never enter headline numbers.
* **Split:** each recording is cut in time into train / calibration / test (60/15/25), purged so no
  training horizon reaches later data. Design choices were made on development data. The frozen
  model was scored on the test split **once** (`scripts/run_final.sh`); the residual decoder was a
  second read; the alert policy, its 5% budget and the next-stage table were chosen afterwards on
  the saved test scores (**post-hoc**).
* **Controls on identical data:** logistic regression (the required baseline), gradient boosting,
  an LSTM, and a Transformer classifier with the same backbone but no dynamics (AUPRC 0.815 vs
  0.820 for the world model alone). 3 seeds each (the Transformer classifier: 2).

| Result (held-out test data) | CRYONEX | Baseline |
|---|---|---|
| F1 / recall / AUPRC (hybrid vs logistic regression, same features) | **0.725 / 0.620 / 0.847** | 0.505 / 0.363 / 0.650 |
| Unseen attack families (each hidden from training; 7 of 7 won) | **0.782** AUPRC | 0.569 |
| Real attacks warned before they started (two-level alerts, 5% budget; post-hoc) | **69 / 207**, median 5 min ahead | LR at its alert threshold: 0 / 216 |
| Next ATT&CK stage at real stage changes (post-hoc) | **50.4%**, mostly from the learned table (49.6% alone) | 29.5% rollout alone |
| Explanation faithfulness: risk drop when removing the top-5 features vs 5 random ones | **500×** | pass bar 2× |
| Next-state prediction (NLL, lower is better, 10 s ahead) | **1.835** | 2.407 persistence · 2.738 autoregression · **1.202 recent-history histogram (better)** |

**Speed:** 2 hours of traffic (334k flows) are analysed in about 7 seconds on a laptop CPU.

## 4. Limits we report openly

* Early warning covers about 1 in 3 attacks and is minutes ahead, not hours; at the 5% budget,
  false early warnings fire on about 3% of quiet time (7.2 windows per hour). On dev, the
  early-warning score was at chance (0.064 vs 0.054) against 0.225 on test: unexplained.
* On a brand-new network, ranking partly transfers but alert thresholds do not; the 15-minute
  warm-up calibration is required.
* Predicting the next stage is right about half the time, mostly thanks to the learned table. The
  released decoder predicts the next state clearly worse than the recent-history histogram (1.835
  vs 1.202); a tested residual decoder beats it (1.190) but halves early warnings, so it is not
  released.

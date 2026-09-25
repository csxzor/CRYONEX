# Technical presentation: 5 slides

Speaker-ready content. Numbers are from `docs/BENCHMARKS.md` (the final test split, read once;
3 seeds).

---

## Slide 1: The problem, and our answer

**Intrusion detection classifies a moment. An infiltration is a process.**

* Recon, then access, then C2, then lateral movement, exfiltration and impact.
* A defender needs to know *where the network is heading*, not only what just happened.

**KC-WM, the Kill-Chain World Model.** It learns `P(S_t+1 | S_t)` over network states, simulates
the next 5 minutes, and reports:

1. infiltration risk;
2. the MITRE ATT&CK stage now and next;
3. why (features, moments, hosts);
4. what to do (D3FEND).

It runs fully offline on a CPU laptop, from a PCAP or a flow CSV.

---

## Slide 2: Architecture

Flow + packet features → 92-dimensional state every 10 s → world model → rollout → hybrid alert, stage, explanation.

* **State.** Flags, ports, timing, fan-out, beacon periodicity, new internal edges, exfil
  asymmetry. Packet level (TTL, TCP window, fragments, retransmits) comes from the raw
  captures.
* **World model.** A causal Transformer belief, plus an explicit kill-chain state (stage,
  progress) whose transition matrix starts from ATT&CK order. It has latent dynamics and a
  binned next-state decoder.
* **Forecast = simulation.** Exact belief propagation over 49 stage states, 30 steps ahead
  (Monte Carlo agrees). The risk is trained *through* the rollout.
* **Hybrid alert.** A calibrated mean of the world model and a gradient-boosted detector. The
  world model provides stages, forecasts and explanations.

---

## Slide 3: Results (held-out test, read once)

| | AUPRC | vs required LR |
|---|---|---|
| **KC-WM hybrid** | **0.847** | **+0.20** |
| KC-WM world model alone | 0.820 | +0.17 |
| Gradient boosting | 0.767 | +0.12 |
| Logistic regression (required) | 0.650 | — |

* **Unseen attack families** (7 leave-one-out tests): hybrid **0.78** vs LR 0.57, winning 7/7. The world model alone beats gradient boosting on 6/7.
* **Explanations are faithful:** deleting the top-5 attributed features lowers risk **500×** more than deleting random ones.

---

## Slide 4: Why the numbers can be trusted

* Temporal, purged splits. **The test split was read once**, and every choice was made on dev data.
* 3 seeds; block-bootstrap CIs; controls on identical rows (LR, gradient boosting, LSTM, and a Transformer with the same backbone).
* Labels audited against the published attack schedule. The audit caught an error in the corrected CIC-IDS2017: an external scan labelled as lateral movement.
* Beacon bursts count as one episode, not hundreds of "onsets".
* **Stated limits:**
  * no model warns before *real* attacks in these public datasets, whose attacks have no precursors;
  * alert thresholds need a short calibration on a new network;
  * near-idle networks (DAPT2020) remain hard.

---

## Slide 5: Decision support and deployment

* **Console (offline Streamlit):** risk timeline, simulated-future fan, kill-chain progress, driving features in native units, host attribution with flagged flows, and ATT&CK + D3FEND cards.
* **Next-stage prediction on kill-chain campaigns:** about 50% vs 14% for a count-based Markov rule. No classifier can do this at all.
* **Deployment:** `uv sync && kcwm serve`, CPU only, a 4 MB model; a 2-hour, 334k-flow capture analysed in under a minute.
* **For CII:** ingests NetFlow, CICFlowMeter or PCAP; calibrate once on 15 min of normal traffic; every alert carries its explanation and ATT&CK/D3FEND mapping.

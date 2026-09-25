# KC-WM: Kill-Chain World Model for network attack forecasting

**SIH 2026 · PS 26153 · NTRO.** Architecture summary (2 pages). Every number cited here is
produced by the repository and listed in `docs/BENCHMARKS.md`.

## 1. What it does

KC-WM learns how a network's state evolves (`P(S_t+1 | S_t)`) from flow- and packet-level
telemetry, and rolls that model forward to forecast whether the current trajectory leads to
compromise within the next 5 minutes. For every 10-second window it reports:

* **infiltration risk:** the probability that a compromise-stage window (Initial Access,
  Lateral Movement, C2, Exfiltration, Impact) occurs within K = 30 windows;
* **ATT&CK stage:** where the network is now, and which stage it is heading into;
* **why:** driving features, influential past windows, counterfactuals, and responsible hosts;
* **what to do:** ATT&CK technique cards and D3FEND countermeasures (offline).

It runs fully offline on a CPU laptop. A 188k-flow capture is analysed in about 12 s.

## 2. Pipeline

```
PCAP ─► NFStream + raw-header plugin ─┐   (TTL, TCP window, fragments, retransmits, payload sizes)
CSV  ─► CIC / CTU-13 / DAPT adapters ─┴─► canonical labelled flows
     ─► 10-s windows per capture session ─► x_t: 92 features in 12 groups + 3 masks
        (volume, flags, timing, ports, hosts, packet, access, lateral, C2, exfil, look-back)
     ─► scaling: robust, fitted on non-empty training windows; warm-up re-centring per network
     ─► KC-WM world model ─┬─► rollout risk, stage forecast, surprise ─┐
     ─► gradient boosting  ─┘                                           ├─► hybrid alert score
                                                                        └─► explanations + ATT&CK/D3FEND
```

* **State features** are structural and per-source, so a single brute-forcer or beaconing bot
  stays visible in a busy window. Examples: login-service hammering, half-open SYNs,
  sequential port access, internal fan-out, new internal host pairs, beacon periodicity, and
  outbound byte asymmetry. They are causal: flows after `t` never change `x_t` (tested).
* **Packet level** comes from the raw CIC-IDS2017 captures (98.8% window coverage). Where a
  source has no PCAP, an explicit mask says so, rather than feeding zeros that look like
  measurements.

## 3. The world model

World state `s_t = (h_t, z_t, p_t)`:

* `h_t`, a causal Transformer (3 layers, d = 96) over the last 64 windows (about 11 min);
* `z_t`, the kill-chain stage (7 ATT&CK phases);
* `p_t`, the furthest stage reached (`p' = max(p, z')`).

Learned components:

| Component | Form |
|---|---|
| nowcast | `q(z_t | h_t)` |
| stage transition | `P(z' | z, p, h) = softmax(log A[z] + log B[p] + W_z h)`, with A and B initialised from, and regularised toward, an ATT&CK-ordered kill chain |
| latent dynamics | `h' = LN(h + MLP[h, emb(z'), emb(p')])` |
| emission | `p(x' | h') = Π_f Categorical(16 quantile bins)`, a full next-state distribution |

**Forecasting is simulation.** The analytic rollout propagates the exact joint belief over
(stage, progress), 49 states, 30 steps ahead. It gives `P_infil(k)` for every k, the stage
marginals and time-to-stage. Monte Carlo rollouts (256 sampled futures) agree with it (tested).

**Training** combines: nowcast and transition cross-entropy; one-step and 30-step imagined
next-state likelihood (the dynamics objective); risk BCE through the free-running rollout
(no future labels); a direct risk head on the belief (pre-registered fallback F-A); and an
ATT&CK prior penalty.

**Campaigns.** Public captures schedule attacks independently, so almost nothing precedes a
compromise. The synthesizer injects real attack snippets, in kill-chain order, into real
benign traffic of the same network and split, with one consistent victim host. They are
used as training augmentation and as a separate benchmark (P4), never in headline numbers.

**Hybrid alert score.** The mean of the Platt-calibrated world-model risk and a
Platt-calibrated gradient-boosting detector on the current window. It was chosen on dev
data because the two are complementary:
* the detector separates networks by risk level;
* the world model discriminates better within a network and on unseen attacks.

**Two alert levels.**
* ⚠️ *Early warning* fires when the world model's own forecast crosses a threshold set on
  quiet benign calibration windows (3% budget). The world model carries the early signal;
  the detector does not.
* 🔴 *Attack in progress* fires on the hybrid score.

The policy was chosen after the final test read and is reported as post-hoc.

## 4. Evaluation design (why the numbers can be trusted)

* **Temporal, purged splits.** Each session is cut 60/15/25. No training horizon reaches
  calibration or test, and scalers are fitted on training rows only (tested).
* **Dev mode.** All choices were made on calibration data. **The test split was read once**,
  by `scripts/run_final.sh`.
* **Episode-level onsets.** Botnet beacon bursts count as one episode, not hundreds of
  "onsets".
* **An early-warning subset** reports windows with no compromise in the last 5 minutes, so
  detecting an attack already in progress cannot pass for forecasting.
* **Block-bootstrap CIs** use 30-minute blocks, and every model runs 3 seeds.
* **Controls on identical rows:** logistic regression (required), stacked LR, gradient
  boosting, an LSTM, and a Transformer classifier with the same backbone but no dynamics.
* **Gates**, pre-registered in the plan:
  * G1 labels vs. the published schedule;
  * G3 dynamics;
  * G4 forecast;
  * G5 stages;
  * G6 unseen attack families;
  * G7 new network;
  * G8 explanation faithfulness.

## 5. Explainability and decision support

* **When:** attention rollout over the 64-window context, plus occlusion.
* **What:** integrated gradients on the rollout's own risk against a benign baseline, grouped
  into feature families and shown in native units. **Faithful:** deleting the top-5 features
  lowers risk 53x more than deleting 5 random ones (G8).
* **What-if:** the smallest set of feature groups whose reset to normal drops the risk
  below the alert threshold.
* **Who:** each host's contribution, by removing its recent flows, re-aggregating, and
  re-simulating.
* **Respond:** ATT&CK techniques extracted from MITRE's STIX bundle, and D3FEND
  countermeasures from MITRE's D3FEND API, both committed for offline use.

## 6. Honest limits

* **Early warning is partial.** On the final test, the two-level alerts warned 25% of real
  attacks before they began (median about 3.5 min). Most attacks in public datasets were
  launched on a schedule with no precursors, and nothing in the traffic can predict them.
  Standard detectors (gradient boosting, logistic regression) warn about 0% in advance.
* **New networks need a short calibration.** Ranking transfers partially (CIC-2017,
  CIC-2018), but alert thresholds do not transfer, and no model transfers to CTU-13.
* **Near-idle networks** (DAPT2020: 49% empty windows, median 1 flow) remain hard for every
  model.
* **Next-stage prediction** beats simple rules on campaigns (about 50% vs 14%), but not on real
  data, where successive attacks are mostly unrelated.

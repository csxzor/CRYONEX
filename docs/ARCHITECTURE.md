# CRYONEX architecture

**SIH 2026 · PS 26153 · NTRO.** How the Kill-Chain World Model turns network telemetry into
forecasts, and how it was evaluated. Every number cited here is produced by the repository and
listed in `docs/BENCHMARKS.md`; the decisions behind them, including the ideas that failed, are
in `docs/DEV_HISTORY.md`.

## 1. What it does

CRYONEX learns how a network's state evolves, `P(S_t+1 | S_t)`, from flow- and packet-level
telemetry, and rolls that model forward to forecast whether the current trajectory leads to
compromise. For every 10-second window it reports:

| Output | Meaning |
|---|---|
| **Infiltration risk** | probability that a compromise-stage window (Initial Access, Lateral Movement, C2, Exfiltration, Impact) occurs within K = 30 windows (5 min) |
| **Two alert levels** | ⚠️ *early warning*: the world model forecasts compromise; 🔴 *attack in progress*: the hybrid score crosses its calibrated threshold |
| **ATT&CK stage** | the stage the network is in now, and the next *new* stage it is heading into |
| **Why** | driving features in native units, the past windows that mattered, a counterfactual, and the responsible hosts |
| **What to do** | ATT&CK technique cards and D3FEND countermeasures, offline |
| **Per-host kill chain** (optional layer) | for each attacked internal host: the stages it has been through over hours or days, its next stage, and P(compromise step within 1 h / 24 h) |

Everything runs offline on a CPU. A 64k-flow CSV is analysed in about 9 s; a 188k-flow capture
in about 12 s.

## 2. System overview

```
                 ┌──────────────────────────── kcwm/ingest ───────────────────────────┐
 PCAP/PCAPNG ──► │ NFStream + raw-header plugin (TTL, TCP window, fragments, retx)    │
 CICFlowMeter ─► │ CIC / CTU-13 / DAPT / DARPA adapters ──► canonical labelled flows  │
 Argus netflow ► │ label ─► family ─► ATT&CK stage (configs/stage_map.yaml)           │
                 └──────────────────────────────┬─────────────────────────────────────┘
                                                │ flows (ts, src, dst, ports, flags, bytes, stage)
                  ┌─────────────────────────────┴───────────────────────────────┐
                  ▼                                                             ▼
   NETWORK LAYER (10-min memory)                                  HOST LAYER (days of memory)
   kcwm/features: 10-s windows per session,                       kcwm/chain/hosts: one row per
   92 features in 12 groups + 3 masks                             (internal host, 10-s window)
                  │                                                             │
   scaling (warm-up or global) + 16 quantile bins                 stage nowcast per host window (GBT)
                  │                                                             │
   ┌──────────────┴──────────────┐                                causal tracker: per-host and
   ▼                             ▼                                network-wide stage memory
   CRYONEX world model         gradient boosting                                  │
   (64-window context)       (current window)                     next-stage + hazard forecasters
   │ rollout risk, stage       │ detector risk                                   │
   │ marginals, surprise       │                                                 │
   ├──── Platt ───┐   ┌─ Platt ┘                                                 │
   │              ▼   ▼                                                          │
   │         hybrid score ──► 🔴 attack in progress                              │
   └──► raw rollout risk ──► ⚠️ early warning                                    │
   stage marginals × transition table ──► next stage                             │
                  │                                                             │
                  └─────────────► kcwm/explain + kcwm/kb ◄──────────────────────┘
                                  why · who · what-if · ATT&CK / D3FEND
                                                │
                            kcwm/inference/engine ──► CLI (cryonex forecast) · Streamlit console
```

The two layers answer different questions. The network layer sees the last ~11 minutes of the
whole network and forecasts the next 5. The host layer remembers what each host has been through
for as long as the recording lasts, because real kill chains move one stage at a time over hours
or days (DAPT2020: one stage per day), which no 10-minute context can see.

## 3. Ingestion and labels (`kcwm/ingest`, `kcwm/stages.py`)

| Reader | Input | Notes |
|---|---|---|
| `pcap.py` | PCAP / PCAPNG | NFStream flows plus a raw-header plugin for TTL, TCP window, fragments, retransmits and payload sizes |
| `flowcsv.py` | CICFlowMeter CSV (CIC-IDS2017, CSE-CIC-IDS2018 corrected), DAPT2020 CSV, CTU-13 Argus binetflow | one adapter per dialect |
| `darpa.py` | DARPA 2000 LLDOS inside-sensor tcpdump + `phase-N.list` | session lists matched to flows by host pair, ports and time (US Eastern local time); a `.labels.csv` sidecar next to a PCAP is picked up as ground truth for display |
| `packet_merge.py` | CIC-IDS2017 raw captures | adds packet-level fields to CSV flows (98.8% window coverage) |
| `schema.py` | | `conform()` maps every reader onto one canonical flow schema |
| `labels.py` | | dataset label → attack family → stage, from `configs/stage_map.yaml` |
| `windowing.py` | | 10-s windows; a capture is split into sessions at gaps over 5 minutes |

**Stages** are the seven entries of `kcwm/stages.py`, in ATT&CK matrix order: Benign,
Reconnaissance, Initial Access, Lateral Movement, Command and Control, Exfiltration, Impact. The
order is part of every saved model's contract (it defines *progress*), so it is append-only.
`configs/stage_map.yaml` is the single auditable place where labels become stages; unmapped
labels fail loudly. One rule lives in code: lateral movement must originate inside the network,
so an external scan is Reconnaissance (found by the G1 label audit, `docs/LABEL_AUDIT.md`).

**Targets** (`kcwm/targets`). Attack windows separated by at most 30 benign windows form one
*episode*; an *onset* needs 5 quiet minutes before it, so a botnet's beacon bursts count as one
episode, not hundreds of onsets. The forecast target at window `t` is "a compromise-stage window
occurs in `t+1 … t+K`", for K ∈ {6, 12, 30}; K = 30 (5 min) is primary.

## 4. State features (`kcwm/features`)

Each window becomes a state vector `x_t`. The registry (`features/registry.py`) defines 116
features in 13 groups; the released model uses 92 of them in 12 groups, because the long-memory
`history` group (30-min and 2-h rolling maxima) cost about 0.03 pooled AUPRC on dev for a small
early-warning gain (`features.exclude_groups` in `configs/default.yaml`).

| Group | # | Examples |
|---|---|---|
| volume | 17 | flows, packets, bytes, per-direction totals |
| flags | 10 | SYN/ACK/RST/FIN shares, half-open SYNs |
| protocol | 4 | TCP/UDP/ICMP mix |
| timing | 6 | inter-arrival and duration statistics |
| ports | 10 | distinct ports, sequential port access, well-known service hits |
| hosts | 10 | distinct sources and destinations, internal fan-out, new internal host pairs |
| packet | 17 | TTL, TCP window, fragments, retransmits, payload sizes (PCAP only) |
| access | 6 | login-service hammering, failed-connection shares |
| lateral | 3 | internal-to-internal fan-out and first contacts |
| c2 | 2 | beacon periodicity, long-lived external sessions |
| exfil | 3 | outbound byte asymmetry |
| lookback | 4 | causal summaries of the windows before `t` |
| *history* | *24* | *excluded from the released model* |

Design rules, each covered by a test:

* **Structural and per-source.** A single brute-forcer or beaconing bot stays visible in a busy
  window, because features measure the most anomalous source, not only totals.
* **Causal.** Flows after `t` never change `x_t`.
* **Explicit masks.** Three masks (`packet`, `ip`, `lookback`) say when a group was not
  measured (for example a CSV source has no TTL), instead of feeding zeros that look like
  measurements.

**Scaling** (`features/normalize.py`). A robust scaler is fitted on non-empty training windows
only (DAPT2020 is 49% empty windows and would otherwise drag every median to zero). Two modes:
`global` (one scaler) and `warmup` (re-centred on the first 90 windows of each session, never
scored), which is what makes thresholds survive a change of network. The released model uses
`warmup`. The world model's emission also needs a discrete target, so each feature is cut into
16 quantile bins fitted on the same rows.

## 5. The world model (`kcwm/model`)

World state `s_t = (h_t, z_t, p_t)`:

* `h_t`: a causal Transformer (3 layers, d = 96) over the last 64 windows (about 11 minutes);
* `z_t`: the kill-chain stage, one of 7;
* `p_t`: progress, the furthest stage reached (`p' = max(p, z')`).

| Component | Form |
|---|---|
| nowcast | `q(z_t | h_t)` |
| stage transition | `P(z' | z, p, h) = softmax(log A[z] + log B[p] + W_z h)`; A and B initialised from, and regularised toward, an ATT&CK-ordered kill chain |
| latent dynamics | `h' = LN(h + MLP[h, emb(z'), emb(p')])` |
| emission | `p(x' | h') = Π_f Categorical(16 bins)`, a full next-state distribution |

**Forecasting is simulation** (`rollout.py`). The analytic rollout propagates the exact joint
belief over (stage, progress), 7 × 7 = 49 states, 30 steps ahead, giving `P_infil(k)` for every
k, the stage marginals and time-to-stage. A Monte Carlo rollout (256 sampled futures) agrees
with it (tested). *Surprise* is the one-step negative log-likelihood `-log p(x_t | h_t-1)`,
marginalised over the unknown next stage.

**Training** (`train.py`) sums:

* nowcast and transition cross-entropy, with stage weights;
* one-step and imagined 30-step next-state likelihood (the dynamics objective), with scheduled
  sampling;
* risk BCE through the free-running rollout, which never sees future labels;
* a direct risk head on the belief, averaged with the rollout (pre-registered fallback F-A);
* the ATT&CK prior penalty `KL(A0 ‖ A)` plus L2 on B's drift, so the kill chain stays
  recognisable.

**Residual decoder (tested variant, not released).** Gate G3 showed that the frozen model's
next-state predictions beat persistence and linear autoregression but lose to the plain
histogram of the last 64 windows, because traffic features are close to stationary over minutes.
With `emit_prior=1` the emission becomes *recent-history histogram × exp(decoder(h'))*, with the
decoder's last layer zero-initialised, so any gain over the histogram is learned dynamics. It
wins G3 at 10 s and 1 min, but halves early warning (24/207 attacks warned vs 69/207), so the
frozen decoder remains the release; the variant is kept at
`artifacts/kcwm-residual-decoder.pt` and reported in `BENCHMARKS.md` §7.

## 6. From forecast to alerts (`kcwm/calibrate`, `kcwm/inference/engine.py`)

**Hybrid score.** The mean of the Platt-calibrated world-model risk and a Platt-calibrated
gradient-boosting detector on the current window. It was chosen on dev because the two are
complementary: the detector separates networks by their risk level; the world model
discriminates better within a network and on unseen attack families.

**Thresholds** come from split-conformal calibration on the calibration split (at a 3% FPR
budget) and are applied unchanged to test and deployment. Alerts need 2 sustained windows
(hysteresis).

**Two alert levels** (post-hoc: chosen after the final test read).

| Level | Fires when | Threshold source |
|---|---|---|
| ⚠️ Early warning | raw world-model rollout risk ≥ threshold for 2 windows, while no level-2 alert is on | (1 − budget) quantile on *quiet benign* calibration windows; budget 5% |
| 🔴 Attack in progress | hybrid score ≥ threshold | conformal, 3% FPR budget |

The world model carries the early signal; the gradient-boosting detector warns about 0% in
advance. The budget trades warnings for noise (`scripts/two_level.py`): 3% warns 51/207 real
test attacks at 3.4 false-warning windows per hour, 5% warns 69/207 at 7.2, 10% warns 91/207 at
31.5.

**Next stage.** The rollout's stage mass alone gets 29% of real test stage changes right. The
released forecast is `rollout^0.25 × table^0.75`, where the table `P(next | current, progress)`
is counted on training labels and read at the model's *own* current-stage belief and progress;
that gives 50% (the table at the true current stage, an upper bound, gives 58%). The table and
the weight (`stages.next_stage_blend`) are stored in the release bundle.

## 7. Host kill-chain layer (`kcwm/chain`)

A second, lighter model that remembers what each internal host has done and forecasts its next
stage. It exists because within-10-minute windows cannot see chains whose stages are hours or
days apart.

| Module | Role |
|---|---|
| `hosts.py` | one row per (internal host, 10-s window) with 28 structural features in both roles (outgoing as source, incoming as destination): distinct peers and ports, failed-connection shares, auth-service hits, first contacts, bytes leaving the network; plus scale-free shares and 1- and 5-minute rolling activity, because a brute force of 10 logins per 10 s (CIC-IDS2017) and one of 320 (CIC-IDS2018) have the same *share* of failed auth flows |
| `targets.py` | per-host stage episodes (merged over 1-hour gaps) and **chain steps**: a host starting a stage different from the one it was most recently doing. A *timeline* is one network's continuing recording (CIC-IDS2017's week, DAPT2020's week); every CTU-13 and DARPA scenario is its own |
| `tracker.py` | causal memory at forecast anchors (every 5 minutes): per host and stage, the strongest evidence, count, and minutes since first and last seen; the same for the whole network (an attack on one host raises its neighbours' odds); current activity; and the ATT&CK prior's next-stage distribution as input features |
| `model.py` | three gradient-boosted tree models: the per-host-window stage nowcast, the next-stage classifier, and hazards P(compromise step within 5 min / 1 h / 24 h). Trees, because there are only a few hundred real chain steps |
| `run.py`, `evaluate.py` | leave-one-group-out protocol with cross-fitted nowcast probabilities, so the forecaster learns from memory as noisy as the memory it sees at test time; scored against an ATT&CK-order rule, a lookup table and the majority class, each given the *true* history |
| `infer.py` | `analyze_hosts()` for one capture: tracked hosts, their chain so far, forecast next stage, and the forecast every 5 minutes |

Evaluation groups: CTU-13 in three scenario blocks, each CIC-IDS2017 day, CIC-IDS2018, DAPT2020,
and DARPA 2000 (never trained on). Design choices were made on the CTU-13 and CIC folds;
DAPT2020 and DARPA 2000 were scored once afterwards. On dev folds the learned model reaches 0.41
next-stage top-1 from inferred history (0.56 from true history). On the held-out chains it is
weak (0.10 on both): the per-host nowcast recognises only 15% (DAPT) and 8% (DARPA) of attack
host-windows on those unseen networks.

The layer is optional in deployment. `scripts/make_chain_bundle.py` trains it on every group
except DARPA 2000 and writes `artifacts/release/chain.joblib`; when that file sits next to
`kcwm.pt`, the engine runs it and the console adds a "Kill chain by host" panel.

## 8. Explainability and decision support (`kcwm/explain`, `kcwm/kb`)

| Question | Method |
|---|---|
| **When** | attention rollout over the 64-window context, plus occlusion |
| **What** | integrated gradients on the rollout's own risk against a benign baseline, grouped into feature families and shown in native units |
| **What-if** | the smallest set of feature groups whose reset to normal drops the risk below the alert threshold |
| **Who** | each host's contribution: remove its recent flows, re-aggregate the windows, re-simulate |
| **Respond** | ATT&CK techniques from MITRE's STIX bundle and D3FEND countermeasures from MITRE's D3FEND API, both committed as subsets in `third_party/` for offline use |

The attributions are faithful (gate G8): on the 40 highest-risk calibration windows, deleting
the top-5 integrated-gradients features lowers the forecast risk by 0.730 on average, against
0.001 for 5 random features.

## 9. Runtime and packaging

**Release bundle** (`artifacts/release/`):

| File | Contents |
|---|---|
| `kcwm.pt` | world-model weights and config, robust scaler, quantile binner, gradient-boosting detector, Platt calibrators, hybrid and early-warning thresholds, next-stage transition table and blend weight, provenance |
| `MANIFEST.json` | sha256, thresholds, the source result and checkpoint, training datasets |
| `chain.joblib` | optional host kill-chain layer (§7) |

`scripts/make_bundle.py` builds `kcwm.pt` from a final-run result (`results/final-p1/hybrid-s17.json`).

**Inference** (`kcwm/inference/engine.py`). `load_bundle()` → `analyze(path)` reads any
supported input (`read_any` detects the format), windows and scales it, runs the world model
and detector in batches, and returns an `Analysis`: the per-window timeline (risk, both alert
levels, nowcast stage, progress, next stage and its probability, surprise), the stage
marginals, the flows, and the host analysis when the chain layer is present. `summary()` and
`to_json()` serve the CLI.

**Interfaces.**

* `cryonex forecast <file> --internal <CIDRs> [--out timeline.json]`: offline CLI summary.
* `cryonex serve`: Streamlit console (`kcwm/ui/app.py`) with the risk timeline and ground-truth
  ribbon when labels exist, a time slider, the stage chain, the "why / who / what-if / respond"
  panels, the learned transition matrix, and the host kill-chain panel.
* `Dockerfile`: a CPU image with the release bundle and samples; CI checks that it starts with
  `--network none`.

## 10. Evaluation design

**Protocols** (`kcwm/eval/protocols.py`, `kcwm/eval/run.py`):

| Protocol | Question | Split |
|---|---|---|
| P1 | forecasting on real traffic, all four training datasets | each session cut 60/15/25 in time (train / calibration / test), purged so no training horizon reaches calibration or test |
| P4 | forecasting real kill chains | synthetic campaigns (below), test span only, never pooled with P1 |
| G6 | unseen attack families | leave one family out, including every anchor whose context or horizon touches it |
| G7 | a new network | train on three datasets, test on every window of the fourth |
| Chain | host kill-chain forecasting | leave one capture group out (§7); DARPA 2000 test-only |

**Campaigns** (`kcwm/synth`). Public captures schedule attacks independently, so almost nothing
precedes a compromise. The synthesizer injects real attack snippets, in kill-chain order, into
real benign traffic of the same network and split, with one consistent victim host. Campaigns
augment training and form the P4 benchmark; they never enter headline numbers.

**Discipline.**

* All choices were made in dev mode (`cryonex evaluate --dev`), which never reads the test split.
  The test split was read once, by `scripts/run_final.sh`. Later additions (the two-level
  policy, the next-stage table, the residual decoder) are labelled post-hoc or as a second read.
* Scalers, bins and thresholds are fitted on training or calibration rows only (tested).
* The **early-warning subset** reports windows with no compromise in the previous 5 minutes, so
  detecting an attack already in progress cannot pass for forecasting.
* Block-bootstrap CIs use 30-minute blocks; every model runs 3 seeds (17, 23, 29).
* Controls run on identical rows: logistic regression (required by the problem statement),
  logistic regression over the last 6 windows, gradient boosting, an LSTM, and a Transformer
  classifier with the same backbone but no dynamics.

**Gates**, pre-registered in the plan:

| Gate | Checks | Where |
|---|---|---|
| G1 | labels vs the published attack schedule | `scripts/label_audit.py` → `docs/LABEL_AUDIT.md` |
| G3 | the learned dynamics predict the next state better than simple predictors | `scripts/g3_dynamics.py`, `kcwm/eval/dynamics.py` |
| G4 | forecast quality vs the controls | P1, P4 |
| G5 | stage nowcast and next-stage forecast | `kcwm/eval/stages.py`, `scripts/next_stage_eval.py` |
| G6 | unseen attack families | `scripts/queue_dev_g6g7.sh` |
| G7 | a new network | `scripts/queue_dev_g6g7.sh` |
| G8 | explanation faithfulness | `scripts/faithfulness.py` |

**Headline results** (P1 test, 3 seeds): hybrid AUPRC 0.847 ± 0.007 vs 0.650 for logistic
regression and 0.767 for gradient boosting; unseen families (G6) mean 0.782 vs 0.569.

## 11. Code map

| Path | Contents |
|---|---|
| `kcwm/ingest` | readers (CSV, CTU-13, DAPT, DARPA, PCAP), canonical schema, labels, windowing, packet-level merge |
| `kcwm/features` | feature registry, window aggregation, causal look-back, scaling, quantile bins |
| `kcwm/targets` | episodes, onsets, forecast targets |
| `kcwm/data` | model arrays, context indexing, purged temporal splits |
| `kcwm/synth` | kill-chain campaign synthesizer |
| `kcwm/model` | world model, analytic and Monte Carlo rollouts, training |
| `kcwm/baselines` | logistic regression, gradient boosting, LSTM, Transformer classifier |
| `kcwm/calibrate` | split-conformal thresholds |
| `kcwm/eval` | protocols, metrics, stage metrics, dynamics (G3), reports |
| `kcwm/chain` | host kill-chain layer |
| `kcwm/explain` | integrated gradients, attention, counterfactuals, host attribution |
| `kcwm/kb` | offline ATT&CK and D3FEND knowledge base |
| `kcwm/inference`, `kcwm/ui` | offline engine, Streamlit console |
| `kcwm/pipeline.py`, `kcwm/cli.py` | `cryonex build / evaluate / campaigns / forecast / serve / report` |
| `configs/default.yaml` | windows, splits, feature groups, alert budgets, next-stage blend, seeds, datasets |
| `configs/stage_map.yaml` | label → family → stage → technique |
| `scripts/` | final run, gates, two-level policy, bundles, samples, benchmark writer |

## 12. Honest limits

* **Early warning is partial.** The two-level alerts warned 69 of 207 real test attacks before
  they began (median 5.0 min), at about 7 false early-warning windows per hour of quiet traffic.
  Most attacks in public datasets were launched on a schedule with no precursors, and nothing
  in the traffic can predict them. On the Heartbleed demo capture, false early warnings start
  at 17:43, half an hour before the exploit at 18:12.
* **New networks need a short calibration.** Ranking transfers partially (CIC-2017, CIC-2018),
  alert thresholds do not, and no model transfers to CTU-13.
* **Near-idle networks** (DAPT2020: 49% empty windows, median 1 flow) remain hard for every
  model; the DAPT2020 and CTU-13 samples raise no alerts.
* **Next-stage prediction** is right on about half of real stage changes. The host layer's
  learned forecaster does not yet generalise to unseen networks: on DARPA 2000 the textbook
  ATT&CK order, given the true history, beats it (0.90 vs 0.40).

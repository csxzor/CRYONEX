# Development history (all on dev data; the test split was read once, at the end)

This records every design decision and the evidence behind it, including the ideas that
failed. "Dev" means the P1 split with the test span untouched. Early rounds scored on a
purged tail of calibration; from round 2 on, the dev test is the full calibration split and
thresholds come from a purged tail of training. Single-seed dev AUPRC has a block-bootstrap
CI of about ±0.1, so rounds 1-3 are indicative and rounds 4-6 use 3 seeds.

The per-round result files, the dev-queue scripts, the Kaggle GPU package and the parked
host kill-chain experiment were removed from `main` to keep the submission focused; all of
them are preserved, unchanged, on the git branch `archive/pre-cleanup`.

## Findings that shaped the design

| Finding | Evidence | Decision |
|---|---|---|
| v1's "onsets" were mostly botnet beacon bursts | 321 of 437 v1 onsets on one day; median run of 2 windows | Episodes merged over 5-minute gaps; onsets need 5 quiet minutes |
| The corrected CIC-IDS2017 labels the attacker's *external* scan as lateral movement | G1 audit: first "Infiltration - Portscan" flows come from 172.16.0.1 at 14:00, 19 min before the exploit | Rule: lateral movement must originate inside the network. Thursday becomes Recon, Access, C2, Lateral |
| "Packet data present" was flagged for CSV-only datasets | CIC CSVs carry only a TCP init window, not TTL, fragments or retransmits | The packet mask requires PCAP-derived TTL |
| Pooled AUPRC mostly measures *network risk levels* | Dev half of calibration: 21 of 25 captures held only one class | Report per-capture AUPRC and use a larger dev split |
| Thresholds do not transfer across time and network | LR at a 3% budget gave 27% FPR on later traffic (global normalisation) | Warm-up (per-network) normalisation: LR FPR 1.9%, HGB AUPRC 0.692 → 0.873 |
| Real pre-compromise windows carry no signal | Early-warning AUPRC = base rate for every model, every round | Campaign synthesizer; early warning claimed on P4 only |

## Rounds

| Round | Change | CRYONEX dev AUPRC | Best control | Outcome |
|---|---|---|---|---|
| 1 | first all-dataset run, global normalisation | 0.534 | LR 0.696 | Diagnosed as a network-level offset, not discrimination |
| 1b | warm-up normalisation | 0.843 | HGB 0.873, Transformer 0.845 | Warm-up helps every model; CRYONEX ties its Transformer twin |
| 2 | + campaigns, stage metrics, regularisation, direct-risk head | 0.848 (FPR 7.4%) | HGB 0.857 | Best on P4 (next stage 52% vs Markov 14%); kept |
| 3 | early stopping on a training tail | 0.750 | Transformer 0.814 | **Rejected**: collapsed CTU-13 (0.963 → 0.871, FPR 1% → 24%) |
| 4 | + DAPT2020 (3 seeds) | 0.723 | HGB 0.82, Transformer 0.742 | DAPT's near-idle traffic (49% empty windows) hurts sequence models |
| 5 | scaling on non-empty windows; DAPT benign down-sampling (3 seeds) | 0.752 | HGB 0.822, Transformer 0.710, LSTM 0.673 | CRYONEX beats its Transformer twin; HGB still best pooled |
| 6 | **hybrid** = calibrated mean of CRYONEX and HGB (3 seeds) | **hybrid 0.832** | HGB 0.822 | Best pooled, stable across seeds (0.827 / 0.837 / 0.833): **frozen recipe** |

## Gates checked on held-out data before the final test

* **G6, unseen attack families** (leave-one-family-out, 7 families): hybrid mean AUPRC
  **0.782** vs LR 0.569, winning all 7. The world model alone averages 0.779 vs gradient
  boosting's 0.725, beating it on 6 of 7. **Pass.**
* **G7, new network:** mixed. Ranking transfers partially to CIC-2018 (hybrid 0.533 vs LR
  0.310) and CIC-2017; nothing transfers to CTU-13, and thresholds do not transfer.
  **Partial**, reported as a limit.
* **G8, faithful explanations:** deleting the top-5 attributed features drops risk by 0.776
  vs 0.015 for 5 random features (53x). **Pass.**

## Ideas tried and rejected

* Early stopping on a training tail (round 3), above.
* Label-free per-network *thresholds* on P1 sessions: P1 sessions start mid-attack, giving
  33% FPR. The policy is only valid where a new capture starts from quiet traffic.
* Alert hysteresis as the fix for FPR: it moved FPR by 0.2 points; the drift is a
  distribution shift, not flicker.
* Isotonic calibration for display: its step output (0.09, 0.33, 0.91) reads as arbitrary.
  Replaced by Platt scaling (equal Brier, lower ECE).

## After the freeze: can early warning improve? (Kaggle GPU, dev split, 3 seeds)

| Experiment | Hybrid dev AUPRC | Early-warning AUPRC (chance 0.054) | Real onsets warned (3 seeds) | Campaign onsets warned |
|---|---|---|---|---|
| frozen recipe (round 6) | 0.832 | 0.060 (world model 0.064) | world model 4/75 | 35/198 |
| E1 + long-memory features (30 min / 2 h rolling maxima) | ~0.80 | 0.080 (world model 0.073) | world model **11/75** | 32/198 |
| E2 + early-warning loss weight 5 | ~0.79 | 0.078 (0.088) | 9/75 | 1–3/198 |
| E3 + early-warning loss weight 10 | ~0.81 | 0.081 (0.089) | 8/75 | 2–3/198 |

Long memory helps warning a little on every seed, and costs about 0.03 of pooled detection;
extra early-warning weight trades campaign warnings away and is rejected. The finding that
matters: within-10-minute network windows cannot see chains whose stages are hours or days
apart (DAPT2020: one stage per day). That motivated the host kill-chain layer below.

## Host kill-chain layer (parked experiment, code on branch `archive/pre-cleanup`)

Protocol: leave-one-group-out (CTU-13 in three scenario blocks, each CIC-IDS2017 day,
CIC-IDS2018, DAPT2020, DARPA 2000 test-only). A *chain step* is a host starting an ATT&CK
stage different from the one it was most recently doing. Design choices were made on the dev
folds (CTU-13, CIC-IDS2017, CIC-IDS2018); DAPT2020 and DARPA 2000 were scored once afterwards.

| Round | Change | Dev next-stage top-1 (deployable / true history) | Outcome |
|---|---|---|---|
| v1 | nowcast on raw host counts; episodes ordered by start | 0.40 / 0.48 (lookup table 0.55) | CIC nowcast misses 10-per-10 s brute force (learned CIC-2018's 320) |
| v2 | + scale-free shares and 1/5-min host activity; previous stage = most recently active | 0.41 / **0.56** (table 0.50, ATT&CK rule 0.47; rule at the inferred state 0.42) | frozen |
| — | Bayesian transition table instead of the learned model | 0.37 / 0.52 | rejected |

Held-out, scored once (lead 1 window):

| Chain | Steps | Learned model, inferred history | Learned model, true history | ATT&CK rule, true history | Lookup table, true history |
|---|---|---|---|---|---|
| DAPT2020 (week-long APT) | 10 | 0.10 | 0.20 | 0.10 | 0.40 |
| DARPA 2000 LLDOS 1.0 + 2.0.2 | 10 | 0.10 | 0.40 | **0.90** | 0.40 |

The per-host stage nowcast recognises 15% (DAPT) and 8% (DARPA) of attack host-windows on
these unseen networks (62–86% on held-out CTU-13 blocks): a DARPA victim sees one ping and a few
RPC calls, far below anything in training. With the true history, the textbook ATT&CK order
predicts DARPA's chain almost perfectly, but the learned model, trained mostly on CTU-13 botnet
cycles, overrides it. Timing (compromise step within 24 h) has skill on dev (AUPRC 0.21 vs base
0.04) and none on the held-out chains. Bug fixed after the first held-out read (a definition,
not a model change): DARPA's two scenarios were one timeline, so March's memory leaked into
April's separate experiment; each scenario is now its own timeline, as CTU-13's are.

## G3 (learned dynamics) and the residual decoder

G3 had never been run on the frozen models. On the P1 test split their next-state predictions
beat persistence (NLL 1.84 vs 2.41 at 10 s) and linear autoregression (2.74), but lose to the
histogram of the last 64 windows (1.20): traffic features are close to stationary over
minutes. A calibrated mixture with that histogram put all its weight on the histogram, so the
decoder added nothing. Fix (the pre-registered G3 fallback, "increase decoder capacity"): the
emission became recent-history histogram x learned correction, zero-initialised, so any gain
is learned dynamics (`emit_prior=1`). Dev (seed 17): gain +0.021 [+0.015, +0.026] at 10 s,
+0.014 [+0.009, +0.018] at 1 min, tie at 5 min; risk AUPRC unchanged (hybrid 0.825 vs 0.827).

Test (a second read, 3 seeds): the gain holds at 10 s (3/3 seeds) and 1 min (2/3); hybrid
AUPRC 0.840 ± 0.020 (frozen 0.847), FPR 2.1% (frozen 4.2%). The rule set before the run (ship
if AUPRC is within ±0.02) did not consider early warning, and that regressed: 24/207 attacks
warned before they started (frozen 51/207), median lead 1.6 min (3.5), and the Heartbleed demo
gains a false "attack in progress" 27 minutes early. **Decision: the frozen model stays the
release** (early warning is the problem statement's headline); the residual decoder is reported
as a tested variant (BENCHMARKS section 7) and kept at artifacts/kcwm-residual-decoder.pt.

## Early-warning budget and next-stage table (post-hoc, after the test read)

* **Next stage.** The rollout's stage mass got 16% of real dev stage changes right; a transition
  table P(next | current, progress) counted on training labels and read at the model's own
  current stage got 37–39%, and rollout^0.25 x table^0.75 got 40–43% (chosen on dev, frozen dev
  models, 3 seeds). One test read: 29% -> **50%** (1 window before) and 29% -> 49% (1 min
  before); the table at the true current stage, an upper bound, is 58%.
* **Early-warning budget** 3% -> 5% of quiet benign calibration windows: 51 -> 69 of 207 real
  attacks warned before they started, median lead 3.5 -> 5.0 min, false early warnings 1.5% ->
  3.2% of quiet test time. The sweep (10%, 20%) shows false warnings exploding beyond that.
  On the Heartbleed demo capture the 5% threshold gives ~17 minutes of false early warnings
  before the real one (3%: ~4 minutes).

## External review, phase 1 (documentation only; no compute, test split not read)

Rule from here on: the P1 test split has been read three times (the frozen model; the residual
decoder, a second read; the post-hoc alert policy and next-stage table, scored on saved test
scores). It is treated as burned. No further tuning, selection or re-runs against it; every
later design choice uses dev data only, with pass/fail criteria written here before running.

Phase 1 changes the README and docs only, and is complete when (criteria): every number it adds
matches `results/` exactly; no headline number changes; `make benchmarks` leaves
`docs/BENCHMARKS.md` unchanged; the tests pass. Numbers added, all from existing results:

* Controls in the README table: logistic regression over the last 6 windows (F1 0.614, FPR 1.7%,
  AUPRC 0.726) and the Transformer classifier with the same backbone and no dynamics (AUPRC 0.815
  vs 0.820 for the world model alone; **2 seeds**, not 3: seed 29 did not complete).
* Learned dynamics, stated in full: the released decoder beats persistence and autoregression but
  loses to the recent-history histogram (NLL 1.835 vs 1.202 at 10 s; 0 of 3 seeds better).
* Next stage, decomposed: rollout alone 0.295; plain Markov table at the true current stage 0.460
  (stage report); learned table at the model's own current stage 0.496; deployed blend 0.504.
  The deployed number is mostly the table.
* Early warning: 69/207 is post-hoc (the budget moved from 3% to 5% after the test read).
  Dev vs test is unreconciled: the world model's early-warning AUPRC was 0.064 on dev (hybrid
  0.060, chance 0.054) but 0.225 on test (hybrid 0.183).

Correction: the earlier entry "~17 minutes of false early warnings" on the Heartbleed capture
measured a span (17:43 to 18:00), not warning time. Measured on the release bundle (5% budget):
first early warning 17:43:10; 13.5 minutes of early-warning windows before the exploit's first
labelled window (18:12:10), of which 3.0 minutes fall in the 5 minutes before it and 10.5
minutes are false.

## External review, phase 2 (tooling; no compute)

`scripts/check_readme_numbers.py` parses the README's results tables and headline sentences and
fails if any number differs from `results/` at the precision the README prints (73 numbers). It
runs in CI and at the end of `make checks`; `make readme-check` runs it alone. Criteria: passes on
the current README; fails on planted errors. Result: pass (0 mismatches); six planted errors (a
detection metric, a sweep count, the next-stage blend, the faithfulness ratio, the dev chance
rate, a deleted table row) were each caught.

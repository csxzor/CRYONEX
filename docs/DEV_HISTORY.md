# Development history (all on dev data; the test split was read once, at the end)

This records every design decision and the evidence behind it, including the ideas that
failed. "Dev" means the P1 split with the test span untouched. Early rounds scored on a
purged tail of calibration; from round 2 on, the dev test is the full calibration split and
thresholds come from a purged tail of training. Single-seed dev AUPRC has a block-bootstrap
CI of about ±0.1, so rounds 1-3 are indicative and rounds 4-6 use 3 seeds.

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

| Round | Change | KC-WM dev AUPRC | Best control | Outcome |
|---|---|---|---|---|
| 1 | first all-dataset run, global normalisation | 0.534 | LR 0.696 | Diagnosed as a network-level offset, not discrimination |
| 1b | warm-up normalisation | 0.843 | HGB 0.873, Transformer 0.845 | Warm-up helps every model; KC-WM ties its Transformer twin |
| 2 | + campaigns, stage metrics, regularisation, direct-risk head | 0.848 (FPR 7.4%) | HGB 0.857 | Best on P4 (next stage 52% vs Markov 14%); kept |
| 3 | early stopping on a training tail | 0.750 | Transformer 0.814 | **Rejected**: collapsed CTU-13 (0.963 → 0.871, FPR 1% → 24%) |
| 4 | + DAPT2020 (3 seeds) | 0.723 | HGB 0.82, Transformer 0.742 | DAPT's near-idle traffic (49% empty windows) hurts sequence models |
| 5 | scaling on non-empty windows; DAPT benign down-sampling (3 seeds) | 0.752 | HGB 0.822, Transformer 0.710, LSTM 0.673 | KC-WM beats its Transformer twin; HGB still best pooled |
| 6 | **hybrid** = calibrated mean of KC-WM and HGB (3 seeds) | **hybrid 0.832** | HGB 0.822 | Best pooled, stable across seeds (0.827 / 0.837 / 0.833): **frozen recipe** |

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

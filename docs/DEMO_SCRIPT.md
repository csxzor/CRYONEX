# Demo video script (2:00)

Record from `kcwm serve` with the release bundle (`artifacts/release/kcwm.pt`), normalisation
`warmup`. Both samples are real CIC-IDS2017 traffic from **outside the model's training
period**. Every number said aloud is on screen or in `docs/BENCHMARKS.md`.

**Scene 1:** `samples/cic17_wednesday_heartbleed.csv.gz`, internal network `192.168.10.0/24`.

| Time | Screen | Narration |
|---|---|---|
| 0:00–0:15 | Console title | "Intrusion detectors look at one moment at a time. An attack is a process. KC-WM is a world model: it learns how a network evolves and simulates its next five minutes, to warn *before* compromise." |
| 0:15–0:30 | Metrics row; risk timeline | "This is 70 minutes of real traffic the model never trained on, parsed offline on a laptop." |
| 0:30–0:40 | Slider to 17:45 (amber) | "At 17:45 there is an early warning with no attack: a false alarm. We show it because we measure false alarms, about 3 an hour on this traffic." |
| 0:40–1:05 | Slider to 18:09 (amber), 18:11 (red), 18:12 (ground-truth marker) | "At 18:09 the world model's simulated futures converge on compromise: **early warning**. At 18:11, **attack in progress**. The Heartbleed exploit starts at 18:12, three minutes after the first warning." |
| 1:05–1:25 | "Why" table; attention bars | "Every alert explains itself: the feature groups driving it and the moments that mattered. It is faithful: deleting the top five features cuts the risk 500 times more than deleting random ones." |
| 1:25–1:40 | "Who"; "Respond": ATT&CK T1190 and D3FEND cards | "Re-simulating without each host shows who drives the forecast; the stage maps to ATT&CK and D3FEND countermeasures, offline." |

**Scene 2** (1:40–2:00): the benchmark table, `docs/BENCHMARKS.md` §1.

"On held-out test data: AUPRC 0.85 vs 0.65 for the required logistic regression. The
two-level alerts warned a quarter of real attacks before they began, about three and a
half minutes ahead, where standard detectors warned almost none. On unseen attack
families: 0.78 vs 0.57."

Backup take: `samples/cic17_friday_scan_ddos.csv.gz`. The botnet is active all afternoon and
the DDoS starts at 18:56. The in-progress alert fires immediately; the world-model forecast
rises in the minutes before but stays under the early-warning threshold, so do not claim an
early warning there.

**Do not claim:**
* that it warns before *most* real attacks. It warns before about 1 in 4; most attacks in
  public datasets have no precursors;
* alerts on the DAPT2020 or CTU-13 samples. The model stays below threshold there, a stated
  limit.

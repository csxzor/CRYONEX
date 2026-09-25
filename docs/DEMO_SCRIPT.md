# Demo video script (2:00)

Record from `kcwm serve` with the release bundle, on
`samples/cic17_friday_scan_ddos.csv.gz` (internal network `192.168.10.0/24`,
normalisation `warmup`). This is real CIC-IDS2017 traffic from outside the model's training
period. Every number said aloud is on screen or in `docs/BENCHMARKS.md`.

| Time | Screen | Narration |
|---|---|---|
| 0:00–0:15 | Console title; sidebar with the sample selected | "Intrusion detectors look at one moment at a time. An attack is a process. KC-WM is a world model: it learns how a network's state evolves and simulates its next five minutes." |
| 0:15–0:35 | Metrics row (334k flows, 720 windows); risk timeline | "This is two hours of real traffic, parsed offline on a laptop. A botnet is beaconing all afternoon, so the risk never drops to zero. That is correct: command-and-control is already a compromise." |
| 0:35–1:00 | Slider to 18:51; forecast curve; kill-chain chips | "At 18:51 the model's simulated futures start converging on Impact. The world-model forecast rises from 0.47 to 0.80 in the minutes before the DDoS, which begins at 18:56. From there every window alerts, and the stage reads Impact." |
| 1:00–1:20 | "Why" table; attention bars | "Every alert explains itself: the feature groups driving it, in native units, and the past windows that mattered. The explanations are faithful: deleting the top five features cuts the risk 500 times more than deleting five random ones." |
| 1:20–1:35 | "Who": host attribution; flagged flows | "Removing each host's recent flows and re-simulating shows who is driving the forecast, down to the flows." |
| 1:35–1:50 | "Respond": ATT&CK T1498 / D3FEND cards | "The predicted stage maps to ATT&CK techniques and D3FEND countermeasures, all offline." |
| 1:50–2:00 | Benchmark table (docs/BENCHMARKS.md §1, §4) | "On held-out test data, AUPRC is 0.85 against 0.65 for the required logistic regression, and it generalises to unseen attack families (0.78 vs 0.57)." |

Backup take: `samples/ctu13_s04_c2_ddos.binetflow.gz` (internal `147.32.0.0/16`) shows the
same flow on a botnet network. Its risk peaks just below the alert threshold, so do not claim
alerts on it.

Do not claim: early warning before *real* attacks in general (only this DDoS ramp, and P4
campaigns), or alerts on the DAPT2020 samples (the model stays below threshold there; that is
a stated limit).

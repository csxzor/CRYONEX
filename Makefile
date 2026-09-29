PY := .venv/bin/python
K := .venv/bin/cryonex
DATASETS := cicids2017 cicids2018 ctu13 dapt2020

.PHONY: setup test lint demo serve build samples campaigns audit final checks benchmarks readme-check kb

setup:            ## create the environment (Python 3.12, CPU PyTorch)
	uv sync --all-extras

test:             ## unit + integration tests (no datasets needed)
	$(PY) -m pytest -q

lint:             ## code-quality check
	.venv/bin/ruff check kcwm tests scripts

demo:             ## offline forecast on the bundled Heartbleed sample
	$(K) forecast samples/cic17_wednesday_heartbleed.csv.gz --internal 192.168.10.0/24

serve:            ## dashboard on http://localhost:8501
	$(K) serve

build:            ## raw datasets -> flows + windows (data/processed); needs the datasets
	$(K) build $(DATASETS)

campaigns:        ## synthetic kill-chain campaigns from real traffic (training augmentation, P4)
	$(K) campaigns $(DATASETS)

audit:            ## gate G1: labels vs the published attack schedule -> docs/LABEL_AUDIT.md
	$(PY) scripts/label_audit.py

final:            ## train the frozen models and read the P1 test split once (hours on CPU)
	scripts/run_final.sh

checks:           ## every other benchmark section: G3, G6, G7, G8, alerts, next stage
	scripts/run_checks.sh
	$(PY) scripts/check_readme_numbers.py

benchmarks:       ## regenerate docs/BENCHMARKS.md from results/
	$(PY) scripts/write_benchmarks.py > /dev/null

readme-check:     ## fail if a README headline number differs from results/ (fast, no data)
	$(PY) scripts/check_readme_numbers.py

samples:          ## cut the demo captures from the raw datasets
	$(PY) scripts/make_samples.py

kb:               ## rebuild the offline ATT&CK / D3FEND subset (needs the network once)
	$(PY) scripts/fetch_kb.py

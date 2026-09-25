PY := .venv/bin/python
K := .venv/bin/kcwm
DATASETS := cicids2017 cicids2018 ctu13

.PHONY: setup test lint build retarget audit dev eval report demo serve samples kb

setup:            ## create the environment (Python 3.12, CPU torch)
	uv sync --all-extras

test:             ## unit + integration tests (no datasets needed)
	$(PY) -m pytest -q

lint:
	.venv/bin/ruff check kcwm tests scripts

build:            ## raw datasets -> flows + windows (data/processed)
	$(K) build $(DATASETS)

retarget:         ## recompute episodes/targets without re-reading raw data
	$(K) retarget

audit:            ## gate G1: labels vs published schedule -> docs/LABEL_AUDIT.md
	$(PY) scripts/label_audit.py

dev:              ## development runs (never touch the test split)
	scripts/queue_dev_p1.sh

report:
	$(K) report p1-all

samples:          ## cut demo captures from the raw datasets
	$(PY) scripts/make_samples.py

kb:               ## rebuild the offline ATT&CK/D3FEND subset (needs network once)
	$(PY) scripts/fetch_kb.py

demo:             ## offline forecast on the bundled sample
	$(K) forecast samples/cic17_thursday_infiltration.csv.gz --internal 192.168.10.0/24

serve:            ## operator console on http://localhost:8501
	$(K) serve

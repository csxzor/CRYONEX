#!/usr/bin/env bash
# Dev round 6: the hybrid (world model + gradient boosting), 3 seeds, round-5 data recipe.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
REG="epochs=15,anchors_per_epoch=8000,patience=4,d=96,layers=3,dropout=0.2,weight_decay=0.05,direct_risk=1"
.venv/bin/kcwm evaluate --dev --datasets cicids2017,cicids2018,ctu13,dapt2020 --mode warmup --campaigns \
  --tag r6 --models hybrid --seeds 17,23,29 --wm "$REG" > runs/r6.log 2>&1
echo done > runs/r6.done

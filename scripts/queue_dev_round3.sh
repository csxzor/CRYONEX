#!/usr/bin/env bash
# Dev round 3: early stopping separated from threshold calibration.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
K=.venv/bin/kcwm
DS=${DS:-cicids2017,cicids2018,ctu13}
COMMON="--dev --datasets $DS --mode warmup --campaigns"
REG="epochs=15,anchors_per_epoch=8000,patience=4,d=96,layers=3,dropout=0.2,weight_decay=0.05,direct_risk=1"
$K evaluate $COMMON --tag r3-kcwm --models kcwm --wm "$REG" > runs/r3_kcwm.log 2>&1
$K evaluate $COMMON --tag r3-transformer --models transformer_clf > runs/r3_tf.log 2>&1
echo done > runs/r3.done

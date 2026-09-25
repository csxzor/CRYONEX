#!/usr/bin/env bash
# Dev round 5: round 4 + empty-window-free normalisation and DAPT benign down-sampling.
# recipe vs controls; then G7 cross-network and G6 leave-one-family-out (dev, 1 seed).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
K=.venv/bin/kcwm
DS=cicids2017,cicids2018,ctu13,dapt2020
SEEDS=17,23,29
REG="epochs=15,anchors_per_epoch=8000,patience=4,d=96,layers=3,dropout=0.2,weight_decay=0.05,direct_risk=1"
COMMON="--dev --datasets $DS --mode warmup --campaigns"

$K evaluate $COMMON --tag r5 --models logreg,logreg_stack,hgb --seeds $SEEDS > runs/r5_tab.log 2>&1
$K evaluate $COMMON --tag r5 --models kcwm --seeds $SEEDS --wm "$REG" > runs/r5_kcwm.log 2>&1
$K evaluate $COMMON --tag r5 --models transformer_clf,lstm --seeds $SEEDS > runs/r5_nn.log 2>&1
echo done > runs/r5.done

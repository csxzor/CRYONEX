#!/usr/bin/env bash
# Dev round 2 (never reads the test split): regularisation and fallback F-A, with G5 stage metrics.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
K=.venv/bin/kcwm
DS=cicids2017,cicids2018,ctu13
COMMON="--dev --datasets $DS --mode warmup --campaigns"
BASE="epochs=15,anchors_per_epoch=8000,patience=4"
REG="$BASE,d=96,layers=3,dropout=0.2,weight_decay=0.05"

$K evaluate $COMMON --tag r2-tabular --models logreg,logreg_stack,hgb > runs/r2_tabular.log 2>&1
$K evaluate $COMMON --tag r2-kcwm-default --models kcwm --wm "$BASE" > runs/r2_default.log 2>&1
$K evaluate $COMMON --tag r2-kcwm-reg --models kcwm --wm "$REG" > runs/r2_reg.log 2>&1
$K evaluate $COMMON --tag r2-kcwm-reg-fa --models kcwm --wm "$REG,direct_risk=1" > runs/r2_reg_fa.log 2>&1
echo done > runs/r2.done

#!/usr/bin/env bash
# FINAL: the one-time read of the P1 test split, frozen recipe (chosen on dev rounds 1-6).
#   data: CIC-IDS2017 + CSE-CIC-IDS2018 + CTU-13 + DAPT2020, warm-up normalisation,
#         empty-window-free scaling, DAPT benign down-sampling, training campaigns
#   world model: d=96, 3 layers, dropout 0.2, weight decay 0.05, direct-risk head (F-A)
#   deployed score: hybrid = mean of Platt-calibrated world model and gradient boosting
# Do not re-run with a changed recipe and report the better number: that would be tuning on test.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
K=.venv/bin/kcwm
DS=cicids2017,cicids2018,ctu13,dapt2020
SEEDS=17,23,29
REG="epochs=15,anchors_per_epoch=8000,patience=4,d=96,layers=3,dropout=0.2,weight_decay=0.05,direct_risk=1"
COMMON="--protocol p1 --datasets $DS --mode warmup --campaigns --seeds $SEEDS"
$K evaluate $COMMON --tag final-p1 --models logreg,logreg_stack,hgb,kcwm,hybrid --wm "$REG" > runs/final_main.log 2>&1
echo main > runs/final_main.done
$K evaluate $COMMON --tag final-p1 --models lstm,transformer_clf > runs/final_nn.log 2>&1
echo done > runs/final.done

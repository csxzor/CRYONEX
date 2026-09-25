#!/usr/bin/env bash
# Development queue (never reads the test split): P1 over all built datasets, dev mode.
# Sequential so jobs don't fight over CPU threads. Logs in runs/dev_*.log.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
K=.venv/bin/kcwm
DS=cicids2017,cicids2018,ctu13
WM=${WM:-epochs=15,anchors_per_epoch=8000,patience=4}
JOBS=${JOBS:-"tab-global tab-warmup wm-warmup wm-global nn-warmup"}

for job in $JOBS; do
  case $job in
    tab-global) $K evaluate --dev --tag p1-all-global --datasets $DS --models logreg,logreg_stack,hgb --mode global ;;
    tab-warmup) $K evaluate --dev --tag p1-all-warmup --datasets $DS --models logreg,logreg_stack,hgb --mode warmup ;;
    wm-global)  $K evaluate --dev --tag p1-all-global --datasets $DS --models kcwm --mode global --wm "$WM" ;;
    wm-warmup)  $K evaluate --dev --tag p1-all-warmup --datasets $DS --models kcwm --mode warmup --wm "$WM" ;;
    nn-global)  $K evaluate --dev --tag p1-all-global --datasets $DS --models lstm,transformer_clf --mode global ;;
    nn-warmup)  $K evaluate --dev --tag p1-all-warmup --datasets $DS --models lstm,transformer_clf --mode warmup ;;
    wm-camp)    $K evaluate --dev --tag p1-all-warmup-camp --datasets $DS --models kcwm --mode warmup --campaigns --wm "$WM" ;;
    tab-camp)   $K evaluate --dev --tag p1-all-warmup-camp --datasets $DS --models logreg,logreg_stack,hgb --mode warmup --campaigns ;;
    nn-camp)    $K evaluate --dev --tag p1-all-warmup-camp --datasets $DS --models lstm,transformer_clf --mode warmup --campaigns ;;
  esac > "runs/dev_$job.log" 2>&1
done
echo done > runs/dev_queue.done

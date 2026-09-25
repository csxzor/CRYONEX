#!/usr/bin/env bash
# Gates G7 (cross-network) and G6 (leave-one-family-out), dev scoring rules, 1 seed.
# Hybrid = world model + gradient boosting; logistic regression as the required reference.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
K=.venv/bin/kcwm
REG="epochs=15,anchors_per_epoch=8000,patience=4,d=96,layers=3,dropout=0.2,weight_decay=0.05,direct_risk=1"
ALL=(cicids2017 cicids2018 ctu13 dapt2020)
for tgt in cicids2018 cicids2017 ctu13; do
  src=$(printf "%s," "${ALL[@]}" | sed "s/$tgt,//; s/,$//")
  for mode in warmup global; do
    $K evaluate --protocol cross --tag g7-$tgt-$mode --datasets $(printf "%s," "${ALL[@]}" | sed 's/,$//') \
      --train-datasets $src --test-dataset $tgt --mode $mode --models logreg,hybrid --seeds 17 --wm "$REG" \
      > runs/g7_${tgt}_${mode}.log 2>&1
  done
done
for fam in PortScan BruteForce WebAttack DoS DDoS Botnet Infiltration; do
  $K evaluate --protocol lofo --family $fam --tag g6-$fam --datasets cicids2017,cicids2018,ctu13,dapt2020 \
    --mode warmup --models logreg,hybrid --seeds 17 --wm "$REG" > runs/g6_$fam.log 2>&1
done
echo done > runs/g6g7.done

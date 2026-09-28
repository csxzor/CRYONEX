#!/usr/bin/env bash
# Reproduce every docs/BENCHMARKS.md section beyond the main P1 table. Run after
# scripts/run_final.sh (which trains the frozen models and reads the P1 test split once).
#
#   G6  unseen attack families (leave one family out)        -> results/g6-<family>/
#   G7  a new network (train on three datasets, test on one) -> results/g7-<dataset>-<mode>/
#   G8  explanation faithfulness                              -> results/g8_faithfulness.json
#   two-level alerts and the early-warning budget sweep       -> results/final-p1/two_level*.json
#   next-stage transition table (chosen on dev, read on test) -> results/{dev-r6,final-p1}/next_stage.json
#   G3  learned dynamics, frozen decoder and residual variant -> results/final-p1{,-g3}/g3_dynamics.json
#
# Needs the processed datasets (see docs/DATASETS.md). CPU: several hours in total.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
K=.venv/bin/cryonex
PY=.venv/bin/python
DS=cicids2017,cicids2018,ctu13,dapt2020
ALL=(cicids2017 cicids2018 ctu13 dapt2020)
REG="epochs=15,anchors_per_epoch=8000,patience=4,d=96,layers=3,dropout=0.2,weight_decay=0.05,direct_risk=1"
mkdir -p runs

# G6: every family removed from training, including campaign snippets, then tested
for fam in PortScan BruteForce WebAttack DoS DDoS Botnet Infiltration; do
  $K evaluate --protocol lofo --family $fam --tag g6-$fam --datasets $DS \
    --mode warmup --models logreg,hybrid --seeds 17 --wm "$REG" > runs/g6_$fam.log 2>&1
done

# G7: train on three networks, test on every window of the fourth, both normalisations
for tgt in cicids2018 cicids2017 ctu13; do
  src=$(printf "%s," "${ALL[@]}" | sed "s/$tgt,//; s/,$//")
  for mode in warmup global; do
    $K evaluate --protocol cross --tag g7-$tgt-$mode --datasets $DS --train-datasets $src \
      --test-dataset $tgt --mode $mode --models logreg,hybrid --seeds 17 --wm "$REG" > runs/g7_${tgt}_${mode}.log 2>&1
  done
done

# G8: faithfulness of the explanations (calibration split of the frozen seed-17 model)
$PY scripts/faithfulness.py runs/final-p1/kcwm-s17.pt 40

# Two-level alerts: early-warning budget sweep, then the deployed budget (configs/default.yaml)
for b in 0.03 0.05 0.1 0.2; do
  $PY -c "import sys; sys.path.insert(0, 'scripts'); from two_level import main; main('results/final-p1', $b, out_name='two_level_sweep_$b.json')"
done
$PY scripts/two_level.py results/final-p1

# Next stage: the dev models choose the blend weight, then the test split is read once
$K evaluate --dev --protocol p1 --datasets $DS --mode warmup --campaigns --seeds 17,23,29 \
  --models hgb,kcwm,hybrid --wm "$REG" --tag r6 > runs/dev_r6.log 2>&1
$PY scripts/next_stage_eval.py results/dev-r6
$PY scripts/next_stage_eval.py results/final-p1

# G3: next-state prediction of the frozen decoder, then the residual-decoder variant
# (a second read of the test split; reported as such, not released)
$PY scripts/g3_dynamics.py results/final-p1
$K evaluate --protocol p1 --datasets $DS --mode warmup --campaigns --seeds 17,23,29 \
  --models hgb,kcwm,hybrid --wm "$REG,emit_prior=1" --tag final-p1-g3 > runs/final_p1_g3.log 2>&1
$PY scripts/g3_dynamics.py results/final-p1-g3
$PY scripts/two_level.py results/final-p1-g3

$PY scripts/write_benchmarks.py > /dev/null
echo "done: docs/BENCHMARKS.md regenerated"

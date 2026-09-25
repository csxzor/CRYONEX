"""Metric definitions: lead time, hysteresis, block bootstrap, conformal budget."""

import numpy as np

from kcwm.calibrate.conformal import calibrate
from kcwm.eval.metrics import (
    block_bootstrap,
    classification_metrics,
    lead_time,
    sustained,
    time_blocks,
)
from kcwm.eval.protocols import compromise_onsets


def test_sustained_respects_sessions():
    alert = np.array([1, 1, 1, 0, 1, 1], dtype=bool)
    sess = np.array([0, 0, 1, 1, 1, 1])
    assert sustained(alert, sess, 2).tolist() == [False, True, False, False, False, True]


def test_lead_time_counts_only_unbroken_runs_before_onset():
    score = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0], dtype=float)
    sess = np.zeros(score.size)
    lt = lead_time(score, np.array([5, 10]), sess, threshold=0.5, max_lookback=30, hysteresis=2, grace=0)
    # onset 5: alert run 2..4 sustained from 3 -> lead 2; onset 10: run 7..9 sustained 8..9 -> lead 2
    assert [p["lead_windows"] for p in lt.per_onset] == [2, 2]
    assert lt.warned_before == 2


def test_compromise_onsets_ignore_recon_and_need_quiet():
    stage = np.array([0, 1, 1, 2, 2, 0, 2, 0, 0, 0, 0, 6])
    on = compromise_onsets(stage, np.zeros(stage.size), quiet=3)
    assert on.tolist() == [False, False, False, True, False, False, False, False, False, False, False, True]


def test_conformal_threshold_meets_budget_on_negatives(rng):
    s = rng.random(5000)
    y = rng.random(5000) < 0.1
    thr = calibrate(s, y, target_fpr=0.03).threshold
    assert (s[~y] >= thr).mean() <= 0.031


def test_block_bootstrap_ci_brackets_point_estimate(rng):
    y = rng.random(2000) < 0.3
    s = y * 0.6 + rng.random(2000) * 0.5
    blocks = time_blocks(np.zeros(2000), np.arange(2000), block=100)
    from sklearn.metrics import average_precision_score

    lo, hi = block_bootstrap(average_precision_score, y, s, blocks, n_boot=200)
    point = average_precision_score(y, s)
    assert lo <= point <= hi and hi - lo < 0.2
    m = classification_metrics(y, s, threshold=0.6)
    assert 0 <= m.fpr <= 1 and m.positives == int(y.sum())

"""Episodes, onsets and forecast targets: the definitions the whole benchmark rests on."""

import numpy as np

from kcwm.targets.episodes import episodes, stage_onsets
from kcwm.targets.targets import forecast_targets


def test_beacon_bursts_are_one_episode_not_many_onsets():
    # quiet(10) then bursts every 3 windows: v1 would count 4 onsets; v2 counts 1 + 3 recurrences.
    att = np.array([0] * 10 + [1, 0, 0, 1, 0, 0, 1, 0, 0, 1] + [0] * 10, dtype=bool)
    ep = episodes(att, np.zeros(att.size), merge_gap=3, onset_quiet=5)
    assert ep["is_onset"].sum() == 1 and ep["is_onset"][10]
    assert ep["is_recurrence"].sum() == 3
    assert len(set(ep["episode"][ep["episode"] >= 0])) == 1


def test_onset_requires_quiet_and_never_at_session_start():
    att = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1], dtype=bool)
    ep = episodes(att, np.zeros(att.size), merge_gap=1, onset_quiet=5)
    # episode at t=0 has nothing observable before it; t=4 follows only 2 quiet windows.
    assert not ep["is_onset"][0] and not ep["is_onset"][4]
    assert ep["is_onset"][11]


def test_episodes_do_not_cross_sessions():
    att = np.array([0, 0, 1, 1, 1, 0, 0, 1], dtype=bool)
    sess = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    ep = episodes(att, sess, merge_gap=1, onset_quiet=2)
    assert ep["episode"][3] != ep["episode"][4]
    # t=4 opens session 1 already under attack (no observable quiet before it); t=7 follows
    # 2 quiet windows of session 1, not session 0's tail.
    assert ep["is_onset"][2] and not ep["is_onset"][4] and ep["is_onset"][7]


def test_stage_onsets_catch_transitions_inside_an_episode():
    stage = np.array([0, 0, 4, 4, 4, 1, 1, 4, 6, 6], dtype=np.int64)
    on = stage_onsets(stage, np.zeros(stage.size), quiet=3)
    assert on.tolist() == [False, False, True, False, False, True, False, False, True, False]


def test_forecast_targets_exclude_now_and_mask_session_end():
    # compromise (InitialAccess=2) at t=5 in session 0 (length 8); session 1 all benign.
    stage = np.array([0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0], dtype=np.int64)
    sess = np.array([0] * 8 + [1] * 3)
    t = forecast_targets(stage, sess, [3])
    assert t["y_3"].tolist()[:8] == [False, False, True, True, True, False, False, False]
    # the current window is excluded: t=5 itself is not a positive
    assert not t["y_3"][5]
    # horizon past the session end is masked, not negative
    assert t["valid_3"].tolist() == [True] * 5 + [False] * 3 + [False] * 3
    assert t["time_to_compromise"][2] == 3 and t["time_to_compromise"][6] == -1


def test_recon_alone_is_not_infiltration():
    stage = np.array([0, 0, 1, 1, 0, 0, 0, 0], dtype=np.int64)  # recon only
    t = forecast_targets(stage, np.zeros(stage.size), [3])
    assert not t["y_3"].any()
    assert t["next_attack_stage"][0] == 1

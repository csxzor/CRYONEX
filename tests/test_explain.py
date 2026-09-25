"""Explanations: IG completeness, attention is a distribution, counterfactual/deletion run."""

import numpy as np
import torch

from kcwm.explain.attributions import attention_rollout, integrated_gradients, temporal_occlusion
from kcwm.explain.counterfactual import deletion_test, group_counterfactual
from kcwm.features.registry import N_FEATURES
from kcwm.model.world_model import KillChainWorldModel

M_CH = 3


def _model():
    torch.manual_seed(0)
    m = KillChainWorldModel(N_FEATURES, M_CH, n_bins=np.full(N_FEATURES, 8), d=32, layers=2,
                            heads=2, ff=64, max_len=16, dropout=0.0)
    with torch.no_grad():
        m.ctx_trans.weight.normal_(0, 0.5)
    return m.eval()


def _x(seed=1):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(16, N_FEATURES + M_CH)).astype(np.float32)
    x[:, N_FEATURES:] = 1.0
    return x


def test_integrated_gradients_completeness():
    m, x = _model(), _x()
    att = integrated_gradients(m, x, horizon=6, n_features=N_FEATURES, steps=128)
    total = att.values.sum()
    assert abs(total - (att.risk - att.baseline_risk)) < 0.05 * max(abs(att.risk - att.baseline_risk), 1e-3) + 1e-3
    assert np.isclose(sum(att.per_group.values()), att.per_feature.sum())


def test_attention_rollout_is_a_distribution_over_the_past():
    a = attention_rollout(_model(), _x())
    assert a.shape == (16,) and np.isclose(a.sum(), 1.0, atol=1e-5) and (a >= 0).all()


def test_occlusion_and_counterfactual_run():
    m, x = _model(), _x()
    occ = temporal_occlusion(m, x, horizon=6, n_features=N_FEATURES, block=4)
    assert occ.shape == (16,)
    cf = group_counterfactual(m, x, horizon=6, n_features=N_FEATURES, threshold=0.0, recent=4, max_groups=2)
    assert len(cf["steps"]) == 2 and not cf["crosses_threshold"]


def test_deletion_test_reports_ratio():
    out = deletion_test(_model(), [_x(i) for i in range(3)], horizon=6, n_features=N_FEATURES, k=5)
    assert out["n"] == 3 and "passes_G8" in out

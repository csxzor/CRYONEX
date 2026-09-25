"""World-model rollout: probability axioms and agreement between the two estimators."""

import numpy as np
import torch

from kcwm.model.rollout import analytic, monte_carlo, one_step_nll
from kcwm.model.world_model import KillChainWorldModel, S, prior_transition

F, M = 10, 3


def _model(seed=0):
    torch.manual_seed(seed)
    m = KillChainWorldModel(F, M, n_bins=np.full(F, 8), d=32, layers=1, heads=2, ff=64, max_len=16)
    m.eval()
    # make the context term matter so the test is not trivially the prior
    with torch.no_grad():
        m.ctx_trans.weight.normal_(0, 0.3)
    return m


def test_prior_is_row_stochastic_and_forward_leaning():
    a = prior_transition()
    np.testing.assert_allclose(a.sum(1), 1.0, atol=1e-9)
    assert (np.diag(a) > 0.8).all()
    # from Recon, Initial Access is likelier than jumping straight to Exfiltration
    assert a[1, 2] > a[1, 5]


def test_analytic_axioms():
    m = _model()
    B = 5
    h0 = torch.randn(B, 32)
    b0 = torch.softmax(torch.randn(B, S), -1)
    p0 = torch.randint(0, S, (B,))
    with torch.no_grad():
        fc = analytic(m, h0, b0, p0, 20)
    p = fc.p_infil.numpy()
    assert ((p >= 0) & (p <= 1)).all()
    assert (np.diff(p, axis=1) >= -1e-6).all(), "P(reach by k) must be monotone in k"
    np.testing.assert_allclose(fc.stage_marg.sum(-1).numpy(), 1.0, atol=1e-5)
    np.testing.assert_allclose(fc.progress_marg.sum(-1).numpy(), 1.0, atol=1e-5)
    # progress never decreases in expectation of its index
    idx = torch.arange(S).float()
    ep = (fc.progress_marg * idx).sum(-1).numpy()
    assert (np.diff(ep, axis=1) >= -1e-5).all()


def test_monte_carlo_matches_analytic_with_mean_field_latent():
    m = _model(1)
    h0 = torch.randn(3, 32)
    b0 = torch.softmax(torch.randn(3, S), -1)
    p0 = torch.tensor([0, 1, 2])
    with torch.no_grad():
        fc = analytic(m, h0, b0, p0, 12)
    g = torch.Generator().manual_seed(0)
    tr = monte_carlo(m, h0, b0, p0, 12, samples=6000, generator=g, mean_field_latent=True)
    assert np.abs(fc.p_infil.numpy() - tr.p_infil.numpy()).max() < 0.03


def test_current_compromise_does_not_count_as_a_future_hit():
    m = _model(2)
    with torch.no_grad():
        # a model that always returns to Benign immediately
        m.logA.fill_(-20.0)
        m.logA[:, 0] = 0.0
        m.logB.zero_()
        m.ctx_trans.weight.zero_()
        m.ctx_trans.bias.zero_()
        fc = analytic(m, torch.zeros(1, 32), torch.nn.functional.one_hot(torch.tensor([4]), S).float(),
                      torch.tensor([4]), 5)
    assert fc.p_infil.max().item() < 1e-6


def test_surprise_is_finite_and_mask_aware():
    m = _model(3)
    h = torch.randn(4, 32)
    bins = torch.randint(0, 8, (4, F))
    mask = torch.ones(4, F)
    mask[0, :5] = 0
    with torch.no_grad():
        u = one_step_nll(m, h, torch.zeros(4, dtype=torch.long), torch.zeros(4, dtype=torch.long), bins, mask)
    assert torch.isfinite(u).all() and (u > 0).all()

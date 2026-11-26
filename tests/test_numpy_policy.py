"""ballrain.train.ppo.NumpyPolicy (the worker's per-step policy): the same logits / value / head choice as
ActorCritic up to float32 summation order, and sampling that follows the softmax."""
import numpy as np
import torch

from ballrain.train.ppo import ActorCritic, NumpyPolicy


def _model(heads):
    torch.manual_seed(0)
    m = ActorCritic(12, 9, (32, 32), "tanh", heads)
    for p in m.parameters():
        torch.nn.init.normal_(p, 0, 0.3)
    return m.eval()


def test_matches_torch_for_both_head_modes():
    rng = np.random.default_rng(0)
    for heads in ("single", "ball_type"):
        m = _model(heads)
        X = rng.normal(size=(300, 12)).astype(np.float32)
        X[:100, 1:4] = -1.0  # no ball-type bit set: head 0
        P = NumpyPolicy(m)
        with torch.no_grad():
            L, V = m(torch.as_tensor(X))
        Ln = np.stack([P.logits(x) for x in X])
        Vn = np.array([P.value(x) for x in X])
        assert np.abs(Ln - L.numpy()).max() < 1e-4
        assert np.abs(Vn - V.numpy()).max() < 1e-4


def test_sampling_follows_the_softmax_and_logp_is_consistent():
    m = _model("ball_type")
    x = np.random.default_rng(1).normal(size=12).astype(np.float32)
    P = NumpyPolicy(m)
    with torch.no_grad():
        probs = torch.softmax(m(torch.as_tensor(x[None]))[0][0], -1).numpy()
    rng = np.random.default_rng(2)
    draws = [P.act(x, rng) for _ in range(20000)]
    freq = np.bincount([a for a, _, _ in draws], minlength=9) / len(draws)
    assert np.abs(freq - probs).max() < 0.015
    a, logp, v = draws[0]
    assert abs(float(logp) - float(np.log(probs[a]))) < 1e-4


def test_refresh_follows_new_weights():
    m = _model("single")
    P = NumpyPolicy(m)
    x = np.ones(12, np.float32)
    before = P.logits(x).copy()
    with torch.no_grad():
        m.pi[-1].bias.add_(1.0)
    P.refresh(m)
    assert np.allclose(P.logits(x), before + 1.0, atol=1e-5)

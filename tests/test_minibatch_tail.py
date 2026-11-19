"""The PPO minibatch split with a batch that is not a multiple of the minibatch (pool chunks fill the batch):
a short remainder joins the previous minibatch; a one-row remainder used to give a NaN advantage std (r8, u16870)."""
import numpy as np
import torch

from ballrain.train.ppo import ActorCritic, ppo_update


def _batch(n, n_obs=6, n_act=4, seed=0):
    r = np.random.default_rng(seed)
    return {"obs": r.normal(size=(n, n_obs)).astype(np.float32), "actions": r.integers(0, n_act, n),
            "logp": np.full(n, -np.log(n_act), np.float32), "adv": r.normal(size=n).astype(np.float32),
            "ret": r.normal(size=n).astype(np.float32)}


def _run(n, minibatch=8, epochs=2):
    torch.manual_seed(1)
    model = ActorCritic(6, 4, hidden=(16,))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    steps = []
    orig = opt.step
    opt.step = lambda *a, **k: (steps.append(1), orig(*a, **k))[1]
    gen = torch.Generator().manual_seed(2)
    out = ppo_update(model, opt, _batch(n), epochs, minibatch, 0.2, 0.5, 0.01, 0.5, gen=gen)
    return model, out, len(steps)


def test_one_row_remainder_is_finite_and_merged():
    model, out, steps = _run(65)  # 8 * 8 + 1: the old split stepped a minibatch of one row (std NaN)
    assert "nonfinite" not in out
    assert all(bool(torch.isfinite(p).all()) for p in model.parameters())
    assert np.isfinite([out["pg_loss"], out["v_loss"], out["entropy"], out["grad_norm"]]).all()
    assert steps == 2 * 8  # the last minibatch has 9 rows


def test_short_and_long_remainders():
    assert _run(67)[2] == 2 * 8  # 3 rows (< half a minibatch) join the previous minibatch
    assert _run(68)[2] == 2 * 9  # 4 rows (half a minibatch) stay their own minibatch
    assert _run(64)[2] == 2 * 8  # a multiple is split as before
    assert _run(3)[2] == 2 * 1  # a batch shorter than a minibatch is one minibatch

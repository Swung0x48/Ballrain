"""Growing a checkpoint's observation (ballrain.train.grow, used by scripts/train_m2.py for --resume, --init-from and
auto-rollback): a policy resumed with the f7 reach block appended gives bit-identical action logits and values for any
values in the new entries; the Adam moments of the new columns are zero and the old ones are kept; a shrink or a
layout that is not a prefix is refused."""
from dataclasses import asdict

import numpy as np
import pytest
import torch

from ballrain.features import FeatureConfig, make_layout
from ballrain.train.grow import GrowError, check_prefix, load_grown
from ballrain.train.ppo import ActorCritic, RunningMeanStd, normalize

R8 = dict(camera="world", hs_block=False, far_waypoints=(20.0, 30.0, 45.0), ref_speed_at=(0.0, 15.0, 30.0))
F6, F7 = asdict(FeatureConfig(**R8)), asdict(FeatureConfig(**R8, reach=True))
N6, N7 = sum(k for _, k in make_layout(FeatureConfig(**R8))), sum(k for _, k in make_layout(FeatureConfig(**R8, reach=True)))


def trained_checkpoint(seed=0):
    """A checkpoint dict as train_m2.save_ckpt writes it, after a few real Adam steps (non-zero moments)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ActorCritic(N6, 9, (256, 256), "tanh")
    opt = torch.optim.Adam(model.parameters(), lr=1.5e-4, eps=1e-5)
    rms = RunningMeanStd(N6)
    rms.update(rng.normal(0.3, 2.0, (4096, N6)))
    for _ in range(3):
        x = torch.as_tensor(normalize(rng.normal(0.3, 2.0, (64, N6)), rms.state_dict()))
        logits, v = model(x)
        loss = logits.logsumexp(-1).mean() + (v ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return {"model": model.state_dict(), "opt": opt.state_dict(), "rms": rms.state_dict(), "n_obs": N6, "n_act": 9,
            "env_config": {"features": F6}}, model, rms


def grown(src, n_new=N7, features=F7):
    model = ActorCritic(n_new, 9, (256, 256), "tanh")
    opt = torch.optim.Adam(model.parameters(), lr=1.5e-4, eps=1e-5)
    rms = RunningMeanStd(n_new)
    assert load_grown(model, opt, rms, src, n_new, features, load_opt=True)
    return model, opt, rms


def test_resumed_policy_is_bit_identical_for_any_values_in_the_new_block():
    """The workers' path (one observation per step: ActorCritic.act / value) is bit-identical. A batched forward (the
    learner's PPO minibatches) is identical up to float32 summation order: the GEMM blocks a 267-long dot product
    differently from a 208-long one (MKL here: first-layer pre-activations differ in the last bits, logits by ~1e-8);
    in float64 the grown model is the old one to ~1e-15, i.e. the zero columns contribute exactly nothing."""
    src, old, old_rms = trained_checkpoint()
    new, _, new_rms = grown(src)
    rng = np.random.default_rng(5)
    x6 = rng.normal(0.3, 2.0, (257, N6))
    for block in (rng.uniform(-1, 1, (257, N7 - N6)), rng.uniform(-1e6, 1e6, (257, N7 - N6)),
                  np.zeros((257, N7 - N6)), np.full((257, N7 - N6), -10.0)):
        x7 = np.concatenate([x6, block], 1)
        a6 = normalize(x6, old_rms.state_dict())
        a7 = normalize(x7, new_rms.state_dict())
        # the worker's path: one row at a time, bit-identical (logits, sampled action, log-prob, value)
        for i in range(len(a6)):
            with torch.no_grad():
                la, va = old(torch.as_tensor(a6[i:i + 1]))
                lb, vb = new(torch.as_tensor(a7[i:i + 1]))
            assert torch.equal(la, lb) and torch.equal(va, vb), i
            if i % 64 == 0:
                g = torch.Generator().manual_seed(i)
                a, lp, v = old.act(a6[i], gen=g)
                g = torch.Generator().manual_seed(i)
                b, lq, w = new.act(a7[i], gen=g)
                assert a == b and lp == lq and v == w
                assert old.value(a6[i]) == new.value(a7[i])
        # a batch: float32 summation order only
        with torch.no_grad():
            l6, v6 = old(torch.as_tensor(a6))
            l7, v7 = new(torch.as_tensor(a7))
        assert float((l6 - l7).abs().max()) <= 1e-6 and float((v6 - v7).abs().max()) <= 1e-5 * (1 + float(v6.abs().max()))
        assert torch.equal(l6.argmax(-1), l7.argmax(-1))
        o64, n64 = ActorCritic(N6, 9).double(), ActorCritic(N7, 9).double()
        o64.load_state_dict(old.state_dict()), n64.load_state_dict(new.state_dict())
        with torch.no_grad():
            d64 = (o64(torch.as_tensor(a6, dtype=torch.float64))[0] - n64(torch.as_tensor(a7, dtype=torch.float64))[0])
        assert float(d64.abs().max()) <= 1e-12


def test_adam_moments_grown_seeded_and_kept_for_the_old_columns():
    src, _, _ = trained_checkpoint()
    model, opt, rms = grown(src)
    names = [n for n, _ in model.named_parameters()]
    old_state = src["opt"]["state"]
    for i, n in enumerate(names):
        s, o = opt.state_dict()["state"][i], old_state[i]
        assert float(s["step"]) == float(o["step"])
        for k in ("exp_avg", "exp_avg_sq"):
            if n in ("pi.0.weight", "v.0.weight"):
                assert s[k].shape == (256, N7)
                assert torch.equal(s[k][:, :N6], o[k])
                if k == "exp_avg":
                    assert not s[k][:, N6:].any()
                else:  # each row's new columns: the mean of its old columns
                    want = o[k].mean(1, keepdim=True).expand(256, N7 - N6)
                    assert torch.equal(s[k][:, N6:], want) and (s[k][:, N6:] > 0).all()
            else:
                assert torch.equal(s[k], o[k])
    assert np.array_equal(rms.mean[:N6], src["rms"]["mean"]) and not rms.mean[N6:].any()
    assert np.array_equal(rms.var[:N6], src["rms"]["var"]) and (rms.var[N6:] == 1).all()
    assert rms.count == src["rms"]["count"]
    # the optimiser steps on the grown model (shapes consistent); the new columns start moving
    x = torch.randn(32, N7)
    logits, v = model(x)
    (logits.logsumexp(-1).mean() + (v ** 2).mean()).backward()
    opt.step()
    assert model.pi[0].weight[:, N6:].abs().sum() > 0


def test_first_steps_of_the_new_columns_are_not_larger_than_the_learning_rate():
    """With the checkpoint's step count large (bias correction gone: r8 is at 2.9e5 optimiser steps) a gradient of the
    old columns' typical size moves a new column by about 0.1 lr per step for the first steps; with zero second
    moments it would move ~3.2 lr (0.1 g / sqrt(0.001 g^2)), the 3-6x the seeding avoids. Logits stay bit-identical."""
    from ballrain.train.grow import grow_optimizer_state
    src, old, old_rms = trained_checkpoint()
    gen = torch.Generator().manual_seed(3)
    for s in src["opt"]["state"].values():
        s["step"] = torch.tensor(293411.0)  # the r8 latest checkpoint
        if s["exp_avg_sq"].shape == (256, N6):  # its pi.0.weight second moments: median 3.4e-7, sqrt 6e-4 >> eps
            s["exp_avg_sq"] = 1e-7 + 9e-7 * torch.rand(256, N6, generator=gen, dtype=s["exp_avg_sq"].dtype)
    lr = 1.5e-4

    def first_steps(seed: bool):
        model = ActorCritic(N7, 9, (256, 256), "tanh")
        opt = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
        rms = RunningMeanStd(N7)
        load_grown(model, opt, rms, src, N7, F7, load_opt=True)
        if not seed:  # the zero-moment growth, for comparison
            sd = grow_optimizer_state(src["opt"], model, N6, N7)
            for s in sd["state"].values():
                if s["exp_avg_sq"].dim() == 2 and s["exp_avg_sq"].shape[1] == N7:
                    s["exp_avg_sq"][:, N6:] = 0
            opt.load_state_dict(sd)
        w = model.pi[0].weight
        v_old = opt.state[w]["exp_avg_sq"][:, :N6].mean(1, keepdim=True)
        g = torch.sqrt(v_old).expand(w.shape[0], N7 - N6)  # a gradient of the old columns' rms size
        moves = []
        for _ in range(5):
            before = w.detach().clone()
            opt.zero_grad()
            w.grad = torch.zeros_like(w)
            w.grad[:, N6:] = g
            opt.step()
            moves.append(float((w.detach() - before)[:, N6:].abs().max()) / lr)
        return moves

    seeded, zero = first_steps(True), first_steps(False)
    assert max(seeded) < 0.6 and min(zero) > 1.5 and max(zero) > 3.0, (seeded, zero)
    # at resume (no step yet) the grown policy is the old one
    new, _, new_rms = grown(src)
    x = np.random.default_rng(1).normal(size=(1, N6))
    with torch.no_grad():
        la, va = old(torch.as_tensor(normalize(x, old_rms.state_dict())))
        lb, vb = new(torch.as_tensor(normalize(np.c_[x, np.ones((1, N7 - N6))], new_rms.state_dict())))
    assert torch.equal(la, lb) and torch.equal(va, vb)


def test_same_size_loads_as_is_and_shrink_or_non_prefix_is_refused():
    src, old, _ = trained_checkpoint()
    model = ActorCritic(N6, 9, (256, 256), "tanh")
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, eps=1e-5)
    assert not load_grown(model, opt, RunningMeanStd(N6), src, N6, F6)
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), old.state_dict().values()))
    with pytest.raises(GrowError, match="shrink"):
        load_grown(ActorCritic(N6 - 9, 9), None, RunningMeanStd(N6 - 9), src, N6 - 9, None)
    # an old layout that is not the start of the new one (here: the look-ahead block missing in the new config,
    # more entries from a larger height grid) is refused
    other = asdict(FeatureConfig(camera="world", hs_block=False, height_grid=9, reach=True))
    n_other = sum(k for _, k in make_layout(FeatureConfig(**other)))
    assert n_other > N6
    with pytest.raises(GrowError, match="prefix"):
        check_prefix(F6, other, N6, n_other)
    check_prefix(F6, F7, N6, N7)  # the f7 growth itself
    check_prefix(None, None, N6, N7)  # old checkpoints without a feature config: sizes only


def test_train_m2_wiring():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import train_m2
    assert train_m2.parse(["--run", "x"]).reach is False
    a = train_m2.parse(["--run", "x", "--reach", "--world-actions", "--level", "2"])
    assert a.reach
    fc = train_m2.make_env_config(a, Path("unused")).features
    assert fc.reach and fc.version == "f7" and fc.camera == "world"
    src = (train_m2.REPO / "scripts" / "train_m2.py").read_text(encoding="utf-8")
    assert "load_grown" in src and 'saved.get("reach")' in src

"""Three unmodified float32 handovers from r8 collapse probes: 967-GG, paper-GG, stone-GG (step 0)."""
from pathlib import Path

import numpy as np
import torch

from ballrain.pool.learner_link import state_from_params
from ballrain.pool.weights import export_state, load_policy
from ballrain.train.grow import load_grown
from ballrain.train.ppo import ActorCritic, RunningMeanStd, normalize, ppo_update


def test_split_train_resume_and_publish(tmp_path):
    torch.manual_seed(13)
    raw = np.load(Path(__file__).parent / 'data/r8_ball_handovers.npz')['obs']
    n = raw.shape[1]
    rms = RunningMeanStd(n)
    rms.update(raw[[0] * 100 + [1]])  # unequal frequencies, stone not seen yet, clipping on
    real = torch.from_numpy(normalize(raw, rms.state_dict()))
    legacy = ActorCritic(n, 9, (32,))
    old_opt = torch.optim.Adam(legacy.parameters())
    l, v = legacy(real)
    (l.sum() + v.sum()).backward()
    old_opt.step()
    src = dict(model=legacy.state_dict(), opt=old_opt.state_dict(), rms=rms.state_dict(), n_obs=n)
    model = ActorCritic(n, 9, (32,), actor_heads='ball_type')
    opt = torch.optim.Adam(model.parameters())
    assert load_grown(model, opt, rms, src, n)
    assert model.context(real).tolist() == [0, 1, 2]
    assert model.context(torch.from_numpy(normalize(np.zeros(n), rms.state_dict()))).item() == 0
    for x in (real, real[1], torch.randn(117, n)):
        assert all(torch.equal(a, b) for a, b in zip(legacy(x), model(x)))
    for head in model.extra_heads:
        for p, q in zip(head.parameters(), model.pi[-1].parameters()):
            for k, value in opt.state[q].items():
                assert torch.equal(opt.state[p][k], value) and opt.state[p][k].data_ptr() != value.data_ptr()

    # A stone-only PPO minibatch must not step wood or paper, even with nonzero Adam history.
    x = real[2:].repeat(16, 1)
    logits, _ = model(x)
    actions = torch.arange(16) % 9
    batch = dict(obs=x, actions=actions, logp=logits.detach().log_softmax(-1)[torch.arange(16), actions],
                 adv=torch.arange(16).float(), ret=torch.ones(16))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    stats = ppo_update(model, opt, batch, 2, 8, 0.2, 0.5, 0.01, 0.5)
    assert 'nonfinite' not in stats
    for k in ('pi.2.weight', 'pi.2.bias', 'extra_heads.0.weight', 'extra_heads.0.bias'):
        assert torch.equal(before[k], model.state_dict()[k])
        assert opt.state[dict(model.named_parameters())[k]]['step'] == 1
    for k in ('extra_heads.1.weight', 'pi.0.weight', 'v.0.weight'):
        assert not torch.equal(before[k], model.state_dict()[k])

    ckpt = tmp_path / 'latest.pt'
    args = dict(actor_heads='ball_type', activation='tanh')
    torch.save(dict(src, args=args, model=model.state_dict(), opt=opt.state_dict()), ckpt)
    saved = torch.load(ckpt, weights_only=False)
    other = ActorCritic(n, 9, (32,), actor_heads=saved['args']['actor_heads'])
    other_opt = torch.optim.Adam(other.parameters())
    assert not load_grown(other, other_opt, rms, saved, n)
    for p, q in zip(model.parameters(), other.parameters()):
        assert torch.equal(p, q)
        assert all(torch.equal(v, other_opt.state[q][k]) for k, v in opt.state[p].items())

    # Exercise the remote template + parameter message -> pool export path with distinguishable heads.
    with torch.no_grad():
        model.extra_heads[0].bias[1] += 1
        model.extra_heads[1].bias[2] += 2
    state = state_from_params(dict(worker=args, env=dict(mode='sr')), dict(
        model={k: v.numpy() for k, v in model.state_dict().items()}, version=1, rms=rms.state_dict()))
    m = export_state(state, tmp_path, 'test')
    assert 'actor_ball_type_v1' in m['requires']
    policy = load_policy(tmp_path / (m['policy_sha'] + '.npz'), m)
    for x in (real, *real):
        np.testing.assert_allclose(policy.logits(x.numpy()), model(x)[0].detach().numpy(), atol=1e-6)

"""Teacher KL term of ppo_update (scripts/train_m2.py --teacher): + coef * KL(teacher || policy) on the masked batch
states pulls the policy towards the teacher there and leaves unmasked states alone."""
import sys

import numpy as np
import torch

from ballrain import instances
from ballrain.train.ppo import ActorCritic, ppo_update


def _setup(seed=0, n=256, n_obs=8, n_act=5):
    torch.manual_seed(seed)
    pi = ActorCritic(n_obs, n_act, (32, 32))
    teacher = ActorCritic(n_obs, n_act, (32, 32))
    obs = np.random.default_rng(seed).normal(size=(n, n_obs)).astype(np.float32)
    with torch.no_grad():
        lt = torch.log_softmax(teacher(torch.as_tensor(obs))[0], -1).numpy()
        lp = torch.log_softmax(pi(torch.as_tensor(obs))[0], -1).numpy()
    batch = {"obs": obs, "actions": np.zeros(n, np.int64), "logp": lp[:, 0].astype(np.float32),
             "adv": np.zeros(n, np.float32), "ret": np.zeros(n, np.float32)}
    return pi, obs, lt, batch


def _kl(pi, obs, lt, mask):
    with torch.no_grad():
        lp = torch.log_softmax(pi(torch.as_tensor(obs))[0], -1).numpy()
    return float((np.exp(lt) * (lt - lp)).sum(-1)[mask].mean())


def _update(pi, batch, lt, mask, coef):
    opt = torch.optim.Adam(pi.parameters(), lr=3e-3)
    return ppo_update(pi, opt, batch, epochs=8, minibatch=64, clip=0.2, vf_coef=0.0, ent_coef=0.0, max_grad_norm=1.0,
                      teacher_logp=lt, teacher_mask=mask, teacher_coef=coef)


def test_kl_falls_on_masked_states_only():
    pi, obs, lt, batch = _setup()
    mask = np.zeros(len(obs), bool)
    mask[: len(obs) // 2] = True
    before_m, before_u = _kl(pi, obs, lt, mask), _kl(pi, obs, lt, ~mask)
    st = _update(pi, batch, lt, mask, 1.0)
    assert _kl(pi, obs, lt, mask) < 0.5 * before_m
    assert st["teacher_n"] == mask.sum() and st["teacher_coef"] == 1.0 and np.isfinite(st["teacher_kl"])
    # unmasked states are not targeted (they move only through shared weights): far less than the masked ones
    assert before_m - _kl(pi, obs, lt, mask) > before_u - _kl(pi, obs, lt, ~mask)


def test_no_mask_or_zero_coef_leaves_the_policy():
    for mask, coef in ((np.zeros(256, bool), 1.0), (np.ones(256, bool), 0.0)):
        pi, obs, lt, batch = _setup(1)
        with torch.no_grad():
            before = pi(torch.as_tensor(obs))[0].clone()
        st = _update(pi, batch, lt, mask, coef)
        with torch.no_grad():
            after = pi(torch.as_tensor(obs))[0]
        assert torch.allclose(before, after, atol=1e-6)  # zero advantages, no entropy / value term: nothing moves pi
        assert "teacher_n" not in st or np.isnan(st.get("teacher_kl", np.nan))


def test_options():
    sys.path.insert(0, str(instances.REPO_ROOT / "scripts"))
    import train_m2
    a = train_m2.parse(["--run", "x"])
    assert a.teacher == "" and a.teacher_coef == 0.1 and a.teacher_decay_updates == 0 and a.teacher_s_min == 331.4
    a = train_m2.parse(["--run", "x", "--teacher", "t.pt", "--teacher-coef", "0.1", "--teacher-coef-floor", "0.05",
                        "--teacher-decay-updates", "5000"])
    assert a.teacher == "t.pt" and a.teacher_coef_floor == 0.05 and a.teacher_decay_updates == 5000
    assert "u_teacher_kl" in train_m2.UPDATE_COLS and "u_teacher_coef" in train_m2.UPDATE_COLS

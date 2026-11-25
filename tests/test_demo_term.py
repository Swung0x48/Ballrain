"""The imitation term of ppo_update (train_m2 --demo) and the demo builder's action mapping."""
import sys
from pathlib import Path

import numpy as np
import torch

from ballrain import tas
from ballrain.train.ppo import ActorCritic, ppo_update

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def _batch(n, n_obs, rng):
    return {"obs": rng.normal(size=(n, n_obs)).astype(np.float32), "actions": rng.integers(0, 9, n),
            "logp": np.full(n, -np.log(9), np.float32), "adv": rng.normal(size=n).astype(np.float32),
            "ret": rng.normal(size=n).astype(np.float32)}


def test_demo_term_pulls_the_policy_toward_the_demonstration():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    m = ActorCritic(6, 9, (32, 32))
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    d_obs = rng.normal(size=(64, 6)).astype(np.float32)
    d_act = np.full(64, 7, np.int64)
    first = last = None
    for _ in range(30):
        st = ppo_update(m, opt, _batch(256, 6, rng), 1, 128, 0.2, 0.5, 0.0, 0.5, gen=torch.Generator().manual_seed(1),
                        demo_obs=d_obs, demo_act=d_act, demo_coef=1.0, demo_batch=64)
        first = st["demo_acc"] if first is None else first
        last = st["demo_acc"]
    assert last > 0.9 and last >= first


def test_no_demo_no_stats():
    rng = np.random.default_rng(1)
    m = ActorCritic(6, 9, (16,))
    st = ppo_update(m, torch.optim.Adam(m.parameters()), _batch(64, 6, rng), 1, 64, 0.2, 0.5, 0.0, 0.5)
    assert "demo_ce" not in st


def test_world_action_mapping_identity_camera():
    import build_demo
    cam = np.eye(4, 3)  # x, y, z rows of an unrotated camera reference
    assert build_demo.world_action(0, cam) == 0
    assert build_demo.world_action(tas.KEY_UP, cam) == 1
    assert build_demo.world_action(tas.KEY_DOWN | tas.KEY_LEFT, cam) == 7

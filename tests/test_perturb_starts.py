"""Perturbed starts (ballrain.train.worker WorkerSpec.perturb_starts): a TAS start in the frame range is followed by a
short random action prefix before the policy takes over; off by default (no env step, unchanged info). No game is launched."""
import sys
from types import SimpleNamespace

import numpy as np

from ballrain.train.worker import Worker, WorkerSpec, _Episode


class _Env:
    def __init__(self, die_at=None):
        self.action_space = SimpleNamespace(n=9)
        self.steps = []
        self.die_at = die_at
        self.reward = SimpleNamespace(s=0.0)

    def step(self, a):
        self.steps.append(a)
        term = self.die_at is not None and len(self.steps) >= self.die_at
        return np.full(3, len(self.steps), np.float32), 0.0, term, False, {"sector": 1, "sr_ms": 10.0 * len(self.steps)}


def _w(ps, env, seed=0):
    w = object.__new__(Worker)
    w.spec = SimpleNamespace(perturb_starts=ps) if ps is not None else SimpleNamespace()
    w.env = env
    w._perturb_rng = np.random.default_rng(seed)
    return w


def _info(kind="tas", frame=8000):
    return {"start": {"kind": kind, "frame": frame}, "sector": 1, "sr_ms": 0.0}


def test_off_leaves_the_start_unchanged():
    for ps in (None, {}):
        env = _Env()
        assert _w(ps, env)._perturb_prefix(_info()) is None
        assert env.steps == []
    assert WorkerSpec(wid=0, copy="x", env=None, seed=0).perturb_starts == {}


def test_prefix_in_range_records_frame_and_actions():
    ps = {"frames": [7850, 8085], "steps": [2, 20], "p": 1.0}
    lens = set()
    for seed in range(20):
        env = _Env()
        out = _w(ps, env, seed)._perturb_prefix(_info(frame=7900))
        assert out is not None and out is not False
        obs, info = out
        acts = info["start"]["perturb"]["actions"]
        assert info["start"]["perturb"]["frame"] == 7900 and info["start"]["kind"] == "tas"
        assert acts == env.steps and 2 <= len(acts) <= 20 and all(0 <= a < 9 for a in acts)
        assert obs[0] == len(acts) and info["sr_ms"] == 10.0 * len(acts)  # the state after the prefix
        lens.add(len(acts))
        ep = _Episode(info, env, 0, None)
        assert ep.d["perturb_frame"] == 7900 and ep.d["perturb_actions"] == "".join(map(str, acts))
    assert len(lens) > 3  # the prefix length varies


def test_same_seed_same_prefix():
    ps = {"frames": [0, 99999], "steps": [4, 40]}
    a = _w(ps, _Env(), 7)._perturb_prefix(_info())[1]["start"]["perturb"]["actions"]
    b = _w(ps, _Env(), 7)._perturb_prefix(_info())[1]["start"]["perturb"]["actions"]
    assert a == b


def test_outside_range_other_kinds_and_p():
    ps = {"frames": [7850, 8085], "steps": [2, 4], "p": 1.0}
    for info in (_info(frame=7849), _info(frame=8086), _info(kind="tas_rest", frame=7900), _info(kind="respawn", frame=7900),
                 {"sector": 1}):
        env = _Env()
        assert _w(ps, env)._perturb_prefix(info) is None and env.steps == []
    env = _Env()
    w = _w({**ps, "p": 0.0}, env)
    assert all(w._perturb_prefix(_info(frame=7900)) is None for _ in range(20)) and env.steps == []


def test_prefix_that_ends_the_life():
    env = _Env(die_at=1)
    assert _w({"frames": [0, 99999], "steps": [3, 3]}, env)._perturb_prefix(_info()) is False


def test_episode_without_perturb_has_no_new_keys():
    ep = _Episode(_info(), _Env(), 0, None)
    assert "perturb_frame" not in ep.d and "perturb_actions" not in ep.d


def test_train_m2_option_reaches_spec_and_remote_template():
    sys.path.insert(0, str(__import__("ballrain.instances", fromlist=["REPO_ROOT"]).REPO_ROOT / "scripts"))
    import train_m2
    assert train_m2.parse(["--run", "x"]).perturb_starts == ""
    a = train_m2.parse(["--run", "x", "--perturb-starts", '{"frames": [7850, 8085], "steps": [2, 20], "p": 0.5}'])
    assert a.perturb_starts == '{"frames": [7850, 8085], "steps": [2, 20], "p": 0.5}'
    src = (train_m2.REPO / "scripts" / "train_m2.py").read_text(encoding="utf-8")
    assert src.count("perturb_starts=json.loads(a.perturb_starts) if a.perturb_starts else {}") == 1  # local workers
    assert '"perturb_starts": json.loads(a.perturb_starts) if a.perturb_starts else {}' in src  # remote template

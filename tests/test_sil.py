"""Self-imitation (train_m2 --sil-coef): the buffer (ballrain.train.sil) and the loss term of ppo_update."""
import numpy as np
import torch

from ballrain.train.ppo import ActorCritic, ppo_update
from ballrain.train.sil import SILBuffer, value_fn


def _batch(n, n_obs, rng):
    return {"obs": rng.normal(size=(n, n_obs)).astype(np.float32), "actions": rng.integers(0, 9, n),
            "logp": np.full(n, -np.log(9), np.float32), "adv": rng.normal(size=n).astype(np.float32),
            "ret": rng.normal(size=n).astype(np.float32)}


def test_buffer_takes_rows_above_the_threshold_only():
    b = SILBuffer(capacity=100, adv_min=2.0)
    raw = np.arange(10, dtype=np.float32)[:, None].repeat(3, 1)
    adv = np.array([0, 5, 1, 3, -4, 0, 2.5, 0, 0, 1.9], np.float32)
    n = b.add(raw, np.arange(10), np.zeros(10), adv, adv_std=1.0)
    assert n == 3 and len(b) == 3
    assert list(b.act) == [1, 3, 6] and b.raw.shape == (3, 3)
    assert b.add(raw, np.arange(10), np.zeros(10), adv, adv_std=0.0) == 0  # no spread: nothing is "better"


def test_buffer_keeps_the_best_rows_when_full():
    b = SILBuffer(capacity=4, adv_min=0.0)
    adv = np.array([1, 9, 2, 8, 3, 7], np.float32)
    b.add(np.zeros((6, 2), np.float32), np.arange(6), np.zeros(6), adv, adv_std=1.0)
    assert sorted(b.act.tolist()) == [1, 3, 4, 5]


def test_refresh_drops_rows_the_value_head_caught_up_with():
    b = SILBuffer(capacity=10, adv_min=0.0, keep_min=0.1)
    b.add(np.zeros((3, 2), np.float32), np.array([0, 1, 2]), np.array([1.0, 5.0, 10.0]), np.ones(3), adv_std=1.0)
    dropped = b.refresh(lambda x: np.full(len(x), 5.0, np.float32), np.zeros((3, 2), np.float32), adv_std=1.0)
    assert dropped == 2 and b.act.tolist() == [2] and np.allclose(b.score, [5.0])


def test_value_fn_matches_the_model():
    m = ActorCritic(4, 9, (8,))
    x = np.random.default_rng(0).normal(size=(5, 4)).astype(np.float32)
    with torch.no_grad():
        _, v = m(torch.as_tensor(x))
    assert np.allclose(value_fn(m)(x), v.numpy())


def test_sil_term_pulls_the_policy_toward_better_than_expected_actions():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    m = ActorCritic(6, 9, (32, 32))
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    s_obs = rng.normal(size=(64, 6)).astype(np.float32)
    s_act = np.full(64, 4, np.int64)
    s_ret = np.full(64, 3.0, np.float32)  # well above the (near-zero) value of an untrained head
    for _ in range(30):
        st = ppo_update(m, opt, _batch(256, 6, rng), 1, 128, 0.2, 0.5, 0.0, 0.5, gen=torch.Generator().manual_seed(1),
                        sil_obs=s_obs, sil_act=s_act, sil_ret=s_ret, sil_coef=1.0, sil_scale=1.0, sil_batch=64)
    with torch.no_grad():
        logits, _ = m(torch.as_tensor(s_obs))
    assert float((logits.argmax(-1) == 4).float().mean()) > 0.9
    assert st["sil_n"] == 64 and st["sil_w"] > 0


def test_sil_rows_below_the_value_do_not_push():
    torch.manual_seed(0)
    rng = np.random.default_rng(2)
    m = ActorCritic(6, 9, (16,))
    s_obs = rng.normal(size=(32, 6)).astype(np.float32)
    st = ppo_update(m, torch.optim.Adam(m.parameters()), _batch(64, 6, rng), 1, 64, 0.2, 0.5, 0.0, 0.5,
                    sil_obs=s_obs, sil_act=np.zeros(32, np.int64), sil_ret=np.full(32, -100.0, np.float32),
                    sil_coef=1.0, sil_scale=1.0)
    assert st["sil_w"] == 0.0 and st["sil_pos"] == 0.0 and st["sil_pg"] == 0.0


def test_sil_off_no_stats_and_value_only_skips_it():
    rng = np.random.default_rng(1)
    m = ActorCritic(6, 9, (16,))
    st = ppo_update(m, torch.optim.Adam(m.parameters()), _batch(64, 6, rng), 1, 64, 0.2, 0.5, 0.0, 0.5)
    assert "sil_pg" not in st
    st = ppo_update(m, torch.optim.Adam(m.parameters()), _batch(64, 6, rng), 1, 64, 0.2, 0.5, 0.0, 0.5,
                    value_only=True, sil_obs=np.zeros((4, 6), np.float32), sil_act=np.zeros(4, np.int64),
                    sil_ret=np.ones(4, np.float32), sil_coef=1.0)
    assert "sil_pg" not in st


def _chunk(T, wid=0, ends=(), adv_rows=()):
    term = np.zeros(T, bool)
    eps = []
    for t, reason in ends:
        term[t] = True
        eps.append({"end": reason})
    return {"actions": np.zeros(T, np.int64), "terminated": term, "truncated": np.zeros(T, bool), "episodes": eps,
            "wid": wid, "copy": f"c{wid}"}, np.array([5.0 if t in adv_rows else 0.0 for t in range(T)], np.float32)


def _offer(b, parts):
    chunks = [c for c, _ in parts]
    adv = np.concatenate([a for _, a in parts])
    n = len(adv)
    raw = np.arange(n, dtype=np.float32)[:, None].repeat(2, 1)
    return b.offer(chunks, raw, np.arange(n), np.ones(n, np.float32), adv, 1.0)


def test_candidate_confirmed_after_its_life_goes_on_for_the_horizon():
    b = SILBuffer(capacity=100, adv_min=2.0, horizon=5)
    assert _offer(b, [_chunk(4, adv_rows=(1,))]) == 0  # 2 rows after it: still pending
    assert len(b) == 0 and b.pending[(0, "c0")]["rows"]
    assert _offer(b, [_chunk(4)]) == 1  # the next chunk of the same worker completes the horizon
    assert b.raw[0, 0] == 1.0 and b.confirmed == 1


def test_candidate_dropped_when_its_life_dies_or_stalls_within_the_horizon():
    for reason in ("BallOff", "stall", "cap", None):
        b = SILBuffer(capacity=100, adv_min=2.0, horizon=10)
        _offer(b, [_chunk(8, ends=((6, reason),), adv_rows=(2,))])
        _offer(b, [_chunk(20)])  # a new life afterwards does not revive it
        assert len(b) == 0 and b.discarded == 1, reason


def test_level_finish_within_the_horizon_confirms():
    b = SILBuffer(capacity=100, adv_min=2.0, horizon=50)
    assert _offer(b, [_chunk(8, ends=((6, "LevelFinish"),), adv_rows=(2, 6))]) == 2


def test_workers_are_followed_separately():
    b = SILBuffer(capacity=100, adv_min=2.0, horizon=3)
    _offer(b, [_chunk(2, wid=0, adv_rows=(1,)), _chunk(10, wid=1, ends=((0, "stall"),))])
    assert len(b) == 0 and b.discarded == 0  # worker 1's stall does not touch worker 0's candidate
    _offer(b, [_chunk(5, wid=0)])
    assert len(b) == 1

"""Non-finite guard of the learner (ballrain.train.nonfinite, ppo.ppo_update, scripts/train_m2.py):
- a finite update is bit-identical to ppo_update before the guard (parameters, optimiser state, RNG draws, stats);
- a batch with NaN / inf is not trained on: no parameter change, the skip is reported, a dump is written;
- a finite batch whose loss or gradient turns non-finite in a later minibatch: parameters and optimiser state are the
  pre-update ones although earlier minibatches had stepped;
- the dump holds the batch and its sources and nothing that names a host, a copy or a path; at most 5 per run directory;
- too many skips in a row stop the learner; non-finite parameters are neither saved nor published (fake run)."""
import copy
import csv
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from ballrain.train.nonfinite import (MAX_CONSECUTIVE, MAX_DUMPS, NonFiniteError, NonFiniteGuard, nonfinite_keys,
                                      nonfinite_rows)
from ballrain.train.ppo import ActorCritic, ppo_update

REPO = Path(__file__).resolve().parents[1]
N, MB, SEED = 512, 64, 11
ARGS = dict(epochs=2, minibatch=MB, clip=0.2, vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5)
SECRETS = ("volunteer-host-01", "train-07", "D:/somewhere/episodes/run-1")  # chunk fields that must not reach a dump


def _ppo_update_base(model, opt, batch, epochs, minibatch, clip, vf_coef, ent_coef, max_grad_norm, device="cpu", gen=None,
                     target_kl=None, teacher_logp=None, teacher_mask=None, teacher_coef=0.0) -> dict:
    """ballrain.train.ppo.ppo_update as it was before the guard (commit cfc1da9), for the bit-identity comparison."""
    obs = torch.as_tensor(batch["obs"], device=device)
    act = torch.as_tensor(batch["actions"], dtype=torch.long, device=device)
    old_logp = torch.as_tensor(batch["logp"], device=device)
    adv = torch.as_tensor(batch["adv"], device=device)
    ret = torch.as_tensor(batch["ret"], device=device)
    n = obs.shape[0]
    stats = {"pg_loss": [], "v_loss": [], "entropy": [], "approx_kl": [], "clip_frac": [], "grad_norm": []}
    use_teacher = teacher_logp is not None and teacher_mask is not None and teacher_coef > 0
    if use_teacher:
        t_logp = torch.as_tensor(teacher_logp, dtype=torch.float32, device=device)
        t_mask = torch.as_tensor(teacher_mask, dtype=torch.bool, device=device)
        stats["teacher_kl"] = []
    stop = False
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen).to(device)
        for i in range(0, n, minibatch):
            idx = perm[i:i + minibatch]
            logits, v = model(obs[idx])
            logp_all = torch.log_softmax(logits, -1)
            logp = logp_all.gather(-1, act[idx, None]).squeeze(-1)
            ent = -(logp_all.exp() * logp_all).sum(-1).mean()
            a = adv[idx]
            a = (a - a.mean()) / (a.std() + 1e-8)
            ratio = (logp - old_logp[idx]).exp()
            pg = -torch.min(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a).mean()
            vl = 0.5 * ((v - ret[idx]) ** 2).mean()
            loss = pg + vf_coef * vl - ent_coef * ent
            if use_teacher:
                m = t_mask[idx]
                if bool(m.any()):
                    lt = t_logp[idx][m]
                    kl_t = (lt.exp() * (lt - logp_all[m])).sum(-1).mean()
                    loss = loss + teacher_coef * kl_t
                    stats["teacher_kl"].append(kl_t.item())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # total norm before clipping
            opt.step()
            with torch.no_grad():
                lr = logp - old_logp[idx]
                kl = ((lr.exp() - 1) - lr).mean().item()
                stats["pg_loss"].append(pg.item())
                stats["v_loss"].append(vl.item())
                stats["entropy"].append(ent.item())
                stats["approx_kl"].append(kl)
                stats["clip_frac"].append(((ratio - 1).abs() > clip).float().mean().item())
                stats["grad_norm"].append(float(gn))
            if target_kl is not None and kl > 1.5 * target_kl:
                stop = True
                break
        if stop:
            break
    out = {k: float(np.mean(v)) if v else float("nan") for k, v in stats.items()}
    out["early_stop"] = float(stop)
    if use_teacher:
        out["teacher_n"] = float(t_mask.sum().item())
        out["teacher_coef"] = float(teacher_coef)
    return out


def _setup(seed=0, n_obs=8, n_act=9):
    """A policy, a finite batch around it (non-zero advantages / returns, old log-probs near the policy's), the old
    values and a teacher's log-probabilities."""
    torch.manual_seed(seed)
    model, teacher = ActorCritic(n_obs, n_act, (32, 32)), ActorCritic(n_obs, n_act, (32, 32))
    rng = np.random.default_rng(seed)
    obs = rng.normal(size=(N, n_obs)).astype(np.float32)
    act = rng.integers(0, n_act, N)
    with torch.no_grad():
        logits, v = model(torch.as_tensor(obs))
        lp = torch.log_softmax(logits, -1).numpy()[np.arange(N), act]
        lt = torch.log_softmax(teacher(torch.as_tensor(obs))[0], -1).numpy()
    batch = {"obs": obs, "actions": act.astype(np.int64),
             "logp": (lp + rng.normal(scale=0.05, size=N)).astype(np.float32),
             "adv": rng.normal(size=N).astype(np.float32), "ret": rng.normal(size=N).astype(np.float32)}
    return model, batch, v.numpy(), lt


def _twin(model, opt=None):
    """An independent copy of the model with an Adam in the same state (a fresh one without opt)."""
    m = copy.deepcopy(model)
    o = torch.optim.Adam(m.parameters(), lr=3e-3, eps=1e-5)
    if opt is not None:
        o.load_state_dict(copy.deepcopy(opt.state_dict()))
    return m, o


def _warm(model, batch):
    """The model after one update, with its Adam (non-zero moments)."""
    m, o = _twin(model)
    _ppo_update_base(m, o, batch, gen=torch.Generator().manual_seed(1), **ARGS)
    return m, o


def _eq(a, b) -> bool:
    """Bit-equality of two nested states (model / optimiser state_dict)."""
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_eq(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_eq(x, y) for x, y in zip(a, b))
    if torch.is_tensor(a):
        return a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and np.array_equal(a, b)
    return a == b


def _state(model, opt):
    return copy.deepcopy({"model": model.state_dict(), "opt": opt.state_dict()})


def _chunks(update: int):
    """Four chunks of N / 4 rows as the learner holds them when it builds update number `update`: two own-worker
    chunks (the second from a remote worker), two pool chunks; with the fields that must never be dumped."""
    T, out = N // 4, []
    for i in range(4):
        c = {"rewards": np.full(T, 0.1 * i, np.float32), "logp": np.full(T, -1.5, np.float32),
             "terminated": np.zeros(T, bool), "truncated": np.zeros(T, bool), "last_value": 0.5,
             "version": update - 1 - i, "wid": i + 1, "backend": "headless", "host": SECRETS[0], "copy": SECRETS[1],
             "runs": [{"dir": SECRETS[2]}]}
        if i == 1:
            c["remote"] = True
        if i >= 2:
            c.update(pool=True, backend="pool", wid=0, pool_task=f"task-{i:04d}", pool_seq=10 + i)
        out.append(c)
    return out


def _load(f) -> dict:
    """The dump's arrays (no pickle), after checking that no member holds one of the chunks' private strings."""
    with zipfile.ZipFile(f) as z:
        for name in z.namelist():
            data = z.read(name)
            for s in SECRETS:
                assert s.encode() not in data and s.encode("utf-32-le") not in data, (name, s)
    with np.load(f, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ============================================================================================== finite updates

@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize("teacher", [False, True])
@pytest.mark.parametrize("target_kl", [None, 1e-4])
def test_finite_update_is_bit_identical_to_the_update_before_the_guard(tmp_path, warm, teacher, target_kl):
    model, batch, values, lt = _setup()
    opt = None
    if warm:
        model, opt = _warm(model, batch)
    kw = dict(ARGS, target_kl=target_kl)
    if teacher:
        kw.update(teacher_logp=lt, teacher_mask=np.arange(N) % 2 == 0, teacher_coef=0.1)
    guard, lines = NonFiniteGuard(tmp_path, log=lambda s: lines.append(s)), []
    runs = {"base": lambda m, o, g: _ppo_update_base(m, o, batch, gen=g, **kw),
            "guarded": lambda m, o, g: ppo_update(m, o, batch, gen=g, **kw),
            "learner": lambda m, o, g: guard.update(3, _chunks(3), {"values": values}, m, o, batch, gen=g, **kw)}
    res = {}
    for name, run in runs.items():
        m, o = _twin(model, opt)
        g = torch.Generator().manual_seed(SEED)
        torch.manual_seed(123)
        stats = run(m, o, g)
        res[name] = (_state(m, o), g.get_state(), torch.get_rng_state(), stats)
    base = res["base"]
    assert not _eq(base[0]["model"], model.state_dict())  # the update did move the parameters
    assert base[3]["early_stop"] == float(target_kl is not None) and np.isfinite(base[3]["grad_norm"])
    for name in ("guarded", "learner"):
        st, g, rng, stats = res[name]
        assert _eq(st, base[0]), name  # parameters and Adam state: every bit
        assert torch.equal(g, base[1]) and torch.equal(rng, base[2]), name  # no extra RNG draw
        np.testing.assert_equal(stats, base[3])  # the same stats, no "nonfinite" key
    assert not lines and not list(tmp_path.iterdir()) and guard.skipped == 0


# ============================================================================================== non-finite batches

@pytest.mark.parametrize("key,value", [("obs", np.nan), ("adv", np.nan), ("logp", np.nan), ("ret", np.inf),
                                       ("obs", -np.inf), ("values", np.nan), ("teacher_logp", np.nan)])
def test_non_finite_batch_is_skipped_reported_and_dumped(tmp_path, key, value):
    model, batch, values, lt = _setup()
    model, opt = _warm(model, batch)
    before = _state(model, opt)
    row = 3 * N // 4 + 7  # a row of the second pool chunk
    arrays = dict(batch, values=values, teacher_logp=lt)
    arrays[key] = arrays[key].copy()
    arrays[key][row] = value
    batch = {k: arrays[k] for k in batch}
    lines, counters = [], {"nonfinite_skipped": 2}  # a resumed run's count continues
    guard = NonFiniteGuard(tmp_path, lines.append, counters)
    g = torch.Generator().manual_seed(SEED)
    out = guard.update(7, _chunks(7), {"values": arrays["values"]}, model, opt, batch, gen=g,
                       teacher_logp=arrays["teacher_logp"], teacher_mask=np.ones(N, bool), teacher_coef=0.1, **ARGS)
    assert out is None and _eq(_state(model, opt), before)  # skipped: nothing changed
    assert torch.equal(g.get_state(), torch.Generator().manual_seed(SEED).get_state())  # the update never started
    assert counters["nonfinite_skipped"] == 3 and guard.skipped == 3 and guard.consecutive == 1
    assert len(lines) == 1 and f"NON-FINITE batch at u7 ({key})" in lines[0]
    assert "own rows 0, pool rows 1 (task/seq lag: task-0003/13 3)" in lines[0] and "dump nonfinite-u7.npz" in lines[0]
    assert [f.name for f in tmp_path.iterdir()] == ["nonfinite-u7.npz"]
    d = _load(tmp_path / "nonfinite-u7.npz")
    for k, v in arrays.items():  # everything the update would have consumed, as it was
        np.testing.assert_array_equal(d[k], v)
    assert d["teacher_mask"].all() and int(d["update"]) == 7 and str(d["where"]) == "batch" and str(d["term"]) == key
    assert np.flatnonzero(d["bad_rows"]).tolist() == [row]
    assert d["row_pool"].sum() == N // 2 and d["row_pool"][row] and d["row_chunk"][row] == 3
    assert d["chunk_pool"].tolist() == [False, False, True, True] and d["chunk_remote"].tolist() == [0, 1, 0, 0]
    assert d["chunk_pool_task"].tolist() == ["", "", "task-0002", "task-0003"] and d["chunk_lag"].tolist() == [0, 1, 2, 3]
    assert d["chunk_pool_seq"].tolist() == [-1, -1, 12, 13] and d["chunk_wid"].tolist() == [1, 2, 0, 0]
    assert d["chunk_start"].tolist() == [0, N // 4, N // 2, 3 * N // 4] and d["chunk_backend"][3] == "pool"
    assert d["chunk_rows_logp"].shape == (N,) and d["chunk_rows_rewards"][row] == np.float32(0.3)
    for k, v in before["model"].items():  # the parameters the update would have started from
        assert torch.equal(torch.as_tensor(d[f"model.{k}"]), v)
    assert torch.equal(torch.as_tensor(d["opt.state.0.exp_avg"]), before["opt"]["state"][0]["exp_avg"])


def test_non_finite_raw_observations_of_an_own_worker_are_found_before_the_statistics(tmp_path):
    raw = np.zeros((N, 8), np.float32)
    raw[5, 2], raw[N // 4 + 1, 0] = np.inf, np.nan  # rows of worker 1 and of the remote worker 2
    assert nonfinite_rows({"raw": np.zeros((N, 8), np.float32), "actions": np.zeros(N, np.int64)}) == (None, [])
    lines = []
    guard = NonFiniteGuard(tmp_path, lines.append)
    assert not guard.check(2, _chunks(2), {"raw": np.zeros((N, 8), np.float32)}) and not lines
    assert guard.check(2, _chunks(2), {"raw": raw})
    assert "NON-FINITE batch at u2 (raw)" in lines[0] and "own rows 2 (workers 1, 2), pool rows 0;" in lines[0]
    d = _load(tmp_path / "nonfinite-u2.npz")
    assert np.flatnonzero(d["bad_rows"]).tolist() == [5, N // 4 + 1] and np.isinf(d["raw"][5, 2])


# ============================================================================================== non-finite updates

def _poisoned(batch, sign: float):
    """The batch with one row whose old log-prob makes the probability ratio overflow (exp(1e4) = inf), in the third
    minibatch of the first epoch of an update with a generator seeded SEED; its advantage has the given sign."""
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(SEED)).numpy()
    rows = perm[2 * MB:3 * MB]
    row = int(rows[5])
    b = {k: v.copy() for k, v in batch.items()}
    b["logp"][row] = -1e4
    b["adv"][row] = 8.0 * sign
    a = b["adv"][rows]
    assert np.sign((b["adv"][row] - a.mean()) / a.std()) == sign  # the sign survives the minibatch normalisation
    assert nonfinite_rows(b) == (None, [])  # a finite batch
    return b, rows


@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize("sign,term", [(-1.0, "pg_loss"), (1.0, "grad_norm")])
def test_non_finite_minibatch_restores_parameters_and_optimiser(warm, sign, term):
    model, batch, _, _ = _setup()
    opt = None
    if warm:
        model, opt = _warm(model, batch)
    bad, rows = _poisoned(batch, sign)
    m0, o0 = _twin(model, opt)
    _ppo_update_base(m0, o0, bad, gen=torch.Generator().manual_seed(SEED), **ARGS)
    assert nonfinite_keys(m0.state_dict())  # before the guard: the whole model is NaN after this update

    m, o = _twin(model, opt)
    before = _state(m, o)
    steps, step = [], o.step
    o.step = lambda *a, **kw: (steps.append(1), step(*a, **kw))[1]
    stats = ppo_update(m, o, bad, gen=torch.Generator().manual_seed(SEED), **ARGS)
    nf = stats["nonfinite"]
    assert len(steps) == 2  # two minibatches had stepped before the bad one
    assert nf["term"] == term and nf["epoch"] == 0 and nf["minibatch"] == 2 and nf["rows"].tolist() == rows.tolist()
    assert not np.isfinite(nf[term]) and (term == "pg_loss" or np.isfinite(nf["pg_loss"]))
    assert _eq(_state(m, o), before)  # parameters and optimiser state of the start of the update
    assert all(p.grad is None for p in m.parameters())
    assert np.isfinite(stats["pg_loss"]) and stats["early_stop"] == 0.0  # the stats of the two undone minibatches

    # the restored learner continues exactly like one that never saw the bad batch
    m1, o1 = _twin(model, opt)
    for mm, oo in ((m, o), (m1, o1)):
        assert "nonfinite" not in ppo_update(mm, oo, batch, gen=torch.Generator().manual_seed(5), **ARGS)
    assert _eq(_state(m, o), _state(m1, o1)) and not _eq(m.state_dict(), before["model"])


def test_non_finite_update_through_the_guard_is_skipped_and_dumped(tmp_path):
    model, batch, values, _ = _setup()
    model, opt = _warm(model, batch)
    before = _state(model, opt)
    bad, rows = _poisoned(batch, -1.0)
    lines = []
    guard = NonFiniteGuard(tmp_path, lines.append)
    assert guard.update(9, _chunks(9), {"values": values}, model, opt, bad, gen=torch.Generator().manual_seed(SEED),
                        **ARGS) is None
    assert _eq(_state(model, opt), before) and guard.skipped == 1
    n_pool = int((rows >= N // 2).sum())
    assert "NON-FINITE update at u9 (pg_loss)" in lines[0] and "dump nonfinite-u9.npz" in lines[0]
    assert f"epoch 0 minibatch 2, its own rows {MB - n_pool}" in lines[0] and f"pool rows {n_pool} (task/seq" in lines[0]
    d = _load(tmp_path / "nonfinite-u9.npz")
    assert str(d["where"]) == "update" and str(d["term"]) == "pg_loss" and int(d["epoch"]) == 0 and int(d["minibatch"]) == 2
    assert d["minibatch_rows"].tolist() == rows.tolist() and np.isinf(d["term_pg_loss"]) and np.isfinite(d["term_v_loss"])
    np.testing.assert_array_equal(d["logp"], bad["logp"])
    np.testing.assert_array_equal(d["values"], values)
    for k, v in before["model"].items():  # the parameters the update started from
        assert torch.equal(torch.as_tensor(d[f"model.{k}"]), v)
    # a finite update afterwards ends the run of skips
    assert guard.update(10, _chunks(10), {"values": values}, model, opt, batch, **ARGS)["early_stop"] == 0.0
    assert guard.consecutive == 0 and guard.skipped == 1 and len(lines) == 1


def test_non_finite_parameters_after_the_last_minibatch_are_restored():
    model, batch, _, _ = _setup()
    opt = torch.optim.Adam(model.parameters(), lr=float("inf"))  # a finite loss and gradient, a non-finite step
    before = _state(model, opt)
    stats = ppo_update(model, opt, batch, 1, N, 0.2, 0.5, 0.01, 0.5)
    assert stats["nonfinite"]["term"] == "params" and stats["nonfinite"]["rows"] is None
    assert np.isfinite(stats["pg_loss"]) and np.isfinite(stats["grad_norm"])
    assert _eq(_state(model, opt), before) and not nonfinite_keys(model.state_dict())


# ============================================================================================== cap, stop, last check

def test_at_most_five_dumps_per_run_directory(tmp_path):
    assert MAX_DUMPS == 5
    lines = []
    guard = NonFiniteGuard(tmp_path, lines.append, max_consecutive=100)
    adv = np.full(N, np.nan, np.float32)
    for u in range(1, 8):
        assert guard.check(u, _chunks(u), {"adv": adv})
    assert sorted(f.name for f in tmp_path.iterdir()) == [f"nonfinite-u{u}.npz" for u in range(1, 6)]
    assert all(f"dump nonfinite-u{u}.npz" in lines[u - 1] for u in range(1, 6))
    assert all("no dump (the run has 5 already)" in s for s in lines[5:]) and guard.skipped == 7
    # the cap is the run directory's: a resumed learner does not start again
    resumed = NonFiniteGuard(tmp_path, lines.append)
    assert resumed.check(8, _chunks(8), {"adv": adv}) and len(list(tmp_path.iterdir())) == 5


def test_a_failed_dump_does_not_stop_the_learner(tmp_path):
    lines = []
    guard = NonFiniteGuard(tmp_path / "missing", lines.append)
    assert guard.check(1, _chunks(1), {"adv": np.full(N, np.nan, np.float32)})
    assert "dump failed (FileNotFoundError)" in lines[0] and guard.skipped == 1


def test_too_many_skips_in_a_row_stop_the_learner(tmp_path):
    assert MAX_CONSECUTIVE == 20
    model, batch, values, _ = _setup()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    lines = []
    guard = NonFiniteGuard(tmp_path, lines.append, max_dumps=1, max_consecutive=3)
    bad = dict(batch, adv=np.full(N, np.nan, np.float32))
    for u in range(1, 4):
        assert guard.update(u, _chunks(u), {"values": values}, model, opt, bad, **ARGS) is None
    assert guard.update(4, _chunks(4), {"values": values}, model, opt, batch, **ARGS) is not None  # finite: reset
    for u in range(5, 8):
        assert guard.update(u, _chunks(u), {"values": values}, model, opt, bad, **ARGS) is None
    with pytest.raises(NonFiniteError, match="4 non-finite updates in a row"):
        guard.update(8, _chunks(8), {"values": values}, model, opt, bad, **ARGS)
    assert guard.skipped == 7 and "in a row 4" in lines[-1]


def test_nonfinite_keys_names_what_must_not_be_saved_or_published():
    model, batch, _, _ = _setup()
    model, opt = _warm(model, batch)
    state = {"model": model.state_dict(), "opt": opt.state_dict(),
             "rms": {"mean": np.zeros(8), "var": np.ones(8), "count": 1e-4}}
    assert nonfinite_keys(state) == []
    with torch.no_grad():
        model.v[0].bias[3] = float("nan")
    opt.state_dict()["state"][0]["exp_avg_sq"][0, 0] = float("inf")
    state["rms"]["var"][2] = np.nan
    assert nonfinite_keys(state) == ["model.v.0.bias", "opt.state.0.exp_avg_sq", "rms.var"]


# ============================================================================================== the learner (fake run)

DRIVER = r"""
import functools, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
import ballrain.train.nonfinite as nonfinite
import ballrain.train.ppo as ppo
import train_m2
first_bad, last_bad, limit = (int(x) for x in sys.argv[2:5])
gae, normalize, update = ppo.gae, ppo.normalize, [1]  # the first batch only fits the statistics (no normalize call)
def counting_normalize(*a, **kw):  # the learner normalises each batch once: the number of the update it builds
    update[0] += 1
    return normalize(*a, **kw)
def bad_gae(*a, **kw):  # a NaN advantage in every chunk of the updates first_bad .. last_bad
    adv, ret = gae(*a, **kw)
    if first_bad <= update[0] <= last_bad:
        adv[0] = np.nan
    return adv, ret
ppo.normalize, ppo.gae = counting_normalize, bad_gae
nonfinite.NonFiniteGuard = functools.partial(nonfinite.NonFiniteGuard, max_consecutive=limit)
raise SystemExit(train_m2.main(sys.argv[5:]))
"""


def _learner(tmp_path, record, first_bad, last_bad, limit, *extra):
    """scripts/train_m2.py --fake (one worker, about six updates) with NaN advantages in some updates. Returns (exit
    code, output, run directory)."""
    import os
    env = dict(os.environ, BALLRAIN_RUNTIME=str(tmp_path / "rt"))
    args = [sys.executable, "-c", DRIVER, str(REPO / "scripts"), str(first_bad), str(last_bad), str(limit),
            "--run", "nan", "--fake", "--workers", "1", "--total-steps", "3000", "--batch-steps", "512", "--chunk", "64",
            "--minibatch", "256", "--epochs", "1", "--log-every", "1", "--ckpt-every", "1", "--record", str(record), *extra]
    p = subprocess.run(args, env=env, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                       encoding="utf-8", timeout=600)
    return p.returncode, p.stdout, tmp_path / "rt" / "m2-train" / "nan"


@pytest.fixture(scope="module")
def record(tmp_path_factory):
    if sys.platform != "win32":
        pytest.skip("ballrain.fakegame's shared memory needs Windows")
    from ballrain.pool import dev
    return dev.fake_record(tmp_path_factory.mktemp("nonfinite-record"))


def test_learner_skips_a_non_finite_update_and_goes_on(tmp_path, record):
    code, out, run = _learner(tmp_path, record, 4, 4, MAX_CONSECUTIVE)
    assert code == 0, out[-3000:]
    assert "NON-FINITE batch at u4 (adv): update skipped, parameters unchanged; own rows 8 (workers 0), pool rows 0; " \
           "dump nonfinite-u4.npz; skipped 1, in a row 1" in out
    assert [f.name for f in run.glob("nonfinite-u*.npz")] == ["nonfinite-u4.npz"]
    d = _load(run / "nonfinite-u4.npz")
    assert d["obs"].shape[0] == d["adv"].shape[0] == d["values"].shape[0] >= 512 and int(d["bad_rows"].sum()) == 8
    ck = {u: torch.load(run / f"ckpt_{u:06d}.pt", weights_only=False) for u in (3, 4, 5)}
    for k in ("model", "opt", "rms"):  # the skipped update left policy, optimiser and statistics alone; the next one trains
        assert _eq(ck[4][k], ck[3][k]), k
        assert not _eq(ck[5][k], ck[4][k]), k
    assert ck[3]["counters"]["nonfinite_skipped"] == 0 and ck[4]["counters"]["nonfinite_skipped"] == 1
    assert ck[4]["counters"]["update"] == 4 and ck[4]["counters"]["steps"] > ck[3]["counters"]["steps"]
    last = torch.load(run / "latest.pt", weights_only=False)
    assert last["counters"]["update"] >= 5 and last["counters"]["nonfinite_skipped"] == 1
    assert not nonfinite_keys({k: last[k] for k in ("model", "opt", "rms")})
    with open(run / "updates.csv", encoding="utf-8") as fh:
        rows = {int(r["update"]): r for r in csv.DictReader(fh)}
    assert [rows[u]["nonfinite_skipped"] for u in (3, 4, 5)] == ["0", "1", "1"]
    assert rows[3]["u_pg_loss"] and not rows[4]["u_pg_loss"] and rows[5]["u_pg_loss"]  # no losses for the skipped one
    logs = {u: next(s for s in out.splitlines() if f" u{u} steps " in s) for u in (3, 4, 5)}
    assert "nonfinite" not in logs[3] and logs[4].endswith(" nonfinite_skipped 1") and logs[5].endswith(" nonfinite_skipped 1")

    # a checkpoint with a non-finite parameter is refused before any worker starts
    last["model"]["pi.0.weight"][0, 0] = float("nan")
    torch.save(last, run / "latest.pt")
    code, out, _ = _learner(tmp_path, record, 0, 0, MAX_CONSECUTIVE, "--resume")
    assert code != 0 and "refusing to start from non-finite parameters (model.pi.0.weight)" in out
    assert "workers on" not in out


def test_learner_stops_after_too_many_skips_in_a_row(tmp_path, record):
    code, out, run = _learner(tmp_path, record, 2, 10 ** 6, 2)
    assert code != 0
    assert "STOPPED by the non-finite guard: 3 non-finite updates in a row (limit 2): the learner stops" in out
    assert "NonFiniteError" in out and "stopped at update 3" in out
    assert sorted(f.name for f in run.glob("nonfinite-u*.npz")) == ["nonfinite-u2.npz", "nonfinite-u3.npz", "nonfinite-u4.npz"]
    last = torch.load(run / "latest.pt", weights_only=False)  # the final checkpoint: the finite pre-update parameters
    assert last["counters"]["nonfinite_skipped"] == 3 and not nonfinite_keys({k: last[k] for k in ("model", "opt", "rms")})

"""Relaxed verification of the f7 reach block (docs/f7-native-port.md 7): fastreach is an approximate model, so the
client's port and the verifier's replay may differ on a few rows (a flag flipping at a threshold). On the fake world a
feature slice stands in for the reach block (verifier.reach_slice) and the VERIFIER's own rows are perturbed there, which
is exactly an honest flip: the client's values are self-consistent, only the replay's reach differs.
- within REACH_MAX_ROW_FRAC: the task passes, the rows are patched with the verifier's values for the learner;
- beyond the fraction, or beyond the steering bound: that chunk alone is rejected, the account is NOT flagged;
- everything outside the reach block stays strict (an observation mismatch there flags the account), and a strict
  mismatch next to reach deviations flags as before."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_pool_verifier import Player, pool, reseal, world  # noqa: E402,F401  (fixtures)

from ballrain.pool import learner_link as LL, protocol as P, rollout as R  # noqa: E402
from ballrain.pool.server_verify import read_reach_patch  # noqa: E402

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="ballrain.fakegame's shared memory needs Windows")

SL = slice(38, 46)  # edge_fan on the fake world: 8 continuous entries standing in for the 59 reach entries


def _perturb_replay(monkeypatch, rows: set, delta: float, sl=SL):
    """The verifier's own rows (task row index i) differ by delta on sl: what a fastreach flip looks like."""
    real_start, real_step = R.TaskRunner.start, R.TaskRunner.step

    def bump(o, i):
        if i in rows:
            o = np.array(o, np.float32, copy=True)
            o[sl] = o[sl] + np.float32(delta)
        return o

    def start(self):
        return bump(real_start(self), 0)

    def step(self, a, stop=None):
        st = real_step(self, a, stop=stop)
        st.next_obs = bump(st.next_obs, self.steps)
        return st
    monkeypatch.setattr(R.TaskRunner, "start", start)
    monkeypatch.setattr(R.TaskRunner, "step", step)


def _play_and_send(pool, tmp_path, user="eligible", min_chunks=2):
    pl = Player(pool, user, tmp_path)
    for _ in range(8):
        t = pl.task()
        chunks, _, _ = pl.play(t)
        if len(chunks) >= min_chunks:
            break
    for c in chunks:
        assert pl.send(c).status == 200
    return pl, t, chunks


def test_reach_deviations_within_the_fraction_pass_and_are_patched_for_the_learner(pool, tmp_path, monkeypatch):
    pl, t, chunks = _play_and_send(pool, tmp_path)
    pool.verifier.reach_slice, pool.verifier.reach_max_row_frac = SL, 0.05
    _perturb_replay(monkeypatch, {3, 10}, 0.01)  # 2 of 64 rows of chunk 0: 3 % <= 5 %; a tiny shift of the action
    assert pool.verifier.run_once() == 1
    rows = pool.store.chunks(t.task_id)
    assert all(r["verify"] == "passed" for r in rows) and not pool.store.is_suspect(rows[0]["forum_id"])
    patch = read_reach_patch(pool.cfg.pool_dir / "chunks" / rows[0]["file"])
    assert patch is not None and list(patch[0]) == [3, 10] and patch[2] == SL.start
    sub = P.decode_chunk(chunks[0])
    want = sub.arrays["obs"][[3, 10], SL].astype(np.float32) + np.float32(0.01)
    assert np.allclose(patch[1], want, atol=2e-3)  # the verifier's values (float16 copy + delta)
    m = LL.chunk_message(sub, 1, patch)
    assert np.allclose(m["raw"][[3, 10], SL], patch[1]) and np.array_equal(m["raw"][4], sub.arrays["obs"][4].astype(np.float32))
    assert read_reach_patch(pool.cfg.pool_dir / "chunks" / rows[1]["file"]) is None  # no deviant row there
    assert any("reach-deviant rows in chunks [0] (patched)" in x for x in pool.logs)


def test_too_many_reach_rows_reject_that_chunk_only_and_flag_nobody(pool, tmp_path, monkeypatch):
    pl, t, chunks = _play_and_send(pool, tmp_path)
    pool.verifier.reach_slice, pool.verifier.reach_max_row_frac = SL, 0.02
    _perturb_replay(monkeypatch, {2, 5, 8}, 0.01)  # 3 of 64 rows of chunk 0 > 2 %
    assert pool.verifier.run_once() == 1
    rows = pool.store.chunks(t.task_id)
    assert rows[0]["verify"] == "failed" and rows[0]["status"] == "rejected"
    assert json.loads(rows[0]["reason"])["code"] == "replay_mismatch"
    assert all(r["verify"] == "passed" and r["status"] == "released" for r in rows[1:])
    fid = rows[0]["forum_id"]
    assert not pool.store.is_suspect(fid) and pl.me()["token_state"] == "active"
    assert not list((pool.cfg.pool_dir / "alerts").glob("*.json"))
    assert not [r for r in pool.store.to_ingest(100) if r["seq"] == 0 and r["task_id"] == t.task_id]


def test_steering_bound_rejects_a_chunk_without_a_flag(pool, tmp_path, monkeypatch):
    """One deviant row whose reach values move the taken action's log-prob beyond the bound: that chunk is rejected,
    nobody is flagged; with a wide bound the same task passes."""
    pl, t, chunks = _play_and_send(pool, tmp_path)
    pool.verifier.reach_slice, pool.verifier.reach_max_row_frac = SL, 0.05
    pool.verifier.reach_max_logp_shift = 1e-9
    _perturb_replay(monkeypatch, {4}, 3.0)
    assert pool.verifier.run_once() == 1
    rows = pool.store.chunks(t.task_id)
    assert rows[0]["status"] == "rejected" and not pool.store.is_suspect(rows[0]["forum_id"])
    assert any("moves the taken action's log-prob" in x for x in pool.logs)


def test_outside_the_reach_block_stays_strict(pool, tmp_path, monkeypatch):
    """The same perturbation one entry outside the reach slice: a strict observation mismatch (suspect, alert)."""
    pl, t, chunks = _play_and_send(pool, tmp_path)
    pool.verifier.reach_slice, pool.verifier.reach_max_row_frac = SL, 0.5
    _perturb_replay(monkeypatch, {3}, 0.01, sl=slice(SL.stop, SL.stop + 1))
    assert pool.verifier.run_once() == 1
    rows = pool.store.chunks(t.task_id)
    bad = [r for r in rows if r["verify"] == "failed"]
    assert bad and json.loads(bad[0]["reason"])["code"] == "replay_mismatch"
    assert pool.store.is_suspect(rows[0]["forum_id"])
    rec = json.loads(sorted((pool.cfg.pool_dir / "alerts").glob("*-suspect-*.json"))[0].read_text(encoding="utf-8"))
    assert rec["detail"] == "observation" and rec["values"]["index"] == SL.stop


def test_a_strict_mismatch_next_to_reach_deviations_still_flags(pool, tmp_path, monkeypatch):
    pl = Player(pool, "eligible", tmp_path)
    t = pl.task()
    chunks, _, _ = pl.play(t)
    forged = reseal(chunks[0], rewards=lambda r: r + np.float32(0.25))  # a forged reward (strict) ...
    for c in [forged] + chunks[1:]:
        assert pl.send(c).status == 200
    pool.verifier.reach_slice, pool.verifier.reach_max_row_frac = SL, 0.5
    _perturb_replay(monkeypatch, {1, 2}, 0.01)  # ... next to tolerated reach rows
    assert pool.verifier.run_once() == 1
    rows = pool.store.chunks(t.task_id)
    assert json.loads(rows[0]["reason"])["code"] == "replay_reward_mismatch" and pool.store.is_suspect(rows[0]["forum_id"])


def test_replay_reports_deviant_rows_and_f6_is_unchanged(pool, tmp_path):
    """Without a reach slice (f6) the replay is exactly the strict one: no report, no tolerance."""
    pl, t, chunks = _play_and_send(pool, tmp_path, min_chunks=1)
    assert pool.verifier.reach_slice is None
    assert pool.verifier.run_once() == 1
    rows = pool.store.chunks(t.task_id)
    assert all(r["verify"] == "passed" for r in rows)
    assert all(read_reach_patch(pool.cfg.pool_dir / "chunks" / r["file"]) is None for r in rows)

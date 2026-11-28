"""Slow-split failure (ballrain.train.worker WorkerSpec.slow_fail): a split (hand-over or previous checkpoint -> this
checkpoint / LevelFinish) that lost more than lag_s against the start record ends the life as a failure. Off by
default: the worker's chunks are identical with it off and with a threshold never reached. No game is launched."""
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ballrain.train.worker import Worker, WorkerSpec, _Episode

REPO = Path(__file__).resolve().parents[1]


def _w(slow_fail, death=1.0, refund=0.4, paid=10.0):
    w = object.__new__(Worker)
    w.spec = SimpleNamespace(slow_fail=slow_fail)
    w.env = SimpleNamespace(reward=SimpleNamespace(cfg=SimpleNamespace(death_penalty=death, death_refund=refund),
                                                   sector_paid=paid))
    return w


def _ep(lag, ref):
    ep = object.__new__(_Episode)
    ep.split_lag, ep.split_ref = lag, ref
    return ep


def test_rule():
    w = _w({"lag_s": 3.0})
    assert w._slow_fail(_ep(3.01, 25.0)) == pytest.approx(-(1.0 + 0.4 * 10.0))  # a death's reward by default
    assert w._slow_fail(_ep(3.0, 25.0)) is None  # not more than lag_s
    assert w._slow_fail(_ep(-1.0, 25.0)) is None  # faster than the record
    assert w._slow_fail(_ep(None, None)) is None  # no split measured here
    assert _w({"lag_s": 3.0, "penalty": 50})._slow_fail(_ep(5.0, 25.0)) == -50.0
    assert _w({"lag_s": 3.0, "min_ref_s": 10})._slow_fail(_ep(5.0, 6.0)) is None  # short split: not judged
    assert _w({"lag_s": 3.0, "min_ref_s": 10})._slow_fail(_ep(5.0, 12.0)) is not None
    assert _w({})._slow_fail(_ep(99.0, 25.0)) is None  # off
    assert _w(None)._slow_fail(_ep(99.0, 25.0)) is None
    assert WorkerSpec(wid=0, copy="x", env=None, seed=0).slow_fail == {}


def test_split_lag_is_the_lives_own_stretch():
    """The lag is measured from the life's split start (hand-over SR, the record's time at the hand-over frame), so
    lag inherited before the hand-over is never counted; with split_k 0 the bonus stays 0."""
    w = object.__new__(Worker)
    w.spec = SimpleNamespace(split_k=0.0, slow_fail={"lag_s": 1.0}, split_extra=[], split_min_ref_s=5.0)
    w.env = SimpleNamespace(starts=SimpleNamespace(levelfinish_frame=500),
                            reward=SimpleNamespace(cfg=SimpleNamespace(death_penalty=1.0, death_refund=0.4), sector_paid=0.0))
    w._tas_cross = lambda sec: 30000.0  # the record crosses this checkpoint at IGT 30 s
    w._tas_cross_frame = lambda sec: 400
    ep = object.__new__(_Episode)
    ep.splits = []
    ep.seg = (50000.0, 20000.0, 100)  # the life's own SR 50 s at the hand-over (lag inherited: 30 s), record IGT 20 s
    b = w._split_bonus(ep, {"sector": 2, "sr_ms": 62000.0}, finished=False)
    assert b == 0.0
    assert ep.split_lag == pytest.approx(2.0) and ep.split_ref == pytest.approx(10.0)  # 12 s for the record's 10 s
    assert w._slow_fail(ep) is not None


def _run(record: Path, out: Path, n: int, extra: dict) -> dict:
    p = subprocess.run([sys.executable, str(REPO / "tests" / "slow_fail_run.py"), str(REPO), str(record), str(out), str(n),
                        json.dumps(extra)], capture_output=True, text=True, encoding="utf-8", timeout=900, cwd=str(REPO),
                       env={**os.environ, "BALLRAIN_TORCH_POLICY": "1"})
    assert p.returncode == 0, p.stderr[-3000:]
    return json.loads(p.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def record(tmp_path_factory):
    from ballrain.pool import dev
    return dev.fake_record(tmp_path_factory.mktemp("sf-record"))


@pytest.mark.skipif(sys.platform != "win32", reason="ballrain.fakegame's shared memory needs Windows")
def test_worker_off_identical_and_on_fails_slow_splits(record, tmp_path):
    n = 25
    off = _run(record, tmp_path / "off", n, {})
    never = _run(record, tmp_path / "never", n, {"slow_fail": {"lag_s": 1e9}})
    assert off == never  # on but never reached: every chunk and episode identical to off
    eps = [e for c in off["chunks"] for e in c["episodes"]]
    assert any(e["splits"] for e in eps), "no split measured: the comparison would prove nothing"
    always = _run(record, tmp_path / "always", n, {"slow_fail": {"lag_s": -1e9, "penalty": 7.0}})
    eps = [e for c in always["chunks"] for e in c["episodes"]]
    sf = [e for e in eps if e["end"] == "slow_fail"]
    assert sf and all(e["terms"].get("slow_fail") == -7.0 and not e["finish"] for e in sf)
    assert all(len(e["splits"]) == 1 for e in sf)  # each ended at its first split

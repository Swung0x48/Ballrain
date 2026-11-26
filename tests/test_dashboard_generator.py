"""The dashboard server and its generator child: the page's status never waits for a generation pass."""
import argparse
import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "dashboard"))
import server as SV  # noqa: E402

FAKE_CHILD = r'''
import json, os, sys, time
from pathlib import Path
out = Path(sys.argv[1])
def state(**gen):
    (out / "generator_state.json.tmp").write_text(json.dumps({"gen": gen, "level_dims": {"SR01_x": [100, 10]},
                                                              "map_frontier": ["r1", 50], "off": 12.0}))
    os.replace(out / "generator_state.json.tmp", out / "generator_state.json")
state(running=True, last_start=time.time(), generated_at=None, generated_ts=None, seconds=None, errors={})
time.sleep(1.0)  # a "pass" in another process: the server keeps answering meanwhile
(out / "summary_last.json").write_text(json.dumps({"data": {"curves": 1}, "generated_ts": 123}))
state(running=False, last_start=time.time(), generated_at="11-26 10:00:00", generated_ts=123, seconds=1.0, errors={})
for line in sys.stdin:
    if line.strip() == "wake":
        (out / "woke").write_text("1")
(out / "exited").write_text("1")
'''


class SimpleRun:
    """A stand-in run; run_level is patched to give it the record the fake child reports."""


def _level(run):
    return {"level": 1, "record": "SR01_x"}


def _args(**kw):
    a = dict(interval=600, min_refresh=0, train_machine="Mac mini", eval_machine="PC", source_labels="",
             tensorboard_url="", show_shortcut_line=False, no_game_assets=True)
    a.update(kw)
    return argparse.Namespace(**a)


def _wait(cond, timeout=15.0):
    t = time.time()
    while time.time() - t < timeout:
        if cond():
            return True
        time.sleep(0.05)
    return False


def test_status_single_flight(tmp_path, monkeypatch):
    monkeypatch.setattr(SV, "OUT", tmp_path)
    dash = SV.Dashboard(_args())
    calls = []

    def slow():
        calls.append(time.time())
        time.sleep(1.0)
        return {"n": len(calls), "status": {}}
    dash._compute_status = slow
    th = threading.Thread(target=dash.status)
    th.start()
    time.sleep(0.2)
    assert dash.status()["n"] == 1  # nothing cached yet: waits for the running computation, no second one
    th.join()
    assert len(calls) == 1
    dash._expire_status()
    for _ in range(5):  # expired: the last answer at once, one background computation
        t = time.time()
        assert dash.status()["n"] == 1
        assert time.time() - t < 0.5
    assert _wait(lambda: dash.status()["n"] == 2) and len(calls) == 2
    # a restarted server serves the last answer of the one before at once, and computes a new one meanwhile
    dash2 = SV.Dashboard(_args())
    dash2._compute_status = slow
    t = time.time()
    assert dash2.status()["n"] == 2 and time.time() - t < 0.5
    assert _wait(lambda: dash2.status()["n"] == 3)


def test_status_snapshot_ignores_garbage(tmp_path, monkeypatch):
    monkeypatch.setattr(SV, "OUT", tmp_path)
    (tmp_path / SV.STATUS_SNAPSHOT).write_text("[1, 2]")
    dash = SV.Dashboard(_args())
    dash._compute_status = lambda: {"status": {}, "fresh": True}
    assert dash.status()["fresh"] is True


def test_generator_child(tmp_path, monkeypatch):
    monkeypatch.setattr(SV, "OUT", tmp_path)
    monkeypatch.setattr(SV.D, "MAC_DIR", tmp_path / "none")
    monkeypatch.setattr(SV.D, "PC_DIR", tmp_path / "none")
    monkeypatch.setattr(SV.D, "run_level", _level)
    script = tmp_path / "fake_child.py"
    script.write_text(FAKE_CHILD)
    dash = SV.Dashboard(_args())
    dash.generator_cmd = lambda: [sys.executable, str(script), str(tmp_path)]
    sup = threading.Thread(target=dash.supervise, daemon=True)
    sup.start()
    try:
        assert _wait(lambda: dash.gen.get("running") is True)
        t = time.time()
        st = dash.status()
        assert time.time() - t < 5 and st["gen"]["running"] is True and "last_start" not in st["gen"]
        assert dash.level_of(SimpleRun()).n == 100 and dash.map_frontier == ("r1", 50) and dash.off == 12.0
        # the finished pass: its summary is taken, the status says so
        assert _wait(lambda: dash.summary.get("data") == {"curves": 1})
        # (the first request after the change may still get the answer before it, computed again in the background)
        assert _wait(lambda: dash.status()["gen"]["running"] is False and dash.status()["data"] == {"curves": 1}, 5)
        child = dash.child
        ok, _ = dash.request_refresh()
        assert ok and _wait(lambda: (tmp_path / "woke").exists())
        ok, _ = dash.request_refresh()
        assert ok  # min_refresh 0
    finally:
        dash.stop.set()
        sup.join(10)
    assert child.wait(10) == 0 and (tmp_path / "exited").exists()  # stdin closed: the child ends


def test_level_of_uses_reported_sizes(tmp_path, monkeypatch):
    monkeypatch.setattr(SV, "OUT", tmp_path)
    (tmp_path / "generator_state.json").write_text(json.dumps({"gen": {"running": True}, "level_dims": {"SR01_x": [100, 10]}}))
    monkeypatch.setattr(SV.D, "run_level", _level)
    dash = SV.Dashboard(_args())
    L = dash.level_of(SimpleRun())
    assert (L.n, L.first) == (100, 10)
    assert dash.gen["running"] is False  # the previous child's pass state is not taken at start

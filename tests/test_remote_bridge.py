"""RemoteBridge worker specs for a headless data source (e.g. the Windows host with the original 2002 physics_RT.dll):
the learner's template carries no headless paths, the bridge supplies its own HeadlessConfig and a data-source label."""
from types import SimpleNamespace

from ballrain.env import EnvConfig
from ballrain.train.remote import RemoteBridge
from ballrain.train.worker import _Episode


def _bridge(**kw):
    br = RemoteBridge(("127.0.0.1", 1), b"k", ["hlw-00", "hlw-01"], kw.pop("backend", "headless"),
                      lambda c: f"out/{c}", log=lambda s: None, **kw)
    tcfg = EnvConfig(level=1, backend="retail", headless=None, out_root=None)
    br.template = {"env": tcfg, "seed": 1, "run": "t", "worker": {"chunk": 16}}
    br.wid_base = 100
    return br


def test_headless_bridge_supplies_its_own_config_and_label():
    hl = {"binary": "x/SimTool.exe", "root": "x/game", "ball_order": "retail"}
    br = _bridge(label="headless2002", headless=hl)
    s = br._spec(1, "hlw-01", 0)
    assert s.env.backend == "headless" and s.env.headless == hl and s.env.copy == "hlw-01"
    assert s.env.out_root == "out/hlw-01"
    assert br.label == "headless2002"
    assert br.template["env"].headless is None  # the template is not modified
    s.env.headless["root"] = "y"
    assert br._spec(0, "hlw-00", 0).env.headless["root"] == "x/game"  # each worker gets its own copy


def test_retail_bridge_label_defaults_to_the_backend():
    br = _bridge(backend="retail")
    assert br.label == "retail"
    s = br._spec(0, "train-04", 0)
    assert s.env.backend == "retail" and s.env.headless is None


def test_episode_records_the_process_ball_order():
    def env(session):
        return SimpleNamespace(session=session, reward=SimpleNamespace(s=None), action_space=SimpleNamespace(n=9))
    ep = _Episode({"start": {"kind": "tas", "frame": 8800}, "sector": 2, "run_id": "r"},
                  env(SimpleNamespace(handle=SimpleNamespace(ball_order="retail:42"))), 3, None)
    assert ep.d["ball_order"] == "retail:42"
    ep2 = _Episode({"start": {}}, env(None), 3, None)
    assert ep2.d["ball_order"] is None


# ------------------------------------------------------------------------------------- quiet switch (--quiet-file)

def _quiet_bridge(tmp_path):
    br = _bridge(label="headless2002", headless={"binary": "b", "root": "r"}, quiet_file=tmp_path / "quiet",
                 quiet_check_s=5.0)
    calls = {"start": [], "stop": 0}
    br._start_worker = lambda i, name, restart: calls["start"].append((i, name, restart))
    br._stop_workers = lambda data_q, wait_s: calls.__setitem__("stop", calls["stop"] + 1)
    return br, calls


def test_quiet_file_stops_and_restarts_all_workers(tmp_path):
    br, calls = _quiet_bridge(tmp_path)
    restarts = [0, 0]
    br.check_quiet(None, restarts, now=100.0)
    assert not br.quiet and calls == {"start": [], "stop": 0}
    (tmp_path / "quiet").write_text("")
    br.check_quiet(None, restarts, now=102.0)  # within quiet_check_s of the last check: not looked at yet
    assert not br.quiet
    br.check_quiet(None, restarts, now=105.0)
    assert br.quiet and calls["stop"] == 1 and calls["start"] == []
    br.check_quiet(None, restarts, now=111.0)  # still quiet: nothing more to do
    assert calls["stop"] == 1
    (tmp_path / "quiet").unlink()
    br.check_quiet(None, restarts, now=117.0)
    assert not br.quiet and calls["start"] == [(0, "hlw-00", 1), (1, "hlw-01", 1)] and restarts == [1, 1]


def test_quiet_holds_back_parameters_and_keeps_the_newest(tmp_path):
    import queue as _q
    br, _ = _quiet_bridge(tmp_path)
    q = _q.Queue()
    br.ctrl_qs = [q]
    br.quiet = True
    msgs = iter([{"type": "params", "msg": {"v": 1}}, {"type": "params", "msg": {"v": 2}}])

    class Conn:
        def recv(self):
            try:
                return next(msgs)
            except StopIteration:
                br.stop = True
                raise EOFError
    br.conn = Conn()
    br._lost = lambda c: None
    br._recv_loop()
    assert q.empty() and br._last_params == {"v": 2} and br.counts["params"] == 2


def test_quiet_file_present_at_start_starts_no_workers(tmp_path):
    br, calls = _quiet_bridge(tmp_path)
    (tmp_path / "quiet").write_text("")
    br.connect = lambda timeout_s=None: True
    br.start_workers = lambda ctx, dq: calls["start"].append("all")
    seen = []

    def should_stop():
        seen.append(br.quiet)
        return len(seen) > 1
    import queue as _q
    br.run(None, _q.Queue(), should_stop=should_stop)
    assert seen[0] is True and calls["start"] == []


def _stubborn():  # a "worker" that ignores the stop command
    import time as _t
    while True:
        _t.sleep(0.2)


def _polite(q):
    while q.get().get("cmd") != "stop":
        pass


def test_stop_workers_stops_polite_and_kills_stubborn_processes(tmp_path):
    import multiprocessing as mp
    import time as _t
    ctx = mp.get_context("spawn")
    br = _bridge(label="x")
    q1, q2 = ctx.Queue(), ctx.Queue()
    p1, p2 = ctx.Process(target=_polite, args=(q1,), daemon=True), ctx.Process(target=_stubborn, daemon=True)
    p1.start(), p2.start()
    br.procs, br.ctrl_qs = [p1, p2], [q1, q2]
    t0 = _t.time()
    br._stop_workers(ctx.Queue(), wait_s=3.0)
    assert not p1.is_alive() and not p2.is_alive()
    assert _t.time() - t0 < 20

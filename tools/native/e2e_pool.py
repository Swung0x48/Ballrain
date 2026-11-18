"""End-to-end test of the native volunteer client against the Python pool (docs/native-client.md, M2).

Runs on this machine, all on loopback: the mock forum (tools/pool/mock_forum.py), the gateway (ballrain.pool.server_app)
with a real policy (a learner checkpoint, exported like the gateway's task source does), the real start record and
reference line, and the verifier replaying with the x87-soft headless engine; then the native client (nc_client) logs
in through the gateway's login flow (the browser part scripted here with the player's cookies, against the mock
forum's user "eligible"), runs tasks and posts its chunks; the verifier replays every task (probation: every chunk)
and must pass them all. A forged copy of one of the native chunks (reward changed, resealed) is then replayed offline
through the verifier's replay_task and must be caught.

    python tools/native/e2e_pool.py --dir runtime/nc/e2e --ckpt runtime/nc/ckpt/r7_latest.pt \\
        --record <dataset record dir> --engine <x87-soft BallanceMMOSimTool.exe> --game <Ballance folder> \\
        --client build/native-Release/nc_client.exe [--tasks 3] [--max-steps 3000]

Everything it writes stays under --dir (gateway state, chunks, the client's data directory). Exit code 0 = passed.
"""
from __future__ import annotations

import argparse
import html
import json
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "pool"))

from ballrain.pool import protocol as P, rollout as R, task as T, weights as W  # noqa: E402
from ballrain.pool.reasons import Reason  # noqa: E402
from ballrain.pool.server_app import PoolApp, make_server  # noqa: E402
from ballrain.pool.server_config import EngineRef, PoolConfig  # noqa: E402
from ballrain.pool.server_oauth import ForumClient, ForumConfig  # noqa: E402
from ballrain.pool.server_store import Store  # noqa: E402
from ballrain.pool.server_tasks import PolicySource  # noqa: E402
from ballrain.pool.server_verify import Verifier  # noqa: E402
from mock_forum import MockForum  # noqa: E402


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


def children_of(pid: int) -> list:
    """PIDs of the processes whose parent is pid (Windows: CIM)."""
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          f"(Get-CimInstance Win32_Process -Filter 'ParentProcessId={int(pid)}').ProcessId"],
                         capture_output=True, text=True).stdout
    return [int(x) for x in out.split() if x.strip().isdigit()]


def pid_alive(pid: int) -> bool:
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          f"[bool](Get-Process -Id {int(pid)} -ErrorAction SilentlyContinue)"], capture_output=True,
                         text=True).stdout
    return out.strip() == "True"


def get(url, cookie=None):
    """(status, headers, body) of a GET without following redirects (the player's browser)."""
    req = urllib.request.Request(url, headers={"Cookie": cookie} if cookie else {})
    try:
        r = urllib.request.build_opener(NoRedirect).open(req, timeout=20)
        return r.status, r.headers, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers, e.read()


def browser_login(browser_url: str, gateway: str, user: str):
    """The player's browser: the confirmation page, its continue link (with the page's cookie), the forum (mock user),
    the callback (with the OAuth cookie)."""
    st, h, body = get(browser_url)
    assert st == 200, (st, body[:200])
    href = html.unescape(body.decode().split('<a href="')[1].split('"')[0])
    ck = h["Set-Cookie"].split(";")[0]
    st, h, body = get(urllib.parse.urljoin(gateway + "/", href), ck)
    assert st == 302, (st, body[:200])
    oauth_ck = h["Set-Cookie"].split(";")[0]
    st, h2, body = get(h["Location"] + "&mock_user=" + user)
    assert st == 302, (st, body[:200])
    st, _, body = get(h2["Location"], oauth_ck)
    assert st == 200, (st, body[:300])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--record", required=True, help="dataset record directory (raw.npz, starts.npz, camera.npz, meta.json)")
    ap.add_argument("--path-dir", default=str(REPO / "runtime" / "refpaths-alt"))
    ap.add_argument("--engine", required=True)
    ap.add_argument("--game", required=True)
    ap.add_argument("--client", required=True)
    ap.add_argument("--client-engine", help="the client's engine binary (default --engine; the verifier always uses --engine)")
    ap.add_argument("--client-game", help="the client's Ballance folder (default --game; another path / install = the "
                    "machine-dependent parts of the level message differ)")
    ap.add_argument("--client-engine-env", action="append", default=[], help="NAME=VALUE for the client's engine")
    ap.add_argument("--tasks", type=int, default=3)
    ap.add_argument("--instances", type=int, default=1, help="the client's instances (engines at the same time)")
    ap.add_argument("--tui", help="also run the terminal UI (ballrain_pool --snapshot live:25) on the client's data")
    ap.add_argument("--kill-after", type=float, default=0.0,
                    help="first hard-kill the client this many seconds into a run (its engines must die with it: the job "
                         "object), then run it again: the killed run's queued chunks are posted and its tasks closed")
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--heartbeat-s", type=int, default=5, help="the gateway's heartbeat interval (5: heartbeats during tasks)")
    ap.add_argument("--env-draws", default="", help="JSON over the checkpoint's env_config start draws, e.g. "
                    '{"p_rest": 0.8} (format 3: at-rest / rehearsal starts at test rates)')
    ap.add_argument("--rollout", default="", help="JSON over the checkpoint's args (rollout fields), e.g. "
                    '{"p_early_respawn": 0.5}')
    a = ap.parse_args(argv)
    a.client, a.engine, a.game = (str(Path(x).resolve()) for x in (a.client, a.engine, a.game))
    a.client_engine = str(Path(a.client_engine).resolve()) if a.client_engine else a.engine
    a.client_game = str(Path(a.client_game).resolve()) if a.client_game else a.game
    d = Path(a.dir).resolve()
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    # the start record, copied (the gateway reads its record directory; nothing is written into the dataset)
    ds = d / "datasets"
    shutil.copytree(a.record, ds / Path(a.record).name)

    port = a.port
    if not port:  # a free loopback port (the redirect URI must name it before anything starts)
        import socket
        with socket.socket() as so:
            so.bind(("127.0.0.1", 0))
            port = so.getsockname()[1]
    gateway = f"http://127.0.0.1:{port}"
    redirect = f"{gateway}/pool/oauth/callback"
    forum = MockForum("ballrain-pool-e2e", "e2e-secret", redirect).serve("127.0.0.1", 0)
    engine = EngineRef(kind="x87soft", binary=str(Path(a.engine).resolve()), root=str(Path(a.game).resolve()))
    cfg = PoolConfig(pool_dir=d / "pool", public_base_url=gateway, forum=ForumConfig(
        base_url=forum.base_url, client_id="ballrain-pool-e2e", client_secret="e2e-secret", redirect_uri=redirect),
        run="r7", datasets_dir=ds, path_dirs=[Path(a.path_dir).resolve()], max_steps=a.max_steps, max_frames=60000,
        deadline_s=3600, heartbeat_s=a.heartbeat_s, engines={"x87soft": engine})
    from tools_pool_gateway import build  # noqa: E402  (tools/pool/gateway.py, imported by path below)
    app, store, policies, verifier = build(cfg)
    if a.client_engine != a.engine:  # the client runs another build of the x87-soft engine: allow its sha256 too
        engine.sha256.append(T.file_sha256(a.client_engine))
    srv = make_server(app, "127.0.0.1", port)
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.2}, daemon=True).start()
    log(f"gateway {gateway}, mock forum {forum.base_url}")
    state = W.load_trusted_checkpoint(a.ckpt)
    if a.env_draws:
        state["env_config"] = dict(state["env_config"], **json.loads(a.env_draws))
    if a.rollout:
        state["args"] = dict(state["args"], **json.loads(a.rollout))
    pub = policies.publish_state(state, source=Path(a.ckpt).name)
    log(f"policy {pub.policy_sha[:12]} (record {pub.record}, {len(pub.valid)} start frames)")
    issued = []
    new_task = policies.new_task

    def logged_task(*args, **kw):  # the tasks the gateway hands out (format 3: start kind, frontier)
        t = new_task(*args, **kw)
        issued.append(t)
        log(f"task {t.task_id[:8]}: k={t.start_frame} rest={t.rest} life_seed={'yes' if t.life_seed else 'no'} "
            f"respawn_frontier={t.respawn_frontier}")
        return t
    policies.new_task = logged_task

    client_dir = d / "client"
    # ---- login: the native client's login_start / login_poll, the browser part scripted
    p = subprocess.Popen([a.client, "login", "--gateway", gateway, "--data-dir", str(client_dir)], stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, encoding="utf-8")
    first = json.loads(p.stdout.readline())
    assert first["event"] == "login_started", first
    log(f"login code {first['user_code']}")
    browser_login(first["browser_url"], gateway, "eligible")
    out, err = p.communicate(timeout=120)
    assert p.returncode == 0, (out, err)
    log(f"login ok: {json.loads(out.splitlines()[-1])['user']['username']}")

    # ---- the native client plays tasks
    killed_tasks = []
    if a.kill_after > 0:
        run_cmd = [a.client, "run", "--gateway", gateway, "--data-dir", str(client_dir), "--ballance-dir", a.client_game,
                   "--engine", a.client_engine, "--instances", str(a.instances)]
        k = subprocess.Popen(run_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8")
        time.sleep(a.kill_after)
        engines = children_of(k.pid)
        k.kill()  # TerminateProcess: no cleanup in the client at all
        out_k = k.communicate(timeout=60)[0]
        killed_tasks = [json.loads(x)["task_id"] for x in out_k.splitlines()
                        if x.startswith("{") and json.loads(x)["event"] == "task"]
        time.sleep(3)
        alive = [pid for pid in engines if pid_alive(pid)]
        log(f"killed the client after {a.kill_after:.0f} s: {len(engines)} engines, {len(alive)} still alive after 3 s; "
            f"{len(killed_tasks)} tasks were running")
        if not engines or alive:
            return 1
    t0 = time.time()
    p = subprocess.run([a.client, "run", "--gateway", gateway, "--data-dir", str(client_dir), "--ballance-dir", a.client_game,
                        "--engine", a.client_engine, "--tasks", str(a.tasks), "--instances", str(a.instances)]
                       + [x for kv in a.client_engine_env for x in ("--engine-env", kv)], capture_output=True, text=True, encoding="utf-8",
                       timeout=3600)
    events = [json.loads(x) for x in p.stdout.splitlines() if x.strip().startswith("{")]
    (d / "client_events.jsonl").write_text(p.stdout, encoding="utf-8")
    (d / "client_stderr.txt").write_text(p.stderr, encoding="utf-8")
    if p.returncode != 0:
        log(f"client exit {p.returncode}: {p.stderr[-2000:]}")
        return 1
    ends = [e for e in events if e["event"] == "task_end"]
    beats = sum(1 for e in events if e["event"] == "heartbeat")
    other = [e for e in events if e["event"] in ("refused", "task_failed", "error")]
    chunks = [e for e in events if e["event"] == "chunk"]
    log(f"client: {len(ends)} tasks in {time.time() - t0:.0f} s: " + ", ".join(
        f"{e['end']} {e['steps']} steps {e['steps_per_s']:.0f}/s" for e in ends))
    bad = [c for c in chunks if c["status"] != "accepted"]
    log(f"client: {beats} heartbeats; refusals / failures: {other[:3]}")
    if bad or len(ends) != a.tasks or other or (a.heartbeat_s <= 10 and beats == 0):
        log(f"refused chunks / missing tasks: {bad[:3]}")
        return 1

    # ---- the verifier replays every task (probation: every chunk selected)
    t0 = time.time()
    for _ in range(4 * a.tasks):
        if verifier.run_once() == 0:
            break
    rows = [c for e in ends for c in store.chunks(e["task_id"])]
    verdict = {}
    for c in rows:
        verdict[c["verify"]] = verdict.get(c["verify"], 0) + 1
    st = store.stats(store.q("SELECT forum_id FROM accounts")[0]["forum_id"])
    log(f"verifier: {len(rows)} chunks {verdict} in {time.time() - t0:.0f} s; steps verified {st['steps_verified']}")
    for c in rows:
        if c["verify"] != "passed":
            log(f"  chunk {c['task_id'][:8]}/{c['seq']}: {c['verify']} {c['reason']}")
    if killed_tasks:
        states = {t: store.task(t)["state"] for t in killed_tasks}
        log(f"tasks of the killed run: {states}")
        if any(st == "open" for st in states.values()):
            log("a task of the killed run is still open (it should be closed by task_end after the restart)")
            return 1
    want = sum(e["steps"] for e in ends)  # (+ the killed run's prefixes, verified as well)
    ok = all(c["verify"] == "passed" for c in rows) and (st["steps_verified"] >= want if killed_tasks else st["steps_verified"] == want)

    # ---- a forged native chunk (reward changed by 0.25 at step 10, resealed) replayed offline: must be caught
    e = ends[0]
    task = T.Task.from_dict(json.loads(store.q("SELECT body FROM tasks WHERE task_id=?", (e["task_id"],))[0]["body"]))
    files = sorted((d / "pool" / "chunks" / task.task_id).glob("*.brpc"))
    decoded = [P.decode_chunk(f.read_bytes()) for f in files]
    c0 = decoded[0]
    h = {k: v for k, v in c0.header.items() if k not in ("arrays", "protocol", "type")}
    arr = {k: v.copy() for k, v in c0.arrays.items()}
    arr["rewards"][10] += np.float32(0.25)
    decoded[0] = P.decode_chunk(P.encode_chunk(h, arr))
    m, pol = verifier._policy(task.policy_sha)
    sdir, pdir = verifier._assets(task)
    env = T.build_env(task, m, T.LocalGame(engine="x87soft", binary=engine.binary, root=engine.root), sdir, pdir,
                      d / "replay-forged")
    runner = R.TaskRunner(env, task, m)
    try:
        mm = R.replay_task(runner, pol, decoded[:1])
    finally:
        runner.close()
    caught = mm is not None and mm.reason == Reason.REPLAY_REWARD_MISMATCH and mm.step == 10
    log(f"forged chunk: {mm.reason.value if mm else None} at step {mm.step if mm else None} -> "
        f"{'caught' if caught else 'NOT CAUGHT'}")
    tui_ok = True
    if a.tui:  # the terminal UI on the same data directory: the real controller for 25 s, then its main page
        t = subprocess.run([str(Path(a.tui).resolve()), "--data-dir", str(client_dir), "--snapshot", "live:25"],
                           capture_output=True, text=True, encoding="utf-8", timeout=600)
        import re
        page = re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", t.stdout)
        (d / "tui_snapshot.txt").write_text(page, encoding="utf-8")
        sys.stdout.buffer.write(page.encode("utf-8")), sys.stdout.flush()
        tui_ok = t.returncode == 0 and "采样中" in page and "合格玩家" in page
        log(f"tui: exit {t.returncode}, {'OK' if tui_ok else 'NOT OK'} (the page above)")
    srv.shutdown()
    forum.shutdown()
    store.close()
    result = {"tasks": ends, "heartbeats": beats, "chunks": len(rows), "verify": verdict, "steps_verified": st["steps_verified"],
              "forged_caught": caught, "tui": tui_ok if a.tui else None, "passed": bool(ok and caught and tui_ok)}
    (d / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    log("PASSED" if result["passed"] else "FAILED")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    import importlib.util

    spec = importlib.util.spec_from_file_location("tools_pool_gateway", REPO / "tools" / "pool" / "gateway.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules["tools_pool_gateway"] = mod
    raise SystemExit(main())

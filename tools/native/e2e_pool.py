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
    ap.add_argument("--client-engine-env", action="append", default=[], help="NAME=VALUE for the client's engine")
    ap.add_argument("--tasks", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--heartbeat-s", type=int, default=5, help="the gateway's heartbeat interval (5: heartbeats during tasks)")
    a = ap.parse_args(argv)
    a.client, a.engine, a.game = (str(Path(x).resolve()) for x in (a.client, a.engine, a.game))
    a.client_engine = str(Path(a.client_engine).resolve()) if a.client_engine else a.engine
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
    pub = policies.publish_state(W.load_trusted_checkpoint(a.ckpt), source=Path(a.ckpt).name)
    log(f"policy {pub.policy_sha[:12]} (record {pub.record}, {len(pub.valid)} start frames)")

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
    t0 = time.time()
    p = subprocess.run([a.client, "run", "--gateway", gateway, "--data-dir", str(client_dir), "--ballance-dir", a.game,
                        "--engine", a.client_engine, "--tasks", str(a.tasks)]
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
    ok = all(c["verify"] == "passed" for c in rows) and st["steps_verified"] == sum(e["steps"] for e in ends)

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
    srv.shutdown()
    forum.shutdown()
    store.close()
    result = {"tasks": ends, "heartbeats": beats, "chunks": len(rows), "verify": verdict, "steps_verified": st["steps_verified"],
              "forged_caught": caught, "passed": bool(ok and caught)}
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

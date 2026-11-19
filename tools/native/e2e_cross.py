"""End to end on a dev gateway: volunteer cross-checks (ballrain/pool/cross_check.py, docs/volunteer-compute.md 4.1).

Two accounts (the mock forum's "eligible" and "revoked_later"), two native clients of the same build playing at the
same time; every task is drawn (probation 0, task_sample 1) and every drawn task goes to a cross-check
(cross_fraction 1): one client's task is played again by the other, the two plays are compared, a difference goes to
our verifier (arbitration), and only its replay flags an account.

    python tools/native/e2e_cross.py --dir runtime/nc/e2e-cross --ckpt <ckpt> --record <record dir> --path-dir DIR
        --engine <x87-soft BallanceMMOSimTool.exe> --game <Ballance folder> --client <ballrain_coordinator.exe>
        [--tasks 6] [--forge none|chain|action|logp] [--branch-dir DIR]

--forge: the second account's chunk uploads are tampered with on their way to the gateway (a forged client):
  obs (an observation value of row 5, +0.5), action (the action of step 5), logp (its log-prob, 0.3 off). PASSED when
  (none) every cross-check agrees and nothing was flagged, or (forged) every task of the forging account that was
  compared went to arbitration and was rejected, the account is suspect, and the honest account is not.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "pool"))
sys.path.insert(0, str(REPO / "tools" / "native"))

from ballrain.pool import protocol as P, weights as W  # noqa: E402
from ballrain.pool.server_app import make_server  # noqa: E402
from ballrain.pool.server_config import EngineRef, PoolConfig  # noqa: E402
from ballrain.pool.server_oauth import ForumConfig  # noqa: E402
from e2e_pool import browser_login, log  # noqa: E402
from mock_forum import MockForum  # noqa: E402


def login(client, gateway, data_dir, user):
    p = subprocess.Popen([client, "login", "--gateway", gateway, "--data-dir", str(data_dir)], stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, encoding="utf-8")
    first = json.loads(p.stdout.readline())
    browser_login(first["browser_url"], gateway, user)
    out, err = p.communicate(timeout=120)
    if p.returncode != 0:
        raise SystemExit(f"login {user}: {err[-500:]}")
    return (Path(data_dir) / "token").read_text(encoding="utf-8").strip()


def forge_chunk(body: bytes, kind: str) -> bytes:
    c = P.decode_chunk(body)
    h = {k: v for k, v in c.header.items() if k not in ("arrays", "protocol", "type")}
    arr = {k: np.array(v, copy=True) for k, v in c.arrays.items()}
    i = min(5, int(h["steps"]) - 1)
    if kind == "obs":
        arr["obs"][i, 0] = np.float16(float(arr["obs"][i, 0]) + 0.5)
    elif kind == "action":
        arr["actions"][i] = (int(arr["actions"][i]) + 1) % int(h["n_act"])
    elif kind == "logp":
        arr["logp"][i] = np.float32(min(float(arr["logp"][i]) + 0.3, 0.0) if arr["logp"][i] < -0.3 else arr["logp"][i] - 0.3)
    return P.encode_chunk(h, arr)


def main() -> int:
    ap = argparse.ArgumentParser()
    for k in ("--dir", "--ckpt", "--record", "--path-dir", "--engine", "--game", "--client"):
        ap.add_argument(k, required=True)
    ap.add_argument("--branch-dir")
    ap.add_argument("--tasks", type=int, default=6, help="tasks per client (originals and replicas)")
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--forge", default="none", choices=("none", "obs", "action", "logp"))
    a = ap.parse_args()
    d = Path(a.dir).resolve()
    d.mkdir(parents=True, exist_ok=True)
    ds = Path(a.record).resolve().parent
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
        run="r8", datasets_dir=ds, path_dirs=[Path(a.path_dir).resolve()], max_steps=a.max_steps, max_frames=60000,
        deadline_s=3600, heartbeat_s=5, engines={"x87soft": engine}, max_instances=4,
        probation_chunks=0, task_sample=1.0, trusted_after=0, cross_fraction=1.0, cross_wait_s=120,
        cross_cooldown_s=0)
    import importlib.util
    spec = importlib.util.spec_from_file_location("tools_pool_gateway", REPO / "tools" / "pool" / "gateway.py")
    gw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gw)
    app, store, policies, verifier = gw.build(cfg)
    tokens = {}
    forged = []
    real = app.handle

    def handle(method, target, headers, body=b"", client=""):  # the second account's uploads, tampered with
        auth = {str(k).lower(): str(v) for k, v in (headers or {}).items()}.get("authorization", "")
        if a.forge != "none" and method == "POST" and target.startswith("/pool/chunk") and tokens.get("b") \
                and auth.endswith(tokens["b"]):
            try:
                c = P.decode_chunk(body)
                if int(c.header["seq"]) == 0:
                    body = forge_chunk(body, a.forge)
                    forged.append(c.header["task_id"])
            except Exception as e:  # noqa: BLE001
                log(f"forge: {e}")
        return real(method, target, headers, body, client)
    app.handle = handle
    srv = make_server(app, "127.0.0.1", port)
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.2}, daemon=True).start()
    state = W.load_trusted_checkpoint(a.ckpt)
    if a.branch_dir:
        from ballrain.branch_assets import read_bundle, validate
        lv = int(state["env_config"]["level"])
        state["branch_assets"] = read_bundle(lv, a.branch_dir)
        main_sha = state["env_config"]["features"]["path_sha256"]
        state["env_config"] = dict(state["env_config"], features=dict(
            state["env_config"]["features"], branch_sha256=validate(state["branch_assets"], lv, main_sha), branch_dir=None))
    pub = policies.publish_state(state, source=Path(a.ckpt).name)
    log(f"gateway {gateway}; policy {pub.policy_sha[:12]}; forge {a.forge}")
    tokens["a"] = login(a.client, gateway, d / "client-a", "eligible")
    tokens["b"] = login(a.client, gateway, d / "client-b", "revoked_later")
    procs = []
    for name in ("a", "b"):
        cmd = [a.client, "run", "--gateway", gateway, "--data-dir", str(d / f"client-{name}"), "--ballance-dir",
               str(Path(a.game).resolve()), "--engine", str(Path(a.engine).resolve()), "--tasks", str(a.tasks),
               "--instances", "1"]
        procs.append(subprocess.Popen(cmd, stdout=open(d / f"events-{name}.jsonl", "w", encoding="utf-8"),
                                      stderr=open(d / f"stderr-{name}.txt", "w", encoding="utf-8")))
    stop = threading.Event()

    def verify_loop():  # the gateway's verifier thread: cross-check decisions and arbitration replays
        while not stop.is_set():
            try:
                if verifier.run_once() == 0:
                    stop.wait(2)
            except Exception as e:  # noqa: BLE001
                log(f"verifier: {e}")
                stop.wait(2)
    vt = threading.Thread(target=verify_loop, daemon=True)
    vt.start()
    for p in procs:
        p.wait(timeout=3600)
    # every task closed: let the verifier decide and arbitrate
    t_end = time.time() + 900
    while time.time() < t_end:
        open_tasks = store.one("SELECT COUNT(*) AS n FROM tasks WHERE state IN ('open', 'closed')")["n"]
        pending = store.one("SELECT COUNT(*) AS n FROM cross_checks WHERE state IN ('waiting', 'issued')")["n"]
        if open_tasks == 0 and pending == 0:
            break
        time.sleep(3)
    stop.set()
    vt.join(timeout=30)
    crosses = store.q("SELECT * FROM cross_checks")
    by_state = {}
    for c in crosses:
        by_state[c["state"]] = by_state.get(c["state"], 0) + 1
    fa, fb = 1001, store.one("SELECT forum_id FROM accounts WHERE forum_id!=1001")["forum_id"]
    sus = {f: store.is_suspect(f) for f in (fa, fb)}
    tasks = store.q("SELECT task_id, forum_id, state, cross_role, replay FROM tasks")
    states = {}
    for t in tasks:
        k = f"{'a' if t['forum_id'] == fa else 'b'}:{t['state']}"
        states[k] = states.get(k, 0) + 1
    log(f"cross-checks {by_state}; tasks {states}; suspect a={sus[fa]} b={sus[fb]}; verifier replays {verifier.replays}")
    rows = {r["task_id"]: r for r in tasks}
    if a.forge == "none":
        ok = (by_state.get("passed", 0) >= 1 and not by_state.get("arbitrate") and not any(sus.values())
              and all(t["state"] == "done" for t in tasks))
    else:
        compared = [t for t in set(forged) if rows.get(t) and rows[t]["cross_role"] is not None
                    and any(c["state"] in ("arbitrate", "passed") and t in (c["task_a"], c["task_b"]) for c in crosses)]
        caught = [t for t in compared if rows[t]["state"] == "failed"]
        missed = [t[:8] for t in compared if rows[t]["state"] != "failed"]
        honest_bad = [t["task_id"][:8] for t in tasks if t["forum_id"] == fa and t["state"] == "failed"]
        log(f"forged {len(set(forged))} tasks, {len(compared)} compared, caught {len(caught)}, missed {missed}; "
            f"honest tasks rejected {honest_bad}")
        ok = bool(compared) and len(caught) == len(compared) and sus[fb] and not sus[fa] and not honest_bad
    srv.shutdown()
    forum.shutdown()
    store.close()
    log("PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

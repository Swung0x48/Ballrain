"""End-to-end test: gateway -> native client -> verifier -> learner link -> PPO learner, on a tiny run, all on loopback.

    python tools/native/e2e_pool_learner.py --dir <scratch dir> --ckpt <learner checkpoint (init weights)> \\
        --record <dataset record dir> --path-asset <L<nn>.npz of that checkpoint's features> \\
        --engine <x87-soft BallanceMMOSimTool.exe> --game <Ballance folder> --client build/native-Release/ballrain_coordinator.exe \\
        [--level 1] [--tasks 6] [--max-steps 600] [--batch-steps 512] [--extra "<train_m2 args>"]

1. A fresh learner (scripts/train_m2.py, its own BALLRAIN_RUNTIME under --dir, no workers of its own, --init-from the
   checkpoint, a start distribution with every task kind: reverse curriculum, rehearsal behind the frontier with
   level-load 0-0 entries, at-rest starts with and without a drive target, counter randomisation, p_early_respawn 0 with
   respawn_level_starts) listening on a loopback RemoteHub with a throwaway key, run with --pool-ingest.
2. The gateway in this process (mock forum, x87-soft verifier, two verifier workers) with the learner link: the
   learner's first parameters become the task policy (its live EnvConfig and curriculum).
3. The native client logs in (scripted browser) and plays --tasks tasks; the verifier replays them (probation: all).
4. The link feeds the released chunks to the learner, which trains on them; once every chunk is verified and fed the
   learner is stopped. Every chunk must pass, the learner must have trained >= 2 updates on pool steps only (all of
   them fed by the link) with no refused / dropped pool chunks.

Everything stays under --dir (and <repo>/runtime/e2e-pool-learner/ for the reference line, which the learner's
FeatureConfig needs repo-relative). Exit code 0 = passed; result.json has the numbers.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "pool"))
sys.path.insert(0, str(REPO / "tools" / "native"))


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def free_port() -> int:
    with socket.socket() as so:
        so.bind(("127.0.0.1", 0))
        return so.getsockname()[1]


def _gateway_module():
    spec = importlib.util.spec_from_file_location("tools_pool_gateway", REPO / "tools" / "pool" / "gateway.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules["tools_pool_gateway"] = mod
    return mod


def result_fed(fed) -> int:
    return sum(int(r["steps"]) for r in fed)


def learner_args(a, ck_args: dict, rt_rel_path: str, port: int) -> list:
    """train_m2 arguments: the checkpoint's architecture / features, a start distribution with every task kind."""
    g = lambda k, d=None: ck_args.get(k, d)  # noqa: E731
    args = ["--run", "e2e", "--level", str(a.level), "--backend", "headless", "--workers", "0", "--init-from", a.ckpt,
            "--total-steps", str(a.total_steps), "--batch-steps", str(a.batch_steps), "--chunk", "256",
            "--minibatch", "256", "--epochs", "2", "--log-every", "1", "--ckpt-every", "1000",
            "--remote-listen", f"127.0.0.1:{port}", "--remote-backend", "headless", "--pool-ingest",
            "--pool-max-lag", "1000", "--path-dir", rt_rel_path,
            "--hidden", str(g("hidden", "256,256")), "--activation", str(g("activation", "tanh")),
            "--action-repeat", str(g("action_repeat", 2)), "--gamma", str(g("gamma", 0.9985)), "--lam", str(g("lam", 0.975)),
            "--reward", str(g("reward", "{}")), "--split-k", str(g("split_k", 0.0)),
            "--stall-terminal", "--stall-s", "4", "--respawn-stall-s", "4", "--stall-compensate", "--stall-speed", "3",
            "--stall-max-s", "15", "--start-back", "1500", "--cu-step", "200", "--cu-band", "400",
            "--rehearse-starts", a.rehearse, "--p-rehearse", "0.3", "--rehearse-behind-frontier",
            "--rest-starts", a.rest, "--p-rest", "0.3", "--respawn-level-starts", "--p-early-respawn", "0",
            "--counter-rand", json.dumps({"lives": {"values": [0.0, 0.1, 0.2, 0.3], "p": [0.25] * 4},
                                          "sr_s": [0.0, 0.3], "points": [-0.05, 0.45]})]
    if g("world_actions"):
        args.append("--world-actions")
    for k, flag in (("far_waypoints", "--far-waypoints"), ("ref_speed_at", "--ref-speed-at")):
        if g(k):
            args += [flag, str(g(k))]
    if g("hs_block"):
        args.append("--hs-block")
    if a.record_name:
        args += ["--record", a.record_name]
    return args + (a.extra.split() if a.extra else [])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--record", required=True)
    ap.add_argument("--path-asset", required=True, help="the reference line (L<nn>.npz) the checkpoint's features used")
    ap.add_argument("--engine", required=True)
    ap.add_argument("--game", required=True)
    ap.add_argument("--client", required=True)
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--tasks", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--total-steps", type=int, default=10 ** 6, help="the learner's budget (the test stops it itself)")
    ap.add_argument("--batch-steps", type=int, default=512)
    ap.add_argument("--rehearse", default="0-0,0-0,17000-20000", help="--rehearse-starts of the tiny run")
    ap.add_argument("--rest", default="13450-13650,1840-1880@3.5:-170:1.5", help="--rest-starts of the tiny run")
    ap.add_argument("--extra", default="", help="more train_m2 arguments")
    a = ap.parse_args(argv)
    a.client, a.engine, a.game, a.ckpt = (str(Path(x).resolve()) for x in (a.client, a.engine, a.game, a.ckpt))
    from ballrain.train.remote import lower_priority
    lower_priority()  # this PC also runs training workers: the gateway, verifier engines and learner below normal
    d = Path(a.dir).resolve()
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    import torch
    ck = torch.load(a.ckpt, weights_only=False)
    a.record_name = Path(a.record).name
    # the learner's runtime: the start record in its tas-v1, the line inside the repo (FeatureConfig wants it relative)
    rt = d / "rt"
    shutil.copytree(a.record, rt / "datasets" / "tas-v1" / a.record_name)
    line_dir = REPO / "runtime" / "e2e-pool-learner" / "refpaths"
    line_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(a.path_asset, line_dir / f"L{a.level:02d}.npz")
    hub_port = free_port()
    key = secrets.token_urlsafe(24)  # throwaway hub key of this test (never stored)
    largs = learner_args(a, ck.get("args") or {}, "runtime/e2e-pool-learner/refpaths", hub_port)
    env = dict(os.environ, BALLRAIN_RUNTIME=str(rt), BALLRAIN_REMOTE_KEY=key, PYTHONUNBUFFERED="1")
    lout = open(d / "learner.log", "w", encoding="utf-8")
    learner = subprocess.Popen([sys.executable, str(REPO / "scripts" / "train_m2.py")] + largs, env=env, stdout=lout,
                               stderr=subprocess.STDOUT, cwd=str(REPO))
    log(f"learner pid {learner.pid} (hub 127.0.0.1:{hub_port}); log {d / 'learner.log'}")

    import e2e_pool as E  # noqa: E402  (the native e2e's mock-forum login helpers)
    from ballrain.pool import learner_link as LL, task as T
    from ballrain.pool.server_app import make_server
    from ballrain.pool.server_config import EngineRef, PoolConfig
    from ballrain.pool.server_oauth import ForumConfig
    from mock_forum import MockForum
    gw = _gateway_module()
    port = free_port()
    gateway = f"http://127.0.0.1:{port}"
    redirect = f"{gateway}/pool/oauth/callback"
    forum = MockForum("ballrain-pool-e2e", "e2e-secret", redirect).serve("127.0.0.1", 0)
    eng = EngineRef(kind="x87soft", binary=a.engine, root=a.game)
    cfg = PoolConfig(pool_dir=d / "pool", public_base_url=gateway, forum=ForumConfig(
        base_url=forum.base_url, client_id="ballrain-pool-e2e", client_secret="e2e-secret", redirect_uri=redirect),
        run="e2e", datasets_dir=rt / "datasets" / "tas-v1", path_dirs=[line_dir], max_steps=a.max_steps,
        max_frames=60000, deadline_s=3600, heartbeat_s=5, engines={"x87soft": eng}, verify_workers=2,
        learner_hub=f"127.0.0.1:{hub_port}", learner_key_env="E2E_HUB_KEY", learner_publish_every=1,
        learner_publish_min_s=0.0)
    app, store, policies, verifier = gw.build(cfg)
    link = LL.from_config(cfg, policies, store, log=log, environ={"E2E_HUB_KEY": key})
    srv = make_server(app, "127.0.0.1", port)
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.2}, daemon=True).start()
    result = {"passed": False}
    try:
        t0 = time.time()
        while not link.connect() and time.time() - t0 < 300:
            if learner.poll() is not None:
                raise RuntimeError(f"learner exited {learner.returncode} before its hub accepted the link")
            time.sleep(2)
        link.start_recv()
        while link.version is None and time.time() - t0 < 300:
            time.sleep(0.5)
        link.maybe_publish(force=True)
        pub = policies.current
        log(f"policy {pub.policy_sha[:12]} from learner update {link.version}; frontier {pub.curriculum.frontier}")
        link.start_thread()  # publishing + feeding from here on (connect() already done)
        verifier.start_threads(2)

        client_dir = d / "client"
        p = subprocess.Popen([a.client, "login", "--gateway", gateway, "--data-dir", str(client_dir)],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
        first = json.loads(p.stdout.readline())
        E.browser_login(first["browser_url"], gateway, "eligible")
        p.communicate(timeout=120)
        assert p.returncode == 0, "native login failed"
        log("native client logged in")
        t1 = time.time()
        p = subprocess.run([a.client, "run", "--gateway", gateway, "--data-dir", str(client_dir), "--ballance-dir",
                            a.game, "--engine", a.engine, "--tasks", str(a.tasks)], capture_output=True, text=True,
                           encoding="utf-8", timeout=3600)
        (d / "client_events.jsonl").write_text(p.stdout, encoding="utf-8")
        (d / "client_stderr.txt").write_text(p.stderr, encoding="utf-8")
        events = [json.loads(x) for x in p.stdout.splitlines() if x.strip().startswith("{")]
        ends = [e for e in events if e["event"] == "task_end"]
        bad = [e for e in events if e["event"] in ("refused", "task_failed", "error")
               or (e["event"] == "chunk" and e.get("status") != "accepted")]
        log(f"client exit {p.returncode}: {len(ends)} tasks in {time.time() - t1:.0f} s: "
            + ", ".join(f"{e['end']} {e['steps']}" for e in ends))
        tasks = [json.loads(r["body"]) for r in store.q("SELECT body FROM tasks")]
        kinds = sorted({("rest" if t.get("rest") else "tas", bool(t.get("start_frame")),
                         bool((t.get("rest") or {}).get("target"))) for t in tasks})  # task format 3
        log(f"task kinds (start_kind, k > 0, drive target): {kinds}")
        # wait for the verifier to decide every chunk and the link to feed every released one (or the learner to
        # finish its --total-steps), then stop the learner
        t2 = time.time()
        while time.time() - t2 < 1800 and learner.poll() is None:
            r = store.one("SELECT COUNT(*) AS n FROM chunks WHERE verify='pending' OR (status='released' AND ingested IS NULL)")
            if r["n"] == 0:
                break
            time.sleep(3)
        while time.time() - t2 < 1800:  # the verifier goes on even when the learner has finished
            if store.one("SELECT COUNT(*) AS n FROM chunks WHERE verify='pending'")["n"] == 0:
                break
            time.sleep(3)
        time.sleep(10)  # the learner's last batch
        if learner.poll() is None:
            (rt / "m2-train" / "e2e" / "STOP").write_text("e2e", encoding="utf-8")
            learner.wait(300)
        rows = store.q("SELECT verify, status, ingested, steps FROM chunks")
        verdict = {}
        for r in rows:
            verdict[r["verify"]] = verdict.get(r["verify"], 0) + 1
        fed = [r for r in rows if r["ingested"] is not None and r["ingested"] >= 0]
        ck2 = torch.load(rt / "m2-train" / "e2e" / "latest.pt", weights_only=False)
        c = ck2["counters"]
        result = {"tasks": ends, "task_kinds": kinds, "chunks": len(rows), "verify": verdict,
                  "fed_chunks": len(fed), "fed_steps": sum(r["steps"] for r in fed), "link": link.counts,
                  "learner": {k: c.get(k) for k in ("update", "steps", "steps_pool", "pool_refused", "pool_dropped")},
                  "client_exit": p.returncode, "client_problems": bad[:5]}
        # every chunk verified and passed; the learner trained on pool steps only (no workers of its own): at least
        # two updates; trained steps <= pool steps it accepted (the partial batch at the stop is not trained) <=
        # steps the link fed; nothing refused or dropped
        sp = int(c.get("steps_pool") or 0)
        trained = int(c.get("steps") or 0)
        ok = (p.returncode == 0 and not bad and len(ends) == a.tasks and verdict.get("passed", 0) == len(rows)
              and trained >= 2 * a.batch_steps and trained <= sp <= result_fed(fed)
              and int(c.get("update") or 0) >= 2 and not c.get("pool_refused") and not c.get("pool_dropped")
              and learner.returncode == 0)
        result["passed"] = bool(ok)
    finally:
        link.close()
        srv.shutdown()
        forum.shutdown()
        if learner.poll() is None:
            learner.kill()
        lout.close()
        store.close()
        (d / "result.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    log(json.dumps({k: v for k, v in result.items() if k != "tasks"}, default=str))
    log("PASSED" if result["passed"] else "FAILED")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

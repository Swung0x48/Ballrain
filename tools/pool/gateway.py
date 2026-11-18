"""Volunteer compute pool gateway + verifier (prototype; docs/pool-prototype.md, docs/volunteer-compute.md 11).

    # the real thing (forum OAuth, the run's newest checkpoint, 2002 / x87-soft replays), plain HTTP on 127.0.0.1:
    python tools/pool/gateway.py serve [--config runtime/pool/config.toml] [--port 8099] [--no-verify]

    # development: the toy level (ballrain.fakegame), a synthetic policy and the mock forum, all under --dir:
    python tools/pool/gateway.py dev-fake --dir <scratch dir> [--port 8099]

    # accounts (the verifier never revokes: a replay mismatch flags the account suspect and writes <pool>/alerts/)
    python tools/pool/gateway.py admin list
    python tools/pool/gateway.py admin stats
    python tools/pool/gateway.py admin suspects
    python tools/pool/gateway.py admin unflag --forum-id 123
    python tools/pool/gateway.py admin revoke --forum-id 123 [--note "说明"] [--list-updates] [--mark-updates]
    python tools/pool/gateway.py admin unrevoke --forum-id 123

Accepted chunks are stored under <pool>/chunks/. With a [learner] hub in the config (and its key in the environment
variable [learner] key_env) the gateway also runs the learner link (ballrain.pool.learner_link): it takes new task
policies from the learner's live parameters and feeds released chunks back to it (the learner uses them only with
--pool-ingest). Binds 127.0.0.1 only (HTTPS: the cloudflared tunnel's /pool ingress rule, docs/pool-prototype.md 12).
Logs carry no client addresses, tokens, OAuth codes or emails.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ballrain.pool.server_app import PoolApp, make_server  # noqa: E402
from ballrain.pool.server_config import EngineRef, PoolConfig, load_config  # noqa: E402
from ballrain.pool.server_oauth import ForumClient, ForumConfig  # noqa: E402
from ballrain.pool.server_store import Store  # noqa: E402
from ballrain.pool.server_tasks import PolicySource  # noqa: E402
from ballrain.pool.server_verify import Verifier  # noqa: E402

LOOPBACK = ("127.0.0.1", "::1", "localhost")


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def file_sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build(cfg: PoolConfig, launcher_factory=None, clock=time.time):
    """(app, store, policies, verifier) for a configuration; engine sha256 lists default to the configured binaries."""
    for e in cfg.engines.values():
        if not e.sha256 and e.binary and Path(e.binary).is_file():
            e.sha256 = [file_sha256(e.binary)]
    store = Store(Path(cfg.pool_dir) / "state.sqlite3", clock=clock)
    policies = PolicySource(cfg.pool_dir, cfg.run, cfg.datasets_dir, cfg.path_dirs, cfg.ckpt_dir, clock=clock)
    app = PoolApp(cfg, store, policies, ForumClient(cfg.forum), clock=clock, log=log)
    verifier = Verifier(store, cfg.pool_dir, cfg.engines, launcher_factory=launcher_factory, log=log,
                        reach_max_row_frac=getattr(cfg, "reach_max_row_frac", 0.02),
                        reach_max_logp_shift=getattr(cfg, "reach_max_logp_shift", 1.0))
    return app, store, policies, verifier


def run_server(app, policies, verifier, host: str, port: int, verify: bool, refresh_s: int | None):
    if host not in LOOPBACK:
        raise SystemExit(f"--host {host}: the prototype binds the loopback interface only")
    srv = make_server(app, host, port)
    stop = threading.Event()
    if refresh_s:
        def refresher():
            while not stop.wait(refresh_s):
                before = policies.current.policy_sha if policies.current else None
                cur = policies.refresh()
                if cur is not None and cur.policy_sha != before:
                    log(f"policy {cur.policy_sha[:12]} from {cur.source}")
                elif policies.last_error:
                    log(f"policy refresh: {policies.last_error}")
        threading.Thread(target=refresher, name="pool-policy", daemon=True).start()
    if verify:
        verifier.start_threads(getattr(app.cfg, "verify_workers", 1))
    log(f"gateway on http://{host}:{srv.server_address[1]}/pool/ (verifier {'on' if verify else 'off'})")
    try:
        srv.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        verifier.stop_event.set()
        srv.server_close()


def cmd_serve(a):
    cfg = load_config(a.config, a.pool_dir)
    if not cfg.forum.client_id or not cfg.forum.client_secret:
        log("warning: no forum client id / secret (runtime/pool/config.toml or BALLRAIN_POOL_FORUM_CLIENT_ID / _SECRET):"
            " logins are disabled")
    app, store, policies, verifier = build(cfg)
    from ballrain.pool.learner_link import from_config
    link = from_config(cfg, policies, store, log=log)
    if link is not None:  # policies come from the learner's live parameters, not from checkpoint copies
        link.start_thread()
        log("learner link: on (policies from the learner's parameters; released chunks fed back)")
    else:
        pub = policies.refresh()
        if pub is None:
            log(f"no policy published ({policies.last_error or 'no checkpoint'}): tasks answer no_task")
        else:
            log(f"policy {pub.policy_sha[:12]} from {pub.source} (record {pub.record}, {len(pub.valid)} start frames)")
    try:
        run_server(app, policies, verifier, a.host, a.port, not a.no_verify, None if link is not None else cfg.refresh_s)
    finally:
        if link is not None:
            link.close()


def dev_config(d: Path, port: int, forum_url: str, game_files: dict, engine_sha: str) -> PoolConfig:
    public = f"http://127.0.0.1:{port}"
    return PoolConfig(pool_dir=d / "pool", public_base_url=public,
                      forum=ForumConfig(base_url=forum_url, client_id="ballrain-pool-dev", client_secret="dev-secret",
                                        redirect_uri=f"{public}/pool/oauth/callback"),
                      run="dev", datasets_dir=d / "world" / "ds", path_dirs=[d / "world" / "refpaths"],
                      max_steps=2048, max_frames=20000, deadline_s=1800, verify_engines=("2002", "x87soft"),
                      engines={k: EngineRef(kind=k, sha256=[engine_sha], binary="fake", root="fake",
                                            game_files=game_files) for k in ("2002", "x87soft")})


def cmd_dev_fake(a):
    sys.path.insert(0, str(REPO / "tools" / "pool"))
    from mock_forum import MockForum
    from ballrain import fakegame
    from ballrain.pool import dev, protocol as P
    d = Path(a.dir).resolve()
    world = dev.fake_world(d / "world")
    forum = MockForum("ballrain-pool-dev", "dev-secret", f"http://127.0.0.1:{a.port}/pool/oauth/callback").serve(
        "127.0.0.1", a.forum_port)
    gf = {n: hashlib.sha256(f"dev:{n}".encode()).hexdigest() for n in P.GAME_FILE_LIST_2002}
    esha = hashlib.sha256(b"ballrain-pool dev fake engine").hexdigest()
    cfg = dev_config(d, a.port, forum.base_url, gf, esha)
    app, store, policies, verifier = build(cfg, launcher_factory=lambda: fakegame.FakeLauncher(backend="headless"))
    pub = policies.publish_state(world["state"])
    (d / "dev.json").write_text(json.dumps({"gateway": cfg.public_base_url, "forum": forum.base_url,
                                            "engine_sha256": esha, "game_files": gf, "policy_sha": pub.policy_sha,
                                            "mock_users": sorted(forum.users)}, indent=1), encoding="utf-8")
    log(f"dev-fake: mock forum {forum.base_url}; policy {pub.policy_sha[:12]}; client settings in {d / 'dev.json'}")
    run_server(app, policies, verifier, "127.0.0.1", a.port, True, None)
    forum.shutdown()


def cmd_admin(a):
    from ballrain.pool.reasons import Reason, payload
    cfg = load_config(a.config, a.pool_dir)
    store = Store(Path(cfg.pool_dir) / "state.sqlite3")
    if a.action == "list":
        for r in store.q("SELECT forum_id, username, nickname, leaderboard, revoked, revoke_reason FROM accounts "
                         "ORDER BY forum_id"):
            st = store.stats(r["forum_id"])
            rr = json.loads(r["revoke_reason"])["code"] if r["revoke_reason"] else ""
            print(f"{r['forum_id']:>8} {r['username']:<24} {r['nickname']:<16} board={r['leaderboard']} "
                  f"revoked={r['revoked']} {rr} verified={st['steps_verified']} pending={st['steps_pending']}")
    elif a.action == "stats":  # one screen for the coordinator: tasks, chunks, verification, flags, alerts
        q = store.q
        print("tasks by state / end:", {f"{r['state']}/{r['end_reason']}": r["n"] for r in
                                        q("SELECT state, end_reason, COUNT(*) AS n FROM tasks GROUP BY state, end_reason")})
        print("tasks replayed in full (drawn / probation / escalated):",
              q("SELECT COUNT(*) AS n FROM tasks WHERE replay=1")[0]["n"], "of", q("SELECT COUNT(*) AS n FROM tasks")[0]["n"])
        print("chunks by verify / status:", {f"{r['verify']}/{r['status']}": f"{r['n']} ({r['steps']} steps)" for r in
                                            q("SELECT verify, status, COUNT(*) AS n, SUM(steps) AS steps FROM chunks "
                                              "GROUP BY verify, status")})
        print("chunks flagged by the cheap check:", {r["flag"]: r["n"] for r in
                                                     q("SELECT flag, COUNT(*) AS n FROM chunks WHERE flag IS NOT NULL GROUP BY flag")})
        print("chunks fed to a learner:", q("SELECT COUNT(*) AS n FROM chunks WHERE ingested >= 0")[0]["n"])
        now = store.now()
        for h in (1, 24):
            r = q("SELECT COUNT(*) AS n, COALESCE(SUM(steps), 0) AS steps FROM chunks WHERE received >= ?", (now - 3600 * h,))[0]
            print(f"received in the last {h} h: {r['n']} chunks, {r['steps']} steps ({r['steps'] / (3600 * h):.1f} steps/s)")
        print("sessions active in the last 5 min:", {f"{r['engine_kind']} {r['client_version']} {r['platform']}": r["n"] for r in
                                                     q("SELECT engine_kind, client_version, platform, COUNT(*) AS n FROM sessions "
                                                       "WHERE last_seen >= ? GROUP BY engine_kind, client_version, platform",
                                                       (now - 300,))})
        print("suspect accounts:", q("SELECT COUNT(*) AS n FROM accounts WHERE suspect=1")[0]["n"],
              "revoked:", q("SELECT COUNT(*) AS n FROM accounts WHERE revoked=1")[0]["n"])
        al = sorted((Path(cfg.pool_dir) / "alerts").glob("*.json"))
        print(f"alerts: {len(al)}" + (f", newest {al[-1].name}" if al else ""))
    elif a.action == "suspects":  # accounts the verifier flagged (a replay mismatch); nothing was revoked
        for r in store.q("SELECT forum_id, suspect_reason, suspect_at FROM accounts WHERE suspect=1 ORDER BY suspect_at"):
            rec = json.loads(r["suspect_reason"] or "{}")
            lu = rec.get("learner_updates_unreplayed") or {}
            print(f"{r['forum_id']:>8} {rec.get('pseudonym', '')} since {r['suspect_at']} {rec.get('reason')} task "
                  f"{str(rec.get('task_id'))[:12]} chunk {rec.get('chunk')} step {rec.get('step')} {rec.get('detail')} "
                  f"{rec.get('values')}; unreplayed data in learner updates {lu.get('first')}..{lu.get('last')} "
                  f"({lu.get('steps')} steps)")
    elif a.action == "unflag":  # the coordinator / user cleared a suspect account: back to the normal spot checks
        if store.account(a.forum_id) is None:
            raise SystemExit(f"no account {a.forum_id}")
        store.unflag(a.forum_id)
        print(f"unflagged {a.forum_id}")
    elif a.action == "revoke":  # manual only (the verifier never revokes)
        acct = store.account(a.forum_id)
        if acct is None:
            raise SystemExit(f"no account {a.forum_id}")
        rec = json.loads(acct["suspect_reason"] or "{}") if acct["suspect"] else {}
        if rec and not a.note:  # a suspect account: the player sees which task / chunk / step did not replay
            reason = payload(Reason.TOKEN_REVOKED_REPLAY, task_id=rec.get("task_id"), chunk=rec.get("chunk"),
                             step=rec.get("step"))
        else:
            reason = payload(Reason.TOKEN_REVOKED, note=a.note or "无")
        store.revoke(a.forum_id, reason, quarantine=a.quarantine)
        lo, hi, n = store.ingested_updates(a.forum_id)
        ulo, uhi, un = store.ingested_updates(a.forum_id, unreplayed_only=True)
        print(f"revoked {a.forum_id}")
        if a.list_updates or a.mark_updates:
            print(f"  its fed data: learner updates {lo}..{hi} ({n} steps); of it never replayed: {ulo}..{uhi} "
                  f"({un} steps)")
        if a.mark_updates:  # a record for the training side's rollback decision (nothing is undone here)
            from ballrain.pool.server_verify import pseudonym, write_alert
            pth = write_alert(cfg.pool_dir, f"revoked-{pseudonym(a.forum_id)}",
                              {"kind": "revoked", "account": a.forum_id, "pseudonym": pseudonym(a.forum_id),
                               "reason": reason["code"], "learner_updates_all": {"first": lo, "last": hi, "steps": n},
                               "learner_updates_unreplayed": {"first": ulo, "last": uhi, "steps": un},
                               "suspect_record": rec or None})
            print(f"  marked: {pth}")
    elif a.action == "unrevoke":
        store.unrevoke(a.forum_id)
        print(f"unrevoked {a.forum_id} (quarantined chunks stay quarantined)")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Ballrain volunteer compute pool gateway (prototype)")
    ap.add_argument("--config", default=None, help="config.toml (default <pool dir>/config.toml)")
    ap.add_argument("--pool-dir", default=None, help="state directory (default runtime/pool)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8099)
    s.add_argument("--no-verify", action="store_true", help="do not run the verifier thread (run tools/pool/verifier.py)")
    s.set_defaults(fn=cmd_serve)
    f = sub.add_parser("dev-fake")
    f.add_argument("--dir", required=True)
    f.add_argument("--port", type=int, default=8099)
    f.add_argument("--forum-port", type=int, default=8098)
    f.set_defaults(fn=cmd_dev_fake)
    m = sub.add_parser("admin")
    m.add_argument("action", choices=("list", "stats", "suspects", "unflag", "revoke", "unrevoke"))
    m.add_argument("--forum-id", type=int)
    m.add_argument("--note", default="")
    m.add_argument("--no-quarantine", dest="quarantine", action="store_false")
    m.add_argument("--list-updates", action="store_true", help="revoke: print the learner updates its data reached")
    m.add_argument("--mark-updates", action="store_true",
                   help="revoke: also write an alerts/ record of those updates (for a rollback decision; nothing undone)")
    m.set_defaults(fn=cmd_admin)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()

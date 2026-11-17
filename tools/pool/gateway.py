"""Volunteer compute pool gateway + verifier (prototype; docs/pool-prototype.md, docs/volunteer-compute.md 11).

    # the real thing (forum OAuth, the run's newest checkpoint, 2002 / x87-soft replays), plain HTTP on 127.0.0.1:
    python tools/pool/gateway.py serve [--config runtime/pool/config.toml] [--port 8099] [--no-verify]

    # development: the toy level (ballrain.fakegame), a synthetic policy and the mock forum, all under --dir:
    python tools/pool/gateway.py dev-fake --dir <scratch dir> [--port 8099]

    # accounts
    python tools/pool/gateway.py admin list
    python tools/pool/gateway.py admin revoke --forum-id 123 --note "说明"
    python tools/pool/gateway.py admin unrevoke --forum-id 123

Accepted chunks are stored under <pool>/chunks/ and counted; NOTHING is sent to a learner. Binds 127.0.0.1 only (the
tunnel adds HTTPS later). Logs carry no client addresses, tokens, OAuth codes or emails.
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
    verifier = Verifier(store, cfg.pool_dir, cfg.engines, launcher_factory=launcher_factory, log=log)
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
        verifier.start_thread()
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
    pub = policies.refresh()
    if pub is None:
        log(f"no policy published ({policies.last_error or 'no checkpoint'}): tasks answer no_task")
    else:
        log(f"policy {pub.policy_sha[:12]} from {pub.source} (record {pub.record}, {len(pub.valid)} start frames)")
    run_server(app, policies, verifier, a.host, a.port, not a.no_verify, cfg.refresh_s)


def dev_config(d: Path, port: int, forum_url: str, game_files: dict, engine_sha: str) -> PoolConfig:
    public = f"http://127.0.0.1:{port}"
    return PoolConfig(pool_dir=d / "pool", public_base_url=public,
                      forum=ForumConfig(base_url=forum_url, client_id="ballrain-pool-dev", client_secret="dev-secret",
                                        redirect_uri=f"{public}/pool/oauth/callback"),
                      run="dev", datasets_dir=d / "world" / "ds", path_dirs=[d / "world" / "refpaths"],
                      max_steps=2048, max_frames=20000, deadline_s=1800,
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
    elif a.action == "revoke":
        if store.account(a.forum_id) is None:
            raise SystemExit(f"no account {a.forum_id}")
        store.revoke(a.forum_id, payload(Reason.TOKEN_REVOKED, note=a.note or "无"), quarantine=a.quarantine)
        print(f"revoked {a.forum_id}")
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
    m.add_argument("action", choices=("list", "revoke", "unrevoke"))
    m.add_argument("--forum-id", type=int)
    m.add_argument("--note", default="")
    m.add_argument("--no-quarantine", dest="quarantine", action="store_false")
    m.set_defaults(fn=cmd_admin)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()

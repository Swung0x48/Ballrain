"""Gateway configuration: runtime/pool/config.toml (not tracked) plus environment variables for the forum secret.

    [forum]
    base_url = "https://forum.ballance.top"
    client_id = "..."                  # or BALLRAIN_POOL_FORUM_CLIENT_ID
    client_secret = "..."              # or BALLRAIN_POOL_FORUM_CLIENT_SECRET
    # redirect_uri defaults to <pool.public_base_url>/pool/oauth/callback (must be registered exactly at the forum)

    [pool]
    public_base_url = "https://<public host>"   # the site origin (a trailing /pool is dropped); dev: http://127.0.0.1:<port>
    min_account_days = 30
    min_replies = 5
    max_instances = 8
    min_client_version = "0.3.0"     # 0.3: chunk format 3 (engine-state chain v3); older clients get client_too_old
    heartbeat_s = 30
    task_poll_s = 10
    trust_forwarded = true             # required when public_base_url is https (the tunnel): rate limits per
                                       # CF-Connecting-IP; the gateway itself listens on 127.0.0.1 only

    [policy]
    run = "r6"
    ckpt_dir = "runtime/m2-train-mac/r6"   # default runtime/m2-train-mac/<run>; read-only: the newest readable
                                           # ckpt_*.pt is exported
    datasets_dir = "runtime/datasets/tas-v1"
    path_dirs = ["benchmarks/refpaths", "runtime/refpaths-alt"]
    refresh_s = 900

    [task]
    max_steps = 8192
    max_frames = 60000
    deadline_s = 3600

    [verify]
    probation_chunks = 200     # an account's first chunks are all replayed
    sample_rate = 0.1          # after that, this fraction
    [engines.2002]             # what clients may run (sha256 of the engine binary) and what the verifier replays with
    sha256 = []                # default: the sha256 of binary
    binary = "runtime/hl-win/bin-x86p2002d/BallanceMMOSimTool.exe"
    root = "runtime/hl-win/game-min"          # reference game files (hash_game_files) and the replay's game folder
    [engines.x87soft]
    binary = "runtime/hl-win/bin-x86soft/BallanceMMOSimTool.exe"
    root = "runtime/hl-win/game-min"
    # without an [engines] section: these two (DEFAULT_ENGINES), where the files exist

Relative paths are relative to the repository root. Nothing here is ever sent to a client.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .server_oauth import ForumConfig

FORUM_URL = "https://forum.ballance.top"
ENV_CLIENT_ID = "BALLRAIN_POOL_FORUM_CLIENT_ID"
ENV_CLIENT_SECRET = "BALLRAIN_POOL_FORUM_CLIENT_SECRET"


DEFAULT_ENGINES = {"2002": ("runtime/hl-win/bin-x86p2002d/BallanceMMOSimTool.exe", "runtime/hl-win/game-min"),
                   "x87soft": ("runtime/hl-win/bin-x86soft/BallanceMMOSimTool.exe", "runtime/hl-win/game-min")}


def _repo() -> Path:
    from ballrain.instances import REPO_ROOT
    return Path(REPO_ROOT)


def _path(p, base: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else base / p


@dataclass
class EngineRef:
    kind: str
    sha256: list = field(default_factory=list)  # engine binaries clients may run
    binary: str = ""  # the verifier's engine
    root: str = ""  # the verifier's game folder (also the reference game-file hashes)
    game_files: dict = field(default_factory=dict)  # explicit reference hashes (else hash_game_files(root))


@dataclass
class PoolConfig:
    pool_dir: Path
    public_base_url: str
    forum: ForumConfig
    min_account_days: int = 30
    min_replies: int = 5
    max_instances: int = 8
    min_client_version: str = "0.3.0"  # chunk format 3
    heartbeat_s: int = 30
    task_poll_s: int = 10
    session_ttl_s: int = 900
    login_ttl_s: int = 600
    run: str = "r6"
    ckpt_dir: Path | None = None
    datasets_dir: Path | None = None
    path_dirs: list = field(default_factory=list)
    refresh_s: int = 900
    max_steps: int = 8192
    max_frames: int = 60000
    deadline_s: int = 3600
    probation_chunks: int = 200
    sample_rate: float = 0.1
    engines: dict = field(default_factory=dict)  # kind -> EngineRef
    max_pending_bytes_per_account: int = 512 * 1024 * 1024
    max_total_bytes: int = 20 * 1024 * 1024 * 1024
    trust_forwarded: bool = False  # take the client address from CF-Connecting-IP (behind the tunnel; rate limits only)
    max_concurrent: int = 32

    @property
    def secure_cookies(self) -> bool:
        return self.public_base_url.startswith("https://")


def load_config(path=None, pool_dir=None, environ=None) -> PoolConfig:
    environ = os.environ if environ is None else environ
    base = _repo()
    pool_dir = Path(pool_dir) if pool_dir else base / "runtime" / "pool"
    path = Path(path) if path else pool_dir / "config.toml"
    raw = tomllib.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    fo, po, pl, ta, ve = (raw.get(k, {}) for k in ("forum", "pool", "policy", "task", "verify"))
    public = str(po.get("public_base_url", "http://127.0.0.1:8099")).rstrip("/")
    if public.endswith("/pool"):  # the origin is wanted (paths are /pool/...)
        public = public[: -len("/pool")]
    forum = ForumConfig(base_url=str(fo.get("base_url", FORUM_URL)).rstrip("/"),
                        client_id=str(environ.get(ENV_CLIENT_ID) or fo.get("client_id", "")),
                        client_secret=str(environ.get(ENV_CLIENT_SECRET) or fo.get("client_secret", "")),
                        redirect_uri=str(fo.get("redirect_uri", f"{public}/pool/oauth/callback")))
    engines = {}
    if "engines" not in raw:
        for kind, (b, r) in DEFAULT_ENGINES.items():
            if (base / b).is_file() and (base / r / "base.cmo").is_file():
                engines[kind] = EngineRef(kind=kind, binary=str(base / b), root=str(base / r))
    for kind, e in (raw.get("engines") or {}).items():
        engines[str(kind)] = EngineRef(kind=str(kind), sha256=list(e.get("sha256") or []),
                                       binary=str(_path(e["binary"], base)) if e.get("binary") else "",
                                       root=str(_path(e["root"], base)) if e.get("root") else "",
                                       game_files=dict(e.get("game_files") or {}))
    if public.startswith("https://") and po.get("trust_forwarded") is not True:
        # behind the tunnel every request comes from 127.0.0.1: without the forwarded address one client's rate-limit
        # buckets would be everyone's (a single polling controller, or a stranger, could block all logins)
        raise ValueError("public_base_url is https (behind the tunnel): set trust_forwarded = true in [pool] of "
                         f"{path.name} so rate limits use the forwarded client address")
    return PoolConfig(
        pool_dir=pool_dir, public_base_url=public, forum=forum,
        min_account_days=int(po.get("min_account_days", 30)), min_replies=int(po.get("min_replies", 5)),
        max_instances=min(int(po.get("max_instances", 8)), 64), min_client_version=str(po.get("min_client_version", "0.3.0")),
        heartbeat_s=int(po.get("heartbeat_s", 30)), task_poll_s=int(po.get("task_poll_s", 10)),
        trust_forwarded=bool(po.get("trust_forwarded", False)),
        run=str(pl.get("run", "r6")),
        ckpt_dir=_path(pl.get("ckpt_dir") or f"runtime/m2-train-mac/{pl.get('run', 'r6')}", base),
        datasets_dir=_path(pl.get("datasets_dir", "runtime/datasets/tas-v1"), base),
        path_dirs=[_path(p, base) for p in pl.get("path_dirs", ["benchmarks/refpaths", "runtime/refpaths-alt"])],
        refresh_s=int(pl.get("refresh_s", 900)),
        max_steps=int(ta.get("max_steps", 8192)), max_frames=int(ta.get("max_frames", 60000)),
        deadline_s=int(ta.get("deadline_s", 3600)),
        probation_chunks=int(ve.get("probation_chunks", 200)), sample_rate=float(ve.get("sample_rate", 0.1)),
        engines=engines)


def version_tuple(v: str) -> tuple:
    out = []
    for part in str(v).replace("-", ".").replace("+", ".").split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        out.append(int(digits) if digits else 0)
    return tuple(out + [0] * (4 - len(out)))[:4]

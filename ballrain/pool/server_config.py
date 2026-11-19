"""Gateway configuration: runtime/pool/config.toml (not tracked) plus environment variables for the forum secret.

    [forum]
    base_url = "https://forum.ballance.top"
    client_id = "..."                  # or BALLRAIN_POOL_FORUM_CLIENT_ID
    client_secret = "..."              # or BALLRAIN_POOL_FORUM_CLIENT_SECRET
    # redirect_uri defaults to <pool.public_base_url>/pool/oauth/callback (must be registered exactly at the forum)
    # debug_log = true: append the forum's /oauth/token and /api/user replies to <pool>/forum-debug.jsonl, tokens
    # redacted, email addresses replaced by their shape (for working out what the forum sends; switch off after)

    [pool]
    public_base_url = "https://<public host>"   # the site origin (a trailing /pool is dropped); dev: http://127.0.0.1:<port>
    max_instances = 8
    min_client_version = "0.3.0"     # 0.3: chunk format 3 (engine-state chain v3), task format 3; older clients get
                                     # client_too_old
    heartbeat_s = 30
    task_poll_s = 10
    trust_forwarded = true             # required when public_base_url is https (the tunnel): rate limits per
                                       # CF-Connecting-IP; the gateway itself listens on 127.0.0.1 only

    [eligibility]                      # which forum accounts may take a pool token; checked in this order, each can
                                       # be switched off (server_oauth.EligibilityRules)
    account_age = true
    min_account_days = 30              # (old configs: [pool] min_account_days)
    unknown_join_date = "refuse"       # the forum gives no join date: "refuse" (a clear message) or "allow" (only
                                       # the age check is skipped)
    email_confirmed = true
    unknown_email = "refuse"           # the same when the forum gives no email-confirmed flag
    replies = true
    min_replies = 5                    # (old configs: [pool] min_replies)
    unknown_replies = "refuse"         # the same when the forum gives no reply count

    [guards]                           # upload bounds beyond the format's own checks (a broken client, not only NaN)
    obs_slack = 0.5                    # |obs| may exceed the policy's feature clip by this much
    max_abs_value = 1000.0             # |values| (the policy's value estimates)
    max_abs_reward = 1000.0            # |rewards| and |env_rewards| per step

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
    probation_chunks = 200     # an account's first tasks are all replayed in full (until it has this many chunks'
                               # steps replayed or waiting for the replay)
    task_sample = 0.15         # after that, each task is replayed in full with this probability (drawn at issue)
    escalate_tasks = 20        # a chunk failing the cheap policy-consistency check: its task and the account's next
                               # this many tasks are replayed in full
    reach_max_row_frac = 0.02  # f7: rows per chunk whose reach block may differ beyond OBS_TOL (docs/f7-native-port.md 7)
    reach_max_logp_shift = 1.0 # f7: on such a row, the most the submitted reach may move the taken action's log-prob
                               # (both set 2026-11-18 from the native port's measurements, f7-native-port.md 7.1)
    engines = ["x87soft"]      # the engine kinds tasks are issued for = what the verifier replays with (each task is
                               # replayed on its own engine kind; a hello with another kind gets engine_mismatch)
    workers = 1                # replays in parallel (one engine process each; below-normal priority)

    [learner]                  # the learner link (ballrain.pool.learner_link): off without hub
    hub = "<lan host>:<port>"  # the learner's RemoteHub (scripts/train_m2.py --remote-listen); LAN only
    key_env = "BALLRAIN_REMOTE_KEY"  # environment variable holding the hub's shared secret (never in this file)
    publish_every = 10         # publish the learner's weights as a new task policy every N updates
    publish_min_s = 20         # ... but at most once per this many seconds
    ingest = true              # feed released chunks back to the learner (it drops them unless --pool-ingest)
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

from .server_oauth import UNKNOWN_JOIN_DATE, EligibilityRules, ForumConfig

FORUM_URL = "https://forum.ballance.top"
MIN_CLIENT_VERSION = "0.3.0"  # chunk format 3 / task format 3 (load_config never goes below it)
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
    eligibility: "EligibilityRules | None" = None
    guard_obs_slack: float = 0.5  # [guards] (server_app._magnitudes)
    guard_max_abs_value: float = 1000.0
    guard_max_abs_reward: float = 1000.0  # [eligibility]; None: all checks with min_account_days / min_replies
    min_client_version: str = "0.3.0"  # chunk format 3, task format 3
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
    feature_module: Path | None = None  # [task] feature_module: the signed .wasm (feature_module_v1; modules.py)
    server_rewards: bool = False  # [task] server_rewards: reward_inputs_v1 sessions' rewards are the gateway's
    probation_chunks: int = 200
    task_sample: float = 0.15  # per-task full-replay probability after the probation (2026-11-18: per task, not per chunk)
    escalate_tasks: int = 20
    reach_max_row_frac: float = 0.02  # f7 (rollout.REACH_MAX_ROW_FRAC)
    reach_max_logp_shift: float = 1.0  # f7 (rollout.REACH_MAX_LOGP_SHIFT)
    engines: dict = field(default_factory=dict)  # kind -> EngineRef
    verify_engines: tuple = ("x87soft",)  # engine kinds tasks are issued for (the verifier replays each on its kind)
    verify_workers: int = 1
    learner_hub: str = ""  # HOST:PORT of the learner's RemoteHub ("" = no learner link)
    learner_key_env: str = "BALLRAIN_REMOTE_KEY"
    learner_publish_every: int = 10
    learner_publish_min_s: float = 20.0
    learner_ingest: bool = True
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
    fo, po, pl, ta, ve, le = (raw.get(k, {}) for k in ("forum", "pool", "policy", "task", "verify", "learner"))
    public = str(po.get("public_base_url", "http://127.0.0.1:8099")).rstrip("/")
    if public.endswith("/pool"):  # the origin is wanted (paths are /pool/...)
        public = public[: -len("/pool")]
    forum = ForumConfig(base_url=str(fo.get("base_url", FORUM_URL)).rstrip("/"),
                        client_id=str(environ.get(ENV_CLIENT_ID) or fo.get("client_id", "")),
                        client_secret=str(environ.get(ENV_CLIENT_SECRET) or fo.get("client_secret", "")),
                        redirect_uri=str(fo.get("redirect_uri", f"{public}/pool/oauth/callback")),
                        debug_log=str(pool_dir / "forum-debug.jsonl") if fo.get("debug_log") is True else None)
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
    el = raw.get("eligibility", {})
    unknown = set(el) - set(EligibilityRules.__dataclass_fields__)
    if unknown:  # a typo would silently keep (or drop) a check
        raise ValueError(f"[eligibility] in {path.name}: unknown keys {sorted(unknown)}")
    rules = EligibilityRules(
        account_age=bool(el.get("account_age", True)),
        min_account_days=max(0, int(el.get("min_account_days", po.get("min_account_days", 30)))),
        unknown_join_date=str(el.get("unknown_join_date", "refuse")),
        email_confirmed=bool(el.get("email_confirmed", True)),
        unknown_email=str(el.get("unknown_email", "refuse")),
        replies=bool(el.get("replies", True)),
        min_replies=max(0, int(el.get("min_replies", po.get("min_replies", 5)))),
        unknown_replies=str(el.get("unknown_replies", "refuse")))
    for k in ("unknown_join_date", "unknown_email", "unknown_replies"):
        if getattr(rules, k) not in UNKNOWN_JOIN_DATE:
            raise ValueError(f"[eligibility] {k} must be one of {list(UNKNOWN_JOIN_DATE)}")
    for k in ("account_age", "email_confirmed", "replies"):
        if k in el and not isinstance(el[k], bool):
            raise ValueError(f"[eligibility] {k} must be true or false")
    gu = raw.get("guards", {})
    unknown_g = set(gu) - {"obs_slack", "max_abs_value", "max_abs_reward"}
    if unknown_g:
        raise ValueError(f"[guards] in {path.name}: unknown keys {sorted(unknown_g)}")
    guards = {k: float(gu.get(k, d)) for k, d in (("obs_slack", 0.5), ("max_abs_value", 1000.0),
                                                   ("max_abs_reward", 1000.0))}
    if any(not v > 0 for v in guards.values()):
        raise ValueError("[guards] bounds must be positive")
    return PoolConfig(
        guard_obs_slack=guards["obs_slack"], guard_max_abs_value=guards["max_abs_value"],
        guard_max_abs_reward=guards["max_abs_reward"],
        pool_dir=pool_dir, public_base_url=public, forum=forum,
        min_account_days=rules.min_account_days, min_replies=rules.min_replies, eligibility=rules,
        max_instances=min(int(po.get("max_instances", 8)), 64),
        # never below MIN_CLIENT_VERSION (a config pinning an older client would hand format-3 tasks to it)
        min_client_version=max(str(po.get("min_client_version", MIN_CLIENT_VERSION)), MIN_CLIENT_VERSION, key=version_tuple),
        heartbeat_s=int(po.get("heartbeat_s", 30)), task_poll_s=int(po.get("task_poll_s", 10)),
        trust_forwarded=bool(po.get("trust_forwarded", False)),
        run=str(pl.get("run", "r6")),
        ckpt_dir=_path(pl.get("ckpt_dir") or f"runtime/m2-train-mac/{pl.get('run', 'r6')}", base),
        datasets_dir=_path(pl.get("datasets_dir", "runtime/datasets/tas-v1"), base),
        path_dirs=[_path(p, base) for p in pl.get("path_dirs", ["benchmarks/refpaths", "runtime/refpaths-alt"])],
        refresh_s=int(pl.get("refresh_s", 900)),
        max_steps=int(ta.get("max_steps", 8192)), max_frames=int(ta.get("max_frames", 60000)),
        deadline_s=int(ta.get("deadline_s", 3600)),
        feature_module=_path(ta["feature_module"], base) if ta.get("feature_module") else None,
        server_rewards=bool(ta.get("server_rewards", False)),
        probation_chunks=int(ve.get("probation_chunks", 200)),
        task_sample=min(max(float(ve.get("task_sample", 0.15)), 0.0), 1.0),
        escalate_tasks=max(0, int(ve.get("escalate_tasks", 20))),
        reach_max_row_frac=min(max(float(ve.get("reach_max_row_frac", 0.02)), 0.0), 1.0),
        reach_max_logp_shift=max(float(ve.get("reach_max_logp_shift", 1.0)), 0.0),
        engines=engines, verify_engines=tuple(str(k) for k in ve.get("engines", ["x87soft"])),
        verify_workers=max(1, int(ve.get("workers", 1))),
        learner_hub=str(le.get("hub", "")), learner_key_env=str(le.get("key_env", "BALLRAIN_REMOTE_KEY")),
        learner_publish_every=max(1, int(le.get("publish_every", 10))),
        learner_publish_min_s=float(le.get("publish_min_s", 20.0)), learner_ingest=bool(le.get("ingest", True)))


def version_tuple(v: str) -> tuple:
    out = []
    for part in str(v).replace("-", ".").replace("+", ".").split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        out.append(int(digits) if digits else 0)
    return tuple(out + [0] * (4 - len(out)))[:4]

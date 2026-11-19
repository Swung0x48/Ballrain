"""Pool tasks: what the server hands out, and how a client turns one into a BallanceEnv (docs/pool-prototype.md).

A task is ONE game process: the server picks the start (record, start frame k from the record's valid start
frames, after the run's exclude_starts) and the ball-order seed (headless "retail:<seed>"), so the client cannot
pick an easy start. The client plays the task's policy version from the hand-over until the process ends
(LevelFinish, GameOver, a truncation), the task's max_steps / max_frames are reached, or the player stops; deaths in
between respawn in the same process (per-life episodes), so the whole task replays from (k, seed, actions).

build_env_config() removes every client-side random draw of BallanceEnv: no curriculum, no at-rest / rehearsal draws
(p_rest = p_rehearse = 0), start_frame = k (checked to be a valid start frame, so it is not snapped to another),
ball order retail:<seed> explicit (the launcher draws nothing).

Start-state assets (served sha-addressed and immutable, GET /pool/assets/<sha256>):
- the starts pack (export_starts_pack / install_starts_pack): the record's frames (the prefix source), its valid start
  frames and expected hand-over states (starts.npz), the camera cache (camera.npz) and a minimal meta.json -- what
  ballrain.curriculum.TasStarts reads -- installed as a record directory the env's start_record points to;
- the reference line L<nn>.npz (install_path_asset), checked against FeatureConfig.path_sha256;
- a route branch (ballrain/branch.py, docs/l2-stone-branch.md), when the policy follows one: the branch pack
  (export_branch_pack / install_branch_pack) holds the learner's three files as sent (state["branch_assets"]: the
  branch line L<nn>.npz, its station description and rules L<nn>.json, the verified drive.npz), pinned by the
  learner's own features.branch_sha256 (ballrain.branch_assets) and handed to the env as EnvConfig.branch_assets. No
  component reads branch files of its own: the route is the learner's configuration.
Both are plain npz (np.load(allow_pickle=False), dtypes checked); the dataset's own meta.json (local paths) is not
shipped.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import protocol as P

STARTS_FORMAT = "ballrain-pool-starts/1"
RECORD_FRAME_DTYPE = np.dtype([("delta", "<f4"), ("keys", "<u4")])
_STARTS_ARRAYS = {  # name in the pack -> (dtype, ndim); starts.npz keys without the "starts_" prefix
    "record_frames": (RECORD_FRAME_DTYPE, 1),
    "starts_version": (np.dtype("<i8"), 0), "starts_frames_done": (np.dtype("<i8"), 1),
    "starts_sector": (np.dtype("<i8"), 1), "starts_sr_double": (np.dtype("<f8"), 1),
    "starts_points": (np.dtype("<i8"), 1), "starts_lives": (np.dtype("<i8"), 1), "starts_pos": (np.dtype("<f8"), 2),
    "starts_checkpoints": (np.dtype("<i8"), 1), "starts_levelfinish_frame": (np.dtype("<i8"), 0),
    "starts_sr_final": (np.dtype("<f8"), 0),
    "camera_quarter": (np.dtype("|i1"), 1), "camera_dev": (np.dtype("<f4"), 1),
    "meta_json": (np.dtype("|u1"), 1),
}
PATH_ARRAYS = {"points": (np.dtype("<f4"), 2), "sector": (np.dtype("|u1"), 1)}
# optional: the line's TAS times (ReferencePath.t_ms; without them the ref_speed features are zero, e.g. L02.npz)
PATH_OPTIONAL = {"t_ms": (np.dtype("<f8"), 1)}
MAX_ASSET_BYTES = 64 * 1024 * 1024


# the branch pack: the learner's files as raw bytes (uint8, 1-d); the manifest pins them by features.branch_sha256
# (ballrain.branch_assets: the sha256 of the three raw 32-byte digests in this order)
BRANCH_PACK_ARRAYS = ("line_npz", "desc_json", "drive_npz")
# multi-branch routes (wire v2, docs/pool-prototype.md 15): the v2 pack holds manifest_json (branches.json as sent) and,
# per member i in manifest order, m{i:02d}_line_npz / _desc_json / _drive_npz; member ids come from the manifest only
MAX_BRANCH_MEMBERS = 16


def branch_pack_v2_arrays(n: int) -> tuple:
    return ("manifest_json",) + tuple(f"m{i:02d}_{k}" for i in range(n) for k in BRANCH_PACK_ARRAYS)
DRIVE_ARRAYS = {"branch_sha256": (np.dtype("<U64"), 0), "record_sha256": (np.dtype("<U64"), 0),
                "prefix_frame": (np.dtype("<i8"), 0), "frames": (RECORD_FRAME_DTYPE, 1),
                "samples": (np.dtype("<f8"), 2)}


def _c(a) -> np.ndarray:
    """C-contiguous copy-if-needed that keeps 0-d arrays 0-d (np.ascontiguousarray makes them 1-d)."""
    return np.asarray(a, order="C")


# ============================================================================================== the task

@dataclass
class Task:
    task_id: str
    run: str
    policy_sha: str
    level: int
    record: str
    record_sha256: str
    start_frame: int
    ball_order_seed: int
    engine: str  # "2002" | "x87soft"
    assets: dict  # {"starts": sha256, "path": sha256}
    chunk_steps: int = 256
    reward_inputs: bool = False  # reward_inputs_v1: the chunks carry the reward inputs (server_rewards.py)
    max_steps: int = 4096
    max_frames: int = 60000
    deadline_s: int = 3600
    issued_at: int = 0
    kind: str = "sample"  # "sample" | "eval"
    greedy: bool = False
    sample_seed: int = 0  # the action draws of the task (ballrain.pool.sampling); server-side random
    rest: dict | None = None  # format 3: an at-rest start {"target": [x, z] | None, "max_frames": n} (server-drawn)
    life_seed: int = 0  # format 3: the per-life draws (counters, early respawns: ballrain.pool.counters)
    respawn_frontier: int | None = None  # format 3: the curriculum frontier for the early-respawn rule

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        """A received task (protocol.TASK schema; ProtocolError otherwise)."""
        return cls(**P.check(P.TASK, d, "task"))

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in P.TASK.required}
        if self.rest is not None:
            d["rest"] = self.rest
        if self.life_seed:
            d["life_seed"] = int(self.life_seed)
        if self.respawn_frontier is not None:
            d["respawn_frontier"] = int(self.respawn_frontier)
        if self.reward_inputs:
            d["reward_inputs"] = True
        return P.check(P.TASK, d, "task")

    @property
    def ball_order(self) -> str:
        return f"retail:{int(self.ball_order_seed)}"

    @property
    def deadline(self) -> int:
        return int(self.issued_at) + int(self.deadline_s)


def new_task_id() -> str:
    return secrets.token_hex(16)


def new_ball_order_seed() -> int:
    return secrets.randbits(32)


def new_sample_seed() -> int:
    from .sampling import SEED_BITS
    return secrets.randbits(SEED_BITS)


# ============================================================================================== local game

@dataclass
class LocalGame:
    """The player's machine: engine build and Ballance folder (never sent anywhere; only hashes are)."""
    engine: str  # "2002" (Windows, the player's own BuildingBlocks/physics_RT.dll) | "x87soft"
    binary: str  # the headless engine (SimTool) shipped with the controller
    root: str  # the player's Ballance folder
    socket: str = ""  # HeadlessConfig.socket ("" = the platform default)
    extra_env: dict = field(default_factory=dict)

    def headless(self, ball_order: str, engine_options: dict | None = None) -> dict:
        """HeadlessConfig fields: the policy manifest's engine_options (the run's --tas recipe), this machine's
        engine and folder, the task's ball order. extra_env may not set BALLRAIN_HL_BALL_ORDER (it would win)."""
        from ballrain.headless import parse_ball_order
        from .weights import HEADLESS_FIELDS
        opts = dict(engine_options or {})
        if set(opts) - set(HEADLESS_FIELDS):
            raise ValueError(f"engine options {sorted(set(opts) - set(HEADLESS_FIELDS))} not allowed")
        if any(k.upper() == "BALLRAIN_HL_BALL_ORDER" for k in self.extra_env):
            raise ValueError("extra_env may not set the ball order")
        return dict(opts, binary=str(self.binary), root=str(self.root), socket=self.socket,
                    ball_order=parse_ball_order(ball_order), extra_env=dict(self.extra_env))


def file_sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def hash_game_files(root, engine: str, max_files: int = 20000) -> dict:
    """{relative name: sha256} of the player's Ballance folder for hello.game_files (protocol.GAME_FILE_LIST; a
    directory entry is protocol.tree_digest of its regular files, symlinks are not followed and count as a
    mismatch). A missing entry maps to 64 zeros (the server names it in GAME_FILES_MISMATCH)."""
    if engine not in P.ENGINE_KINDS:
        raise ValueError(f"engine {engine!r}")
    root = Path(root)
    names = P.GAME_FILE_LIST_2002 if engine == "2002" else P.GAME_FILE_LIST
    out = {}
    for name in names:
        p = root / name.rstrip("/")
        if name.endswith("/"):
            if not p.is_dir() or p.is_symlink():
                out[name] = "0" * 64
                continue
            entries, n = {}, 0
            for dirpath, dirnames, filenames in os.walk(p, followlinks=False):
                dirnames[:] = sorted(d for d in dirnames if not (Path(dirpath) / d).is_symlink())
                for fn in sorted(filenames):
                    fp = Path(dirpath) / fn
                    rel = fp.relative_to(p).as_posix()
                    n += 1
                    if n > max_files:
                        raise ValueError(f"{name}: more than {max_files} files")
                    entries[rel] = "0" * 64 if fp.is_symlink() or not fp.is_file() else file_sha256(fp)
            out[name] = P.tree_digest(entries)
        else:
            out[name] = file_sha256(p) if p.is_file() and not p.is_symlink() else "0" * 64
    return out


# ============================================================================================== assets

def save_npz_deterministic(path, arrays: dict) -> None:
    """np.savez equivalent whose bytes depend only on the arrays (sorted names, fixed zip timestamps, no
    compression, write_array(allow_pickle=False)), so a re-export of the same content has the same file sha256.
    Readable with np.load(allow_pickle=False). Written to a temporary file and renamed."""
    import zipfile
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for name in sorted(arrays):
            a = np.asarray(arrays[name])
            if a.dtype.hasobject:
                raise ValueError(f"{name}: object arrays are not allowed")
            zi = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            zi.external_attr = 0o600 << 16
            with zf.open(zi, "w", force_zip64=True) as f:
                np.lib.format.write_array(f, a, allow_pickle=False)
    os.replace(tmp, path)


_write_npz = save_npz_deterministic


def export_starts_pack(record_dir, out_dir) -> tuple[str, Path]:
    """Server side: a dataset record directory (raw.npz, starts.npz, camera.npz, meta.json) -> <out_dir>/<sha>.npz,
    sha = the file's sha256. Only the record frames, the start-state cache, the camera cache and a few meta fields
    are copied."""
    from ballrain.curriculum import STARTS_VERSION
    d = Path(record_dir)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    with np.load(d / "raw.npz", allow_pickle=False) as z:
        frames = _c(z["record_frames"].astype(RECORD_FRAME_DTYPE))
    arrays = {"record_frames": frames}
    with np.load(d / "starts.npz", allow_pickle=False) as z:
        if int(z["version"]) != STARTS_VERSION:
            raise ValueError(f"{d}/starts.npz is version {int(z['version'])}, not {STARTS_VERSION}")
        for k in z.files:
            arrays[f"starts_{k}"] = _c(z[k].astype(_STARTS_ARRAYS[f"starts_{k}"][0]))
    with np.load(d / "camera.npz", allow_pickle=False) as z:
        arrays["camera_quarter"] = _c(z["quarter"].astype("|i1"))
        arrays["camera_dev"] = _c(z["dev"].astype("<f4"))
    m = {"format": STARTS_FORMAT, "record": d.name, "record_sha256": meta["record_sha256"], "level": int(meta["level"]),
         "dataset_version": int(meta.get("dataset_version", 1)),
         "levelfinish_frame": meta.get("levelfinish_frame"), "sr_double_final": meta.get("sr_double_final")}
    from .weights import assert_public
    assert_public(m)
    arrays["meta_json"] = np.frombuffer(P.dumps(m), np.uint8).copy()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / f".starts.{os.getpid()}.npz"
    _write_npz(tmp, arrays)
    sha = file_sha256(tmp)
    dst = out_dir / f"{sha}.npz"
    os.replace(tmp, dst)
    return sha, dst


def _load_checked(path: Path, spec: dict, expect_sha: str, extra_ok=(), optional: dict | None = None) -> dict:
    if path.stat().st_size > MAX_ASSET_BYTES:
        raise ValueError(f"{path.name}: larger than {MAX_ASSET_BYTES} bytes")
    if file_sha256(path) != expect_sha:
        raise ValueError(f"{path.name}: sha256 differs from {expect_sha[:12]}")
    out = {}
    with np.load(path, allow_pickle=False) as z:
        names = set(z.files)
        opt = set(optional or {})
        if not set(spec) <= names or names - set(spec) - opt - set(extra_ok):
            raise ValueError(f"{path.name}: arrays {sorted((names - opt - set(extra_ok)) ^ set(spec))} unexpected / missing")
        spec = dict(spec, **(optional or {}))
        for k in z.files:
            a = z[k]
            if a.dtype.hasobject:
                raise ValueError(f"{path.name}: {k} has an object dtype")
            if k in spec:
                dt, nd = spec[k]
                if a.dtype != dt or a.ndim != nd:
                    raise ValueError(f"{path.name}: {k} is {a.dtype} / {a.ndim}-d, expected {dt} / {nd}-d")
                if a.dtype.kind == "f" and not np.isfinite(a).all() and k not in ("starts_sr_final",):
                    raise ValueError(f"{path.name}: {k} has non-finite values")
            out[k] = _c(a)
    return out


def install_starts_pack(pack, cache_dir, expect_sha: str) -> Path:
    """Client side: check a downloaded starts pack (sha256, array names / dtypes, meta) and lay it out as a record
    directory <cache_dir>/starts/<sha>/<record>/ (raw.npz with record_frames only, starts.npz, camera.npz,
    meta.json) for TasStarts; returns that directory. Idempotent."""
    from ballrain.curriculum import STARTS_VERSION
    a = _load_checked(Path(pack), _STARTS_ARRAYS, expect_sha)
    m = P.loads_strict(a["meta_json"].tobytes(), 4096)
    m = P.check(P.Obj({"format": P.Const(STARTS_FORMAT), "record": P.Str(64, P.ID_RE, min_len=1), "record_sha256": P.SHA,
                       "level": P.Int(1, 13), "dataset_version": P.Int(1, 100),
                       "levelfinish_frame": P.Int(0, 10 ** 8, nullable=True),
                       "sr_double_final": P.Float(0, 1e9, nullable=True)}), m, "starts meta")
    if int(a["starts_version"]) != STARTS_VERSION:
        raise ValueError(f"starts pack version {int(a['starts_version'])}, this client reads {STARTS_VERSION}")
    n = len(a["starts_frames_done"])
    for k in ("starts_sector", "starts_sr_double", "starts_points", "starts_lives", "starts_checkpoints"):
        if len(a[k]) != n:
            raise ValueError(f"starts pack: {k} length")
    if a["starts_pos"].shape != (n, 3) or len(a["camera_quarter"]) != len(a["camera_dev"]):
        raise ValueError("starts pack: shapes")
    fd = a["starts_frames_done"]
    if n == 0 or (np.diff(fd) <= 0).any() or fd[0] < 0 or fd[-1] > len(a["record_frames"]):
        raise ValueError("starts pack: start frames not increasing inside the record")
    d = Path(cache_dir) / "starts" / expect_sha / m["record"]
    if (d / ".complete").is_file():
        return d
    d.mkdir(parents=True, exist_ok=True)
    _write_npz(d / "raw.npz", {"record_frames": a["record_frames"]})
    _write_npz(d / "starts.npz", {k[len("starts_"):]: v for k, v in a.items() if k.startswith("starts_")})
    _write_npz(d / "camera.npz", {"quarter": a["camera_quarter"], "dev": a["camera_dev"]})
    (d / "meta.json").write_text(json.dumps({"dataset_version": m["dataset_version"], "record": m["record"] + ".tas",
                                             "record_sha256": m["record_sha256"], "level": m["level"],
                                             "levelfinish_frame": m["levelfinish_frame"],
                                             "sr_double_final": m["sr_double_final"]}, indent=1), encoding="utf-8")
    (d / ".complete").write_text(expect_sha, encoding="utf-8")
    return d


def export_path_asset(asset, out_dir) -> tuple[str, Path]:
    """Server side: copy a reference line L<nn>.npz into the asset store as <sha>.npz (sha = its file sha256, the
    value FeatureConfig.path_sha256 holds)."""
    a = _load_checked(Path(asset), PATH_ARRAYS, file_sha256(asset), extra_ok=("meta",), optional=PATH_OPTIONAL)
    meta = a.get("meta")
    if meta is None or meta.dtype.kind != "U" or meta.ndim != 0:
        raise ValueError("reference line: meta must be a string")
    from .weights import assert_public
    assert_public(P.loads_strict(str(meta).encode("utf-8"), 16384))  # shipped as is: no local paths in it
    sha = file_sha256(asset)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{sha}.npz"
    if not dst.is_file():
        shutil.copyfile(asset, dst)
    return sha, dst


def find_path_asset(level: int, sha256: str, dirs) -> Path:
    """Server side: the L<nn>.npz among candidate directories whose file sha256 is the policy's
    FeatureConfig.path_sha256 (a run may train on another line than benchmarks/refpaths, e.g. one under runtime/).
    FileNotFoundError when none matches."""
    for d in dirs:
        p = Path(d) / f"L{int(level):02d}.npz"
        if p.is_file() and file_sha256(p) == sha256:
            return p
    raise FileNotFoundError(f"no L{int(level):02d}.npz with sha256 {sha256[:12]} in the given directories")


def install_path_asset(asset, cache_dir, level: int, expect_sha: str) -> Path:
    """Client side: check a downloaded reference line (sha256, arrays, meta level) and place it as
    <cache_dir>/refpaths/<sha>/L<nn>.npz; returns the directory (the FeatureConfig.path_dir to use)."""
    a = _load_checked(Path(asset), PATH_ARRAYS, expect_sha, extra_ok=("meta",), optional=PATH_OPTIONAL)
    meta = a.get("meta")
    if meta is None or meta.dtype.kind != "U" or meta.ndim != 0:
        raise ValueError("reference line: meta must be a string")
    m = P.loads_strict(str(meta).encode("utf-8"), 16384)
    if not isinstance(m, dict) or m.get("level") != int(level):
        raise ValueError(f"reference line is not level {level}")
    n = len(a["points"])
    if a["points"].shape != (n, 3) or len(a["sector"]) != n or ("t_ms" in a and len(a["t_ms"]) != n):
        raise ValueError("reference line: shapes")
    d = Path(cache_dir) / "refpaths" / expect_sha
    d.mkdir(parents=True, exist_ok=True)
    dst = d / f"L{int(level):02d}.npz"
    if not dst.is_file() or file_sha256(dst) != expect_sha:
        tmp = d / f".{dst.name}.{os.getpid()}.tmp"
        shutil.copyfile(asset, tmp)
        os.replace(tmp, dst)
    return d


def _raw(v, name: str) -> bytes:
    """A branch file as bytes: bytes / bytearray / memoryview, or a 1-d uint8 array (an npz round trip)."""
    if isinstance(v, (bytes, bytearray, memoryview)):
        b = bytes(v)
    elif isinstance(v, np.ndarray) and v.dtype == np.uint8 and v.ndim == 1:
        b = v.tobytes()
    else:
        raise ValueError(f"branch asset {name}: not bytes")
    if not b or len(b) > MAX_ASSET_BYTES:
        raise ValueError(f"branch asset {name}: empty or larger than {MAX_ASSET_BYTES} bytes")
    return b


def _npz_arrays(b: bytes, what: str) -> dict:
    import io
    with np.load(io.BytesIO(b), allow_pickle=False) as z:
        out = {k: z[k] for k in z.files}
    for k, a in out.items():
        if a.dtype.hasobject:
            raise ValueError(f"{what}: {k} has an object dtype")
    return out


def _check_branch_line_and_drive(line: bytes, drive: bytes, level: int) -> None:
    """The structure the native client relies on beyond ballrain.branch_assets.validate (array names, dtypes, shapes,
    finite values, no local paths in the line's meta)."""
    from .weights import assert_public
    a = _npz_arrays(line, "branch line")
    spec = dict(PATH_ARRAYS, **PATH_OPTIONAL)
    if set(a) != set(spec) | {"meta"}:
        raise ValueError(f"branch line: arrays {sorted(set(a) ^ (set(spec) | {'meta'}))} unexpected / missing")
    for k, (dt, nd) in spec.items():
        if a[k].dtype != dt or a[k].ndim != nd or (a[k].dtype.kind == "f" and not np.isfinite(a[k]).all()):
            raise ValueError(f"branch line: {k}")
    n = len(a["points"])
    if a["points"].shape != (n, 3) or len(a["sector"]) != n or len(a["t_ms"]) != n or a["meta"].dtype.kind != "U" \
            or a["meta"].ndim != 0:
        raise ValueError("branch line: shapes")
    m = P.loads_strict(str(a["meta"]).encode("utf-8"), 16384)
    if not isinstance(m, dict) or m.get("level") != int(level):
        raise ValueError(f"branch line is not level {level}")
    assert_public(m)
    z = _npz_arrays(drive, "branch drive")
    if set(z) != set(DRIVE_ARRAYS):
        raise ValueError(f"branch drive: arrays {sorted(set(z) ^ set(DRIVE_ARRAYS))} unexpected / missing")
    for k, (dt, nd) in DRIVE_ARRAYS.items():
        if z[k].dtype != dt or z[k].ndim != nd:
            raise ValueError(f"branch drive: {k} is {z[k].dtype} / {z[k].ndim}-d")
    sm = z["samples"]
    if int(z["prefix_frame"]) <= 0 or sm.shape[1:] != (2,) or not len(sm) or not np.isfinite(sm).all() or \
            not ((sm[:, 0] >= 0) & (sm[:, 0] <= len(z["frames"]))).all():
        raise ValueError("branch drive: prefix frame / samples")


def branch_bundle(ba: dict, level: int, main_sha256: str | None) -> tuple[dict, str, bool]:
    """The learner's state["branch_assets"] ({level, line_npz, desc_json, drive_npz} raw bytes, sha256 optional: it is
    recomputed and, when given, must agree) checked as the learner's workers check it (ballrain.branch_assets.validate:
    level, digests, the main line pairing, the rules) plus the native client's structure checks. Returns (bundle with
    sha256, features.branch_sha256, whether the rules differ from the v1 defaults).

    A v2 envelope ({version: 2, level, manifest: bytes, members: [v1 envelope, ...]}, docs/l2-branches.md) is checked
    member by member the same way, then as a whole (the manifest's rules, the v2 pin); rules_changed is whether any
    member's rules differ (branch_assets.uses_custom_rules)."""
    from ballrain.branch_assets import filenames, validate, bundle_rules, uses_custom_rules
    from ballrain.branch_rules import DEFAULT_RULES
    from ballrain.path import RouteError
    from .weights import assert_public
    if not isinstance(ba, dict) or ba.get("level") != int(level):
        raise ValueError(f"the learner's branch assets are not for level {level}")
    if "version" in ba:
        if ba["version"] != 2:
            raise ValueError(f"the learner's branch assets: unsupported version {ba['version']!r}")
        raw = _raw(ba.get("manifest"), "manifest")
        members = ba.get("members")
        if not isinstance(members, list) or not 1 <= len(members) <= MAX_BRANCH_MEMBERS:
            raise ValueError(f"the learner's branch assets: 1..{MAX_BRANCH_MEMBERS} members expected")
        if any(not isinstance(c, dict) or "version" in c for c in members):
            raise ValueError("the learner's branch assets: a member is not a single-line bundle")
        checked = [branch_bundle(c, level, main_sha256)[0] for c in members]
        bundle = dict(version=2, level=int(level), manifest=raw, members=checked)
        try:
            key = validate(bundle, int(level), main_sha256)  # the manifest's rules, member count, the v2 pin
        except (RouteError, ValueError, KeyError) as e:
            raise ValueError(f"branch assets: {e}") from None
        assert_public(json.loads(raw))
        return bundle, key, uses_custom_rules(bundle)
    blobs = {k: _raw(ba.get(k), k) for k in filenames(level)}
    digests = {k: hashlib.sha256(v).hexdigest() for k, v in blobs.items()}
    if ba.get("sha256") not in (None, digests):
        raise ValueError("the learner's branch assets: sha256 differs from the bytes")
    bundle = dict(level=int(level), **blobs, sha256=digests)
    try:
        key = validate(bundle, int(level), main_sha256)
        rules = bundle_rules(bundle)
    except (RouteError, ValueError, KeyError) as e:
        raise ValueError(f"branch assets: {e}") from None
    assert_public(json.loads(blobs["desc_json"]))
    _check_branch_line_and_drive(blobs["line_npz"], blobs["drive_npz"], level)
    return bundle, key, rules != DEFAULT_RULES


def export_branch_pack(ba: dict, level: int, out_dir, main_sha256: str | None) -> tuple[str, Path]:
    """Server side: the learner's branch bundle -> <out_dir>/<sha>.npz (sha = the file's sha256), the three files'
    bytes as uint8 arrays (deterministic npz: the same bundle gives the same sha)."""
    bundle, _, _ = branch_bundle(ba, level, main_sha256)
    if bundle.get("version") == 2:
        arrays = {"manifest_json": np.frombuffer(bundle["manifest"], np.uint8).copy()}
        for i, c in enumerate(bundle["members"]):
            arrays.update({f"m{i:02d}_{k}": np.frombuffer(c[k], np.uint8).copy() for k in BRANCH_PACK_ARRAYS})
    else:
        arrays = {k: np.frombuffer(bundle[k], np.uint8).copy() for k in BRANCH_PACK_ARRAYS}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / f".branch.{os.getpid()}.npz"
    _write_npz(tmp, arrays)
    sha = file_sha256(tmp)
    dst = out_dir / f"{sha}.npz"
    os.replace(tmp, dst)
    return sha, dst


def install_branch_pack(pack, cache_dir, level: int, expect_sha: str, branch_sha256: str,
                        main_sha256: str | None) -> Path:
    """Client side: check a downloaded branch pack (its sha256; branch_bundle; the bundle digest is the manifest's
    features.branch_sha256) and lay it out as <cache_dir>/branch/<sha>/ with L<nn>.npz, L<nn>.json and drive.npz;
    returns the directory. Idempotent."""
    from ballrain.branch_assets import filenames
    from ballrain.multi_branch import manifest as read_manifest
    pack = Path(pack)
    with np.load(pack, allow_pickle=False) as z:
        names = set(z.files)
    if "manifest_json" in names:  # v2 (docs/pool-prototype.md 15): the member count from the array names, then exact
        n = len(names - {"manifest_json"}) // len(BRANCH_PACK_ARRAYS)
        if not 1 <= n <= MAX_BRANCH_MEMBERS:
            raise ValueError(f"branch pack: {n} members (1..{MAX_BRANCH_MEMBERS})")
        a = _load_checked(pack, {k: (np.dtype("|u1"), 1) for k in branch_pack_v2_arrays(n)}, expect_sha)
        ba = dict(version=2, level=int(level), manifest=a["manifest_json"].tobytes(),
                  members=[dict(level=int(level), **{k: a[f"m{i:02d}_{k}"].tobytes() for k in BRANCH_PACK_ARRAYS})
                           for i in range(n)])
    else:
        a = _load_checked(pack, {k: (np.dtype("|u1"), 1) for k in BRANCH_PACK_ARRAYS}, expect_sha)
        ba = dict(level=int(level), **{k: a[k].tobytes() for k in BRANCH_PACK_ARRAYS})
    bundle, key, _ = branch_bundle(ba, level, main_sha256)
    if key != branch_sha256:
        raise ValueError("branch pack: not the policy's branch (features.branch_sha256)")
    d = Path(cache_dir) / "branch" / expect_sha
    if (d / ".complete").is_file():
        return d
    d.mkdir(parents=True, exist_ok=True)

    def put(sub: Path, name: str, data: bytes):
        sub.mkdir(parents=True, exist_ok=True)
        tmp = sub / f".{name}.{os.getpid()}.tmp"
        tmp.write_bytes(data)
        os.replace(tmp, sub / name)
    if bundle.get("version") == 2:  # the on-disk layout ballrain.branch_assets.read_bundle reads
        put(d, "branches.json", bundle["manifest"])
        for e, c in zip(read_manifest(bundle["manifest"]), bundle["members"], strict=True):
            for k, name in filenames(level).items():
                put(d / e["id"], name, c[k])
    else:
        for k, name in filenames(level).items():
            put(d, name, bundle[k])
    (d / ".complete").write_text(expect_sha, encoding="utf-8")
    return d


def read_branch_dir(branch_dir, level: int, branch_sha256: str, main_sha256: str | None) -> dict:
    """The installed branch directory as the env's EnvConfig.branch_assets bundle, checked against the manifest's pin
    (TaskMismatch otherwise)."""
    from ballrain.branch_assets import filenames
    from ballrain.multi_branch import manifest as read_manifest
    from ballrain.path import RouteError
    d = Path(branch_dir)

    def one(sub: Path) -> dict:
        return dict(level=int(level), **{k: (sub / name).read_bytes() for k, name in filenames(level).items()})
    try:
        if (d / "branches.json").is_file():  # v2: the manifest and one directory per member, in manifest order
            raw = (d / "branches.json").read_bytes()
            ba = dict(version=2, level=int(level), manifest=raw,
                      members=[one(d / e["id"]) for e in read_manifest(raw)])
        else:
            ba = one(d)
        bundle, key, _ = branch_bundle(ba, level, main_sha256)
    except (OSError, ValueError, RouteError) as e:
        raise TaskMismatch(f"the branch directory: {e}") from None
    if key != branch_sha256:
        raise TaskMismatch("the branch directory is not the policy's branch (features.branch_sha256)")
    return bundle


# ============================================================================================== env

class TaskMismatch(ValueError):
    """The task, the policy manifest, the assets and the local engine do not fit together."""


def build_env_config(task: Task, manifest: dict, local: LocalGame, starts_dir, path_dir, out_root,
                     copy: str = "pool-00", branch_dir=None):
    """The EnvConfig a client runs a task with: the manifest's env / feature / reward configuration, the task's
    start (start_record = the installed record directory, start_frame = k) and ball order, the local engine; every
    random draw of the env off. Raises TaskMismatch when the pieces do not belong together."""
    from ballrain.env import EnvConfig
    from ballrain.features import FeatureConfig
    from ballrain.rewards import RewardConfig
    if manifest.get("policy_sha") != task.policy_sha:
        raise TaskMismatch("the manifest is not the task's policy version")
    if local.engine != task.engine:
        raise TaskMismatch(f"task engine {task.engine}, local engine {local.engine}")
    env = dict(manifest["env"])
    if int(env.get("level", 0)) != task.level or env.get("start_record") != task.record:
        raise TaskMismatch("task level / record differ from the policy's")
    sd = Path(starts_dir)
    if sd.name != task.record:
        raise TaskMismatch("the starts directory is not the task's record")
    smeta = json.loads((sd / "meta.json").read_text(encoding="utf-8"))
    if smeta.get("record_sha256") != task.record_sha256:
        raise TaskMismatch("the installed record has another sha256")
    feats = dict(manifest["features"])
    if feats.get("path_sha256") != task.assets["path"]:
        raise TaskMismatch("the task's reference line is not the policy's")
    feats["path_dir"] = str(Path(path_dir).resolve())
    bundle = None  # the route branch the policy follows (features.branch_sha256): the env installs it itself
    if feats.get("branch_sha256"):
        if not task.assets.get("branch") or branch_dir is None:
            raise TaskMismatch("the policy follows a route branch: the task needs its branch pack")
        bundle = read_branch_dir(branch_dir, task.level, feats["branch_sha256"], feats.get("path_sha256"))
    elif task.assets.get("branch") or branch_dir is not None:
        raise TaskMismatch("a branch pack for a policy without a route branch")
    for k in ("far_waypoints", "ref_speed_at"):
        feats[k] = tuple(feats.get(k) or ())
    env.pop("start_record")
    return EnvConfig(**env, start_record=str(sd.resolve()), start_frame=int(task.start_frame) or None,
                     start_rest=dict(task.rest) if task.rest is not None else None,
                     curriculum=None, rest_starts=None, p_rest=0.0, rehearse_starts=None, p_rehearse=0.0,
                     warm_pool=False, warm_copy=None, trace=False, prefix=None, seed=int(task.ball_order_seed),
                     copy=copy, out_root=str(out_root), backend="headless", branch_assets=bundle,
                     headless=local.headless(task.ball_order, manifest.get("engine_options")),
                     features=FeatureConfig(**feats), reward=RewardConfig(**manifest["reward"]))


def build_env(task: Task, manifest: dict, local: LocalGame, starts_dir, path_dir, out_root, copy: str = "pool-00",
              launcher=None, branch_dir=None):
    """BallanceEnv for a task (build_env_config), checked: its action keys and observation size are the manifest's,
    and k is one of the record's valid start frames (after exclude_starts), so the env does not snap it."""
    from ballrain.env import BallanceEnv
    cfg = build_env_config(task, manifest, local, starts_dir, path_dir, out_root, copy, branch_dir)
    e = BallanceEnv(cfg, launcher=launcher)
    try:
        if [int(a) for a in e.actions] != [int(k) for k in manifest["actions"]["keys"]]:
            raise TaskMismatch("the env's action set differs from the policy's")
        if e.observation_space.shape[0] != int(manifest["arch"]["n_obs"]):
            raise TaskMismatch("the env's observation size differs from the policy's")
        k = int(task.start_frame)
        # an at-rest start with a drive target may hand over in an excluded range (TasStarts.expected knows it)
        drive = task.rest is not None and (task.rest.get("target") is not None or task.rest.get("branch_s") is not None)
        if k and (e.starts is None or (e.starts.expected(k) is None if drive else e.starts.index_of(k) is None)):
            raise TaskMismatch(f"start frame {k} is not a valid start frame of {task.record}")
        if task.rest is not None and not k:
            raise TaskMismatch("an at-rest start needs a TAS start frame")
    except BaseException:
        e.close()
        raise
    return e


def valid_start_frames(record_dir, exclude_starts=None) -> np.ndarray:
    """Server side: the start frames a task may use (TasStarts.valid after the run's exclude_starts), plus 0."""
    from ballrain.curriculum import TasStarts
    st = TasStarts(str(record_dir))
    v = st.valid
    keep = np.ones(len(v), bool)
    for lo, hi in exclude_starts or ():
        keep &= ~((v >= int(lo)) & (v < int(hi)))
    return np.r_[0, v[keep]].astype(np.int64)

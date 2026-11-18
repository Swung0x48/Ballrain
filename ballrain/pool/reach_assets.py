"""f7 reach fields as task assets, server side (docs/f7-native-port.md 8; the client side is the native session's).

The static fastreach fields of a level belong to its geometry, not to a policy. The gateway, when it publishes an f7
policy, walks the policy's start record the way a game process does -- the full level message, then each incremental
level message in order -- and takes every distinct fields key (FastReach.fields_key) the geometry passes through (L2: two,
the 小节 3-4 machinery physicalised or not). For each it builds or loads the fields in the server cache
<runtime>/fastreach (FastReach.fields_for_tracker, the learner's own code and cache name L<nn>-<sha16(key)>.npz),
writes a zstd copy into the asset store as <pool>/assets/<zsha256>.zst (served sha-addressed and immutable like every
other asset) and returns the task entries {level, key_sha256, sha256, size, zsha256, zsize}.

Only fields of the pinned build are shipped: every key must end in the running code's BUILD_VERSION, and the policy's
reach_version must equal this code's reach_features.reach_version(), else the export refuses (the gateway would hand out
fields the client and the verifier compute with another definition).

The verifier replays from the same server cache; check_cache() tells it whether every entry of a task is there with
the task's sha256 (install_from_assets() restores a missing one from the asset store), so the client and the verifier
compute from byte-identical fields.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

MAX_SIZE = 128 * 1024 * 1024  # uncompressed field file (docs/f7-native-port.md 8)
MAX_ZSIZE = 32 * 1024 * 1024
MAX_ENTRIES = 8
ZSTD_LEVEL = 9  # L2: 58.5 MB -> 4.0 MB in 0.2 s


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cache_dir() -> Path:
    """The server-side field cache: <runtime>/fastreach (the learner's and the workers' cache on this host)."""
    from ballrain import instances
    return instances.runtime_dir() / "fastreach"


def field_name(level: int, key_sha256: str) -> str:
    return f"L{int(level):02d}-{key_sha256[:16]}.npz"


def record_fields(record_dir, level: int, cache=None) -> list:
    """[(fields key, cache file)] of every geometry the start record's level stream passes through, in order of first
    appearance; each built (about 4 s) or loaded in the cache. Needs numba for a build."""
    from ballrain import protocol as BP
    from ballrain.fastreach import FastReach
    from ballrain.state import GameTracker
    rd = Path(record_dir)
    cache = Path(cache) if cache is not None else cache_dir()
    tr = GameTracker(int(level))
    tr.level_no = int(level)
    full, _ = BP.parse_level_message((rd / "level.msg").read_bytes())
    tr.merge_level(full)
    out, seen = [], set()

    def take():
        key = FastReach.fields_key(tr)
        if key in seen:
            return
        seen.add(key)
        FastReach.fields_for_tracker(tr, cache_dir=cache)  # build or load (locked, atomic, content-addressed)
        ks = hashlib.sha256(key.encode()).hexdigest()
        out.append((key, cache / field_name(level, ks)))
    take()
    inc = rd / "level.msg.inc"
    if inc.is_file():
        for m in BP.parse_level_stream(inc.read_bytes())[0]:
            tr.merge_level(m)
            take()
    return out


def export_reach_assets(record_dir, level: int, reach_version: str, assets_dir, cache=None) -> list:
    """The task entries of an f7 policy's level (see the module doc); the zstd files written to assets_dir."""
    from compression import zstd

    from ballrain import fastreach as FR
    from ballrain.reach_features import reach_version as code_version
    if reach_version != code_version():
        raise ValueError(f"the policy's reach_version {reach_version!r} is not this gateway's {code_version()!r}: "
                         "its fields would not be the ones the client and the verifier compute with")
    fields = record_fields(record_dir, level, cache)
    if not fields or len(fields) > MAX_ENTRIES:
        raise ValueError(f"{len(fields)} field geometries on level {level} (1..{MAX_ENTRIES} expected)")
    ad = Path(assets_dir)
    ad.mkdir(parents=True, exist_ok=True)
    entries = []
    for key, path in fields:
        if key.rsplit(":", 1)[-1] != str(FR.BUILD_VERSION):
            raise ValueError(f"field {path.name} is not of build {FR.BUILD_VERSION}")
        raw = path.read_bytes()
        if len(raw) > MAX_SIZE:
            raise ValueError(f"field {path.name}: {len(raw)} bytes > {MAX_SIZE}")
        z = zstd.compress(raw, level=ZSTD_LEVEL)
        if len(z) > MAX_ZSIZE:
            raise ValueError(f"field {path.name}: {len(z)} compressed bytes > {MAX_ZSIZE}")
        zsha = hashlib.sha256(z).hexdigest()
        dst = ad / f"{zsha}.zst"
        if not dst.is_file():
            tmp = ad / f".{zsha}.{os.getpid()}.tmp"
            tmp.write_bytes(z)
            os.replace(tmp, dst)
        entries.append({"level": int(level), "key_sha256": hashlib.sha256(key.encode()).hexdigest(),
                        "sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw), "zsha256": zsha,
                        "zsize": len(z)})
    return entries


_CHECKED: dict = {}  # (path, size, mtime_ns) -> sha256 (a replay checks 2 x 58 MB once per file version, not per task)


def _cached_sha(p: Path) -> str | None:
    try:
        st = p.stat()
    except OSError:
        return None
    k = (str(p), st.st_size, st.st_mtime_ns)
    if k not in _CHECKED:
        if len(_CHECKED) > 64:
            _CHECKED.clear()
        _CHECKED[k] = _sha256_file(p)
    return _CHECKED[k]


def install_from_assets(entry: dict, assets_dir, cache=None) -> Path:
    """Restore one entry's field file into the cache from the asset store (sha-checked, size-capped)."""
    from compression import zstd
    cache = Path(cache) if cache is not None else cache_dir()
    src = Path(assets_dir) / f"{entry['zsha256']}.zst"
    z = src.read_bytes()
    if len(z) != int(entry["zsize"]) or hashlib.sha256(z).hexdigest() != entry["zsha256"]:
        raise ValueError(f"asset {src.name} differs from the task's entry")
    d = zstd.ZstdDecompressor()
    raw = d.decompress(z, max_length=int(entry["size"]) + 1)
    if len(raw) != int(entry["size"]) or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise ValueError(f"asset {src.name}: the decompressed field differs from the task's entry")
    cache.mkdir(parents=True, exist_ok=True)
    dst = cache / field_name(entry["level"], entry["key_sha256"])
    tmp = cache / f".{dst.name}.{os.getpid()}.tmp"
    tmp.write_bytes(raw)
    os.replace(tmp, dst)
    return dst


def check_cache(entries, assets_dir=None, cache=None) -> str | None:
    """None when every entry's field is in the cache with the task's sha256 (a missing one is first restored from
    assets_dir when given); else a short reason (the verifier records replay_failed and retries later)."""
    cache = Path(cache) if cache is not None else cache_dir()
    for e in entries or ():
        p = cache / field_name(e["level"], e["key_sha256"])
        sha = _cached_sha(p)
        if sha is None and assets_dir is not None:
            try:
                install_from_assets(e, assets_dir, cache)
            except (OSError, ValueError) as ex:
                return f"field {p.name}: missing and not restorable ({type(ex).__name__}: {str(ex)[:120]})"
            sha = _cached_sha(p)
        if sha != e["sha256"]:
            return f"field {p.name}: {'missing' if sha is None else 'sha256 differs from the task'}"
    return None

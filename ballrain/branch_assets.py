"""Wire format for a learner-owned diversion; only the producer opens a source directory."""

import hashlib
import io
import json
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import BadZipFile, ZipFile

import numpy as np

from .branch_rules import BranchRules
from .path import RouteError, resolve_path_dir


def filenames(level):
    return {
        "line_npz": f"L{level:02d}.npz",
        "desc_json": f"L{level:02d}.json",
        "drive_npz": "drive.npz",
    }


def read_bundle(level, directory):
    root = resolve_path_dir(directory)
    if (root / "branches.json").is_file():
        from .multi_branch import manifest

        raw = (root / "branches.json").read_bytes()
        members = [read_line(level, root / e["id"]) for e in manifest(raw)]
        return dict(version=2, level=level, manifest=raw, members=members)
    return read_line(level, root)


def read_line(level, root):
    blobs = {k: (root / name).read_bytes() for k, name in filenames(level).items()}
    return dict(
        level=level,
        **blobs,
        sha256={k: hashlib.sha256(v).hexdigest() for k, v in blobs.items()},
    )


def bundle_rules(bundle):
    if bundle.get("version") == 2:
        from .branch_rules import DEFAULT_RULES

        return DEFAULT_RULES
    return BranchRules.from_desc(json.loads(bundle["desc_json"]))


def uses_custom_rules(bundle):
    from .branch_rules import DEFAULT_RULES
    members = bundle['members'] if bundle.get('version') == 2 else (bundle,)
    return any(bundle_rules(member) != DEFAULT_RULES for member in members)


def uses_entry_context(bundle):
    if bundle is None or bundle.get("version") != 2:
        return False
    from .multi_branch import manifest

    return any("entry_context" in item for item in manifest(bundle["manifest"]))


def ordinary_start_frames(starts, bundle):
    """Filter ordinary prefixes without changing the TAS record seen by explicit drills."""
    if bundle is None or bundle.get("version") != 2:
        return starts.valid
    from .multi_branch import manifest

    mask = None
    for item in manifest(bundle["manifest"]):
        spec = item.get("ordinary_exclude")
        if spec is None:
            continue
        if starts.meta.get("record_sha256") != spec["record_sha256"]:
            raise RouteError("ordinary prefix exclusion is for a different pinned TAS record")
        if mask is None:
            mask = np.ones(len(starts.valid), dtype=bool)
        for lo, hi in spec["ranges"]:
            mask &= ~((starts.valid >= lo) & (starts.valid < hi))
    if mask is None:
        return starts.valid
    if not mask.any():
        raise RouteError("ordinary prefix exclusion left no valid starts")
    return starts.valid[mask]


HANDOVER_TABLE = "moving_handover_v1"
HANDOVER_APPEND_TABLE = "moving_handover_append_v1"
HANDOVER_DENSE_TABLE = "moving_handover_append_v2"
HANDOVER_CADENCE = "moving_replay_step_v1"


def moving_handover_rows(drive):
    if HANDOVER_TABLE not in drive:
        raise ValueError("branch drive has no pinned moving handovers")
    if HANDOVER_DENSE_TABLE in drive and HANDOVER_APPEND_TABLE not in drive:
        raise ValueError("moving handover append chain has a missing table")
    tables = [drive[name] for name in (HANDOVER_TABLE, HANDOVER_APPEND_TABLE, HANDOVER_DENSE_TABLE)
              if name in drive]
    if any(a.ndim != 2 or a.shape[1] != 10 or not len(a) or a.dtype.kind != "f"
           or not np.isfinite(a).all() for a in tables):
        raise ValueError("invalid moving handover table")
    rows = np.concatenate(tables) if len(tables) > 1 else tables[0]
    if len(np.unique(rows[:, 0])) != len(rows):
        raise ValueError("duplicate moving handover offsets")
    return rows


def _only_added_moving_metadata(old, new, entry):
    """A narrow exception: original NPY members (including headers) must be byte-identical."""
    if (entry.get("legacy") or any(old[k] != new[k] for k in ("level", "desc_json", "line_npz"))):
        return False
    base = {f"{name}.npy" for name in (HANDOVER_TABLE, HANDOVER_CADENCE)}
    extra = {f"{HANDOVER_APPEND_TABLE}.npy"}
    dense = {f"{HANDOVER_DENSE_TABLE}.npy"}
    stages = (set(), base, base | extra, base | extra | dense)
    try:
        with ZipFile(io.BytesIO(old["drive_npz"])) as a, ZipFile(io.BytesIO(new["drive_npz"])) as b:
            keys = set(a.namelist())
            next_keys = set(b.namelist())
            metadata = base | extra | dense
            old_stage = keys & metadata
            new_stage = next_keys & metadata
            if (len(keys) != len(a.namelist()) or len(next_keys) != len(b.namelist())
                    or old_stage not in stages or new_stage not in stages
                    or not old_stage < new_stage or keys - metadata != next_keys - metadata
                    or any(a.read(k) != b.read(k) for k in keys)):
                return False
        with np.load(io.BytesIO(new["drive_npz"]), allow_pickle=False) as d:
            rows = moving_handover_rows(d)
            cadence = d[HANDOVER_CADENCE].item()
            if type(cadence) is not int or not 1 <= cadence <= 16:
                return False
            samples, n = d["samples"], len(d["frames"])
            for r in rows:
                if not np.equal(r[:4], np.floor(r[:4])).all():
                    return False
                offset, phase, ball, sector = map(int, r[:4])
                if not (0 < offset <= n and 0 <= phase < len(entry["phases"])):
                    return False
                valid = samples[samples[:, 0] == offset, 1]
                p = entry["phases"][phase]
                if (len(valid) != 1 or not p["local_s"][0] <= valid[0] <= p["local_s"][1]
                        or np.count_nonzero(samples[:, 1] == valid[0]) != 1 or ball != p["when"]["ball"]
                        or sector != p["when"].get("sector", json.loads(new["desc_json"])["sector"])):
                    return False
    except (BadZipFile, ValueError, TypeError, IndexError, KeyError, OSError):
        return False
    return True


def contextual_append_end(old, new, level, main_sha, old_pin, new_pin):
    """Validate a contextual append and/or moving-only additions; return its last affected main station.

    None means no guidance change. Existing member order, manifest entries, descriptions, and lines cannot change.
    """
    from .multi_branch import manifest

    err = RouteError("teacher branch exception requires a validated append-only contextual bundle or moving-only addition")
    if not old or not new or old.get("version") != 2 or new.get("version") != 2:
        raise err
    if validate(old, level, main_sha) != old_pin or validate(new, level, main_sha) != new_pin:
        raise err
    before, after = manifest(old["manifest"]), manifest(new["manifest"])
    n = len(before)
    if len(after) < n or after[:n] != before or any(e.get("entry_context") != "ordinary" for e in after[n:]):
        raise err
    moving_added = False
    for a, b, entry in zip(old["members"], new["members"][:n], before, strict=True):
        if a != b:
            if not _only_added_moving_metadata(a, b, entry):
                raise err
            moving_added = True
    if len(after) == n and not moving_added:
        raise err
    ends = [float(json.loads(m["desc_json"])["main_s"][1]) for m in new["members"][n:]]
    if not np.isfinite(ends).all():
        raise err
    return max(ends, default=None)


def rest_member(bundle, key):
    """The unqualified v1 rest syntax can address exactly one legacy member."""
    from .multi_branch import manifest

    entries = manifest(bundle["manifest"])
    indices = [
        i
        for i, e in enumerate(entries)
        if (e["id"] == key if key is not None else e.get("legacy"))
    ]
    if len(indices) != 1:
        raise RouteError(
            f"unknown or ambiguous rest branch {key!r}; specify branch:<id>:<s>"
        )
    i = indices[0]
    return entries[i]["id"], bundle["members"][i]


def validate(bundle, level, main_sha):
    if not isinstance(bundle, dict) or bundle.get("level") != level:
        raise RouteError("branch bundle level mismatch")
    if "version" in bundle and bundle["version"] != 2:
        raise RouteError("unsupported branch bundle version")
    if bundle.get("version") == 2:
        from .multi_branch import manifest

        if not isinstance(bundle.get("manifest"), bytes) or not isinstance(
            bundle.get("members"), list
        ):
            raise RouteError("invalid v2 payload")
        entries = manifest(bundle["manifest"])
        children = bundle["members"]
        if len(entries) != len(children) or any(
            "version" in child for child in children
        ):
            raise RouteError("manifest/member mismatch")
        digest = hashlib.sha256(bundle["manifest"]).digest()
        return hashlib.sha256(
            digest
            + b"".join(bytes.fromhex(validate(c, level, main_sha)) for c in children)
        ).hexdigest()
    hashes = {}
    for k in filenames(level):
        v = bundle.get(k)
        if not isinstance(v, bytes):
            raise RouteError(f"branch bundle missing {k} bytes")
        hashes[k] = hashlib.sha256(v).hexdigest()
    if hashes != bundle.get("sha256"):
        raise RouteError("branch bundle sha256 mismatch")
    desc = json.loads(bundle["desc_json"])
    BranchRules.from_desc(desc)
    if (
        desc.get("level") != level
        or not main_sha
        or desc.get("main_sha256") != main_sha
        or desc.get("branch_sha256") != hashes["line_npz"]
    ):
        raise RouteError("branch description does not match level, main or branch line")
    with np.load(io.BytesIO(bundle["drive_npz"]), allow_pickle=False) as drive:
        if str(drive["branch_sha256"]) != hashes["line_npz"]:
            raise RouteError("drive does not match branch line")
    return hashlib.sha256(
        b"".join(bytes.fromhex(hashes[k]) for k in filenames(level))
    ).hexdigest()


def install(bundle, level, main_sha, expected):
    """The owner retains the TemporaryDirectory for the lifetime of its extractors."""
    if bundle is None and expected is None:
        return None
    key = validate(bundle, level, main_sha)
    if key != expected:
        raise RouteError("branch bundle does not match feature pin")
    if bundle.get("version") == 2:
        return _materialize_multi(
            level,
            key,
            bundle["manifest"],
            tuple(tuple(c[k] for k in filenames(level)) for c in bundle["members"]),
        )
    return _materialize(level, key, tuple(bundle[k] for k in filenames(level)))


@lru_cache(maxsize=4)
def _materialize(level, key, blobs):
    # Worker recovery can construct several envs. Reuse the installation (and ReferencePath cache entry).
    temp = TemporaryDirectory(prefix=f"ballrain-branch-{key}-")
    try:
        for name, blob in zip(filenames(level).values(), blobs, strict=True):
            (Path(temp.name) / name).write_bytes(blob)
    except BaseException:
        temp.cleanup()
        raise
    return temp


@lru_cache(maxsize=4)
def _materialize_multi(level, key, raw, members):
    from .multi_branch import manifest

    temp = TemporaryDirectory(prefix=f"ballrain-branches-{key}-")
    try:
        root = Path(temp.name)
        (root / "branches.json").write_bytes(raw)
        for e, blobs in zip(manifest(raw), members, strict=True):
            child = root / e["id"]
            child.mkdir()
            for name, blob in zip(filenames(level).values(), blobs, strict=True):
                (child / name).write_bytes(blob)
    except BaseException:
        temp.cleanup()
        raise
    return temp

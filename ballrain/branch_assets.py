"""Wire format for a learner-owned diversion; only the producer opens a source directory."""

import hashlib
import io
import json
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

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

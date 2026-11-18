"""Wire format for a learner-owned diversion; only the producer opens a source directory."""
import hashlib
import io
import json
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .path import RouteError, resolve_path_dir


def filenames(level):
    return {"line_npz": f"L{level:02d}.npz", "desc_json": f"L{level:02d}.json", "drive_npz": "drive.npz"}


def read_bundle(level, directory):
    root = resolve_path_dir(directory)
    blobs = {k: (root / name).read_bytes() for k, name in filenames(level).items()}
    return dict(level=level, **blobs, sha256={k: hashlib.sha256(v).hexdigest() for k, v in blobs.items()})


def validate(bundle, level, main_sha):
    if not isinstance(bundle, dict) or bundle.get("level") != level:
        raise RouteError("branch bundle level mismatch")
    hashes = {}
    for k in filenames(level):
        v = bundle.get(k)
        if not isinstance(v, bytes):
            raise RouteError(f"branch bundle missing {k} bytes")
        hashes[k] = hashlib.sha256(v).hexdigest()
    if hashes != bundle.get("sha256"):
        raise RouteError("branch bundle sha256 mismatch")
    desc = json.loads(bundle["desc_json"])
    if (desc.get("level") != level or not main_sha or desc.get("main_sha256") != main_sha
            or desc.get("branch_sha256") != hashes["line_npz"]):
        raise RouteError("branch description does not match level, main or branch line")
    with np.load(io.BytesIO(bundle["drive_npz"]), allow_pickle=False) as drive:
        if str(drive["branch_sha256"]) != hashes["line_npz"]:
            raise RouteError("drive does not match branch line")
    return hashlib.sha256(b"".join(bytes.fromhex(hashes[k]) for k in filenames(level))).hexdigest()


def install(bundle, level, main_sha, expected):
    """The owner retains the TemporaryDirectory for the lifetime of its extractors."""
    if bundle is None and expected is None:
        return None
    key = validate(bundle, level, main_sha)
    if key != expected:
        raise RouteError("branch bundle does not match feature pin")
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

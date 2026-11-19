"""Hot-updated client logic: the signed feature module (feature_module_v1, native/wasm/feature_module.cpp, built by
tools/native/build_module.py) the gateway hands to clients that can run it.

The module is a .wasm with an ed25519 signature over its exact bytes (<module>.wasm.sig, hex; nc_release sign) made
with the gateway's module key; clients check it against the key their build carries (BALLRAIN_MODULE_PUBKEY) and
the task's sha256 before loading it. Served sha-addressed at /pool/assets/<sha256>.wasm. A task carries it in
assets.feature_module only for a session that declared feature_module_v1; the module computes the same features as
the client's built-in extractor and as the verifier's Python extractor (tools/native nc_golden --module: bit for bit
on the golden streams), so tasks with and without it verify alike.
"""
from __future__ import annotations

import hashlib
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

MODULE_ABI = 1
MAX_MODULE_BYTES = 64 << 20
_HEX128 = re.compile(r"[0-9a-f]{128}")


@dataclass(frozen=True)
class FeatureModuleRef:
    sha256: str
    size: int
    sig: str  # hex ed25519 signature of the module bytes
    abi: int = MODULE_ABI

    def asset(self) -> dict:
        """The task's assets.feature_module entry (protocol.TASK)."""
        return {"sha256": self.sha256, "size": self.size, "sig": self.sig, "abi": self.abi}


def load_module(path, assets_dir) -> FeatureModuleRef:
    """The configured module (its .sig beside it) copied into assets_dir as <sha256>.wasm. Raises ValueError."""
    p = Path(path)
    data = p.read_bytes()
    if not data.startswith(b"\0asm") or len(data) > MAX_MODULE_BYTES:
        raise ValueError(f"{p.name}: not a wasm module (or too large)")
    sig = Path(str(p) + ".sig").read_text(encoding="ascii").strip().lower()
    if not _HEX128.fullmatch(sig):
        raise ValueError(f"{p.name}.sig: not a hex ed25519 signature")
    sha = hashlib.sha256(data).hexdigest()
    dst = Path(assets_dir) / f"{sha}.wasm"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.is_file() or dst.stat().st_size != len(data):
        tmp = dst.with_suffix(".wasm.tmp")
        shutil.copyfile(p, tmp)
        tmp.replace(dst)
    return FeatureModuleRef(sha256=sha, size=len(data), sig=sig)

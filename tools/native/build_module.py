"""Build the feature module (native/wasm/feature_module.cpp + the core feature sources) as wasm32-wasip1 with wasi-sdk,
and optionally sign it (ed25519, nc_release sign: the .sig is the hex signature of the module's exact bytes).

    python tools/native/build_module.py --wasi-sdk <wasi-sdk-34 dir> --out runtime/wasm/out
        [--sign-key <module-signing.key> --nc-release build/native-Release/nc_release.exe]

Writes <out>/feature-<sha16>.wasm (+ .sig) and prints its sha256. The module is the same bytes on every platform: the
client and the verifier load it with wasmtime (Cranelift, NaN canonicalisation on), so its float results agree bit for
bit wherever it runs (the host's reach kernel aside: native fastreach, as before).
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NATIVE = REPO / "native"
CORE = ["util", "json", "sha256", "npz", "wire", "state", "geometry", "path", "branch", "features", "ingest", "stream",
        "engine_chain", "reach_block"]
MINIZ = ["miniz", "miniz_tdef", "miniz_tinfl", "miniz_zip"]
CXXFLAGS = ["--target=wasm32-wasip1", "-O2", "-std=c++17", "-fwasm-exceptions", "-mllvm", "-wasm-use-legacy-eh=false",
            "-ffp-contract=off", "-fno-fast-math"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wasi-sdk", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sign-key")
    ap.add_argument("--nc-release")
    a = ap.parse_args()
    sdk = Path(a.wasi_sdk)
    cxx, cc = sdk / "bin" / "clang++.exe", sdk / "bin" / "clang.exe"
    if not cxx.exists():
        cxx, cc = sdk / "bin" / "clang++", sdk / "bin" / "clang"
    sysroot = sdk / "share" / "wasi-sysroot"
    inc = ["-I" + str(NATIVE / "core" / "include"), "-I" + str(NATIVE / "core" / "src"),
           "-I" + str(NATIVE / "third_party" / "miniz")]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        objs = []
        srcs = [(NATIVE / "wasm" / "feature_module.cpp", True), (NATIVE / "wasm" / "fastreach_import.cpp", True)]
        srcs += [(NATIVE / "core" / "src" / f"{n}.cpp", True) for n in CORE]
        srcs += [(NATIVE / "third_party" / "miniz" / f"{n}.c", False) for n in MINIZ]
        for src, is_cxx in srcs:
            o = Path(td) / (src.stem + ".o")
            cmd = ([str(cxx)] + CXXFLAGS if is_cxx else [str(cc), "--target=wasm32-wasip1", "-O2", "-ffp-contract=off",
                                                          "-fwasm-exceptions", "-mllvm", "-wasm-use-legacy-eh=false"])
            subprocess.run(cmd + inc + ["-c", str(src), "-o", str(o)], check=True)
            objs.append(str(o))
        wasm = Path(td) / "feature.wasm"
        subprocess.run([str(cxx)] + CXXFLAGS + ["-mexec-model=reactor", "-Wl,--strip-all", "-Wl,--gc-sections",
                        "-L" + str(sysroot / "lib" / "wasm32-wasip1" / "eh"), *objs, "-lunwind", "-o", str(wasm)],
                       check=True)
        data = wasm.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        dst = out / f"feature-{sha[:16]}.wasm"
        shutil.copyfile(wasm, dst)
    # the feature sets this module computes (the gateway grants features_<v> for them, modules.py): the versions
    # core/src/features.cpp implements
    import json
    (out / f"{dst.name}.json").write_text(json.dumps({"abi": 1, "feature_versions": ["f6", "f7"]}) + "\n",
                                          encoding="utf-8")
    print(f"{dst.name}: {len(data)} bytes, sha256 {sha}")
    if a.sign_key:
        if not a.nc_release:
            raise SystemExit("--sign-key needs --nc-release")
        subprocess.run([str(Path(a.nc_release).resolve()), "sign", "--key", a.sign_key, "--manifest", str(dst)], check=True)
        print(f"signed: {dst.name}.sig")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Golden replays for CI (docs/native-client.md section 9): no game files and no engine are needed. The Python side
plays ballrain.fakegame's synthetic level through the headless socket transport (tools/native/golden.py --fake) and
records the engine bytes, features and policy outputs; the native nc_golden replays the recordings and must agree
(features bit-identical within --tol, log-probs / values within the verifier's tolerances, the drawn actions).

    python tools/native/ci_goldens.py --nc-golden build/native-Release/nc_golden.exe --out runtime/nc/ci-goldens

The real-level goldens (level geometry from the game, recordings that need the game files) stay local and run before
each release (docs/native-client.md section 9, "pre-release checks").
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc-golden", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=1500)
    a = ap.parse_args(argv)
    out = Path(a.out).resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    from ballrain.pool import dev, weights as W
    w = dev.fake_world(out / "world")
    m = W.export_state(w["state"], out / "weights", "ci")
    manifest = out / "weights" / f"{m['policy_sha']}.json"
    path_dir = str(w["path_dirs"][0])
    failed = 0
    for mode, seed in (("policy", 1), ("random", 2), ("policy", 3)):
        name = f"{mode}_s{seed}"
        rec = subprocess.run([sys.executable, str(REPO / "tools" / "native" / "golden.py"), mode, "--fake",
                              "--manifest", str(manifest), "--path-dir", path_dir, "--out", str(out / name),
                              "--seed", str(seed), "--max-steps", str(a.steps)], capture_output=True, text=True)
        if rec.returncode != 0:
            print(f"{name}: recording failed\n{rec.stderr[-2000:]}")
            failed += 1
            continue
        rp = subprocess.run([str(Path(a.nc_golden).resolve()), "--manifest", str(manifest), "--stream",
                             str(out / name / "stream.bin"), "--path-dir", path_dir], capture_output=True, text=True)
        tail = (rp.stdout.strip().splitlines() or [""])[-4:]
        print(f"{name}: {'OK' if rp.returncode == 0 else 'FAILED'} | " + " | ".join(tail))
        if rp.returncode != 0:
            print(rp.stderr[-2000:])
            failed += 1
    print("PASSED" if not failed else f"{failed} FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

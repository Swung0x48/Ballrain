"""Differential test of the BranchFollower: ballrain/branch.py against native/core/src/branch.cpp (nc_branch_diff).

The step sequences: the scenarios of tests/test_stone_branch.py (switch and respawn, the merge that must not pay twice,
the progress walk on both lines) and seeded random walks along both lines with lateral noise, ball-type and sector
flips, backtracking, jumps and resets. Every output field must be bit-identical.

    python tools/native/branch_diff.py --nc build/native-Release/nc_branch_diff.exe \
        --main benchmarks/refpaths-v2/L02.npz --branch-dir benchmarks/refpaths-alt/L02-stone
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ballrain.branch import load_branch  # noqa: E402
from ballrain.path import ReferencePath  # noqa: E402

IN = np.dtype([("p", "<f8", 3), ("sector", "<i4"), ("bt", "<i4"), ("reset", "<i4")])
OUT = np.dtype([("s", "<f8"), ("local_s", "<f8"), ("elapsed", "<f8"), ("remaining", "<f8"), ("dist", "<f8"),
                ("point", "<f8", 3), ("tangent", "<f8", 3), ("branch", "<i4"), ("active", "<i4"), ("seg", "<i8")])


def scenarios(f, rng, n_random):
    """(name, [(pos, sector, bt, reset)]) lists."""
    out = []
    p = np.array([948.74, 2.77, -56.51])
    out.append(("switch_and_respawn", [(p, 4, 1, 1), (p, 4, 2, 0), (f.alt.point_at(1700.0), 4, 2, 1),
                                       (f.alt.point_at(1700.0), 5, 2, 1), (f.main.P[0], 1, 1, 1)]))
    q = f.alt.point_at(f.hi - 0.4)
    after = f.main.point_at(f.stations[0, -2] + 1)
    seq = [(q, 4, 2, 1)] + [(q, 4, bt, 0) for _ in range(5) for bt in (1, 2)]
    seq += [(after, 4, 1, 0), (after + [0, 0, 0.2], 4, 1, 0)]
    seq += [x for _ in range(5) for x in ((q, 4, 2, 0), (after, 4, 1, 0))]
    out.append(("merge_no_double_pay", seq))
    ms, bs = f.stations
    for name, path, bt, bounds in (("walk_main", f.main, 1, ms), ("walk_alt", f.alt, 2, bs)):
        out.append((name, [(path.point_at(d), 4, bt, int(i == 0))
                           for i, d in enumerate(np.linspace(bounds[1], bounds[-2], 600))]))
    lo_m, hi_m = ms[1] - 60, ms[-2] + 60
    lo_b, hi_b = bs[1] - 60, bs[-2] + 60
    for k in range(n_random):
        seq, on_alt = [], bool(rng.integers(2))
        d = rng.uniform(lo_b, hi_b) if on_alt else rng.uniform(lo_m, hi_m)
        for i in range(int(rng.integers(200, 900))):
            reset = int(i == 0 or rng.random() < 0.004)
            r = rng.random()
            if r < 0.02:  # change line (the ball turns stone / paper) and jump to the matching station
                on_alt = not on_alt
                d = float(np.interp(d, ms, bs)) if on_alt else float(np.interp(d, bs, ms))
            elif r < 0.03:  # a big jump (respawn-like teleport) without a reset
                d = rng.uniform(lo_b, hi_b) if on_alt else rng.uniform(lo_m, hi_m)
            else:
                d += rng.normal(0.35, 0.6)  # mostly forward, sometimes back
            path = f.alt if on_alt else f.main
            d = float(np.clip(d, 0, path.length))
            pos = path.point_at(d) + rng.normal(0, [0.8, 0.3, 0.8])
            bt = (2 if on_alt else 1) if rng.random() > 0.03 else int(rng.integers(1, 4))
            sector = 4 if rng.random() > 0.02 else int(rng.choice([3, 5]))
            seq.append((pos, sector, bt, reset))
        out.append((f"random_{k}", seq))
    return out


def py_run(f, seq):
    rows = np.zeros(len(seq), OUT)
    for i, (pos, sector, bt, reset) in enumerate(seq):
        if reset:
            f.reset()
        path, pr = f.update(np.asarray(pos, dtype=np.float64), sector, bt)
        rows[i] = (pr["s"], pr["local_s"], pr["ref_elapsed_ms"], pr["ref_remaining_ms"], pr["dist"], pr["point"],
                   pr["tangent"], int(bool(pr["branch"])), int(path is f.alt), pr["seg"])
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nc", required=True)
    ap.add_argument("--main", required=True, help="the main reference asset (L<nn>.npz)")
    ap.add_argument("--branch-dir", required=True)
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--random", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    main_path = ReferencePath.load(a.main)
    f = load_branch(main_path, a.level, a.branch_dir, str(Path(a.main).parent), back=10, ahead=30, margin=20,
                    reacquire=5)
    bad = 0
    with tempfile.TemporaryDirectory() as td:
        for name, seq in scenarios(f, np.random.default_rng(a.seed), a.random):
            steps = np.zeros(len(seq), IN)
            for i, (pos, sector, bt, reset) in enumerate(seq):
                steps[i] = (pos, sector, bt, reset)
            fi, fo = Path(td) / "in.bin", Path(td) / "out.bin"
            steps.tofile(fi)
            subprocess.run([str(Path(a.nc).resolve()), "--main", a.main, "--branch-dir", a.branch_dir, "--level", str(a.level),
                            "--in", str(fi), "--out", str(fo)], check=True, capture_output=True)
            got, want = np.fromfile(fo, OUT), py_run(f, seq)
            diffs = [k for k in OUT.names if not np.array_equal(got[k], want[k])]
            n_alt = int(want["active"].sum())
            print(f"{name:22s} steps={len(seq):4d} on_branch={n_alt:4d} -> {'PASS' if not diffs else 'FAIL ' + str(diffs)}")
            if diffs:
                bad += 1
                k = diffs[0]
                g = got[k].reshape(len(seq), -1)
                w = want[k].reshape(len(seq), -1)
                i = int(np.flatnonzero((g != w).any(axis=1))[0])
                print(f"   first {k} difference at step {i}: native {g[i]} python {w[i]}")
    print("PASSED" if not bad else f"FAILED {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

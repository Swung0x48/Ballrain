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
                ("point", "<f8", 3), ("tangent", "<f8", 3), ("branch", "<i4"), ("active", "<i4"), ("seg", "<i8"),
                ("member", "<i4"), ("phase", "<i4")])


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
    multi = hasattr(f, "by_id")
    ids = [e["id"] for e, _ in f.members] if multi else []
    for i, (pos, sector, bt, reset) in enumerate(seq):
        if reset:
            f.reset()
        path, pr = f.update(np.asarray(pos, dtype=np.float64), sector, bt)
        if multi:
            member = ids.index(pr["branch_id"]) if pr["branch_id"] is not None else -1
            phase = pr["phase"] if pr["phase"] is not None else -1
            active = int(bool(pr["branch"]))
        else:
            member, phase, active = -1, -1, int(path is f.alt)
        rows[i] = (pr["s"], pr["local_s"], pr["ref_elapsed_ms"], pr["ref_remaining_ms"], pr["dist"], pr["point"],
                   pr["tangent"], int(bool(pr["branch"])), active, pr["seg"], member, phase)
    return rows


def multi_scenarios(m, rng, n_random):
    """tests/test_multi_branch.py's sequences on the wire-v2 routes, then seeded walks through every member."""
    out = []
    for key in ("back-to-wood", "east-platform"):
        e, f = m.by_id[key]
        sec, ms = f.sector, f.stations[0]
        seq = [(f.main.point_at(ms[1]), sec, 0, 1)]
        ball = 0
        for ph in e["phases"]:
            lo, hi = ph["local_s"]
            ball = ph["when"]["ball"]
            seq += [(f.alt.point_at(x), sec, ball, 0) for x in np.linspace(lo, hi, 1 + int(np.ceil((hi - lo) * 4)))]
        for _ in range(12):
            seq += [(f.main.point_at(st), sec, ball, 0) for st in (ms[-2] - 0.8, ms[-2] + 1)]
        cond = e["phases"][0]["when"]
        seq += [(np.mean(cond["box"], axis=0), sec, cond["ball"], 1)]
        out.append((f"multi_paid_{key}", seq))
    e, f = m.by_id["east-platform"]
    center = np.mean(e["phases"][0]["when"]["box"], axis=0)
    out.append(("multi_paper_plate", [(f.alt.point_at(f.hi), 3, 1, 1), (center, 2, 1, 0), (center, 3, 0, 0),
                                      (center, 3, 1, 0), (f.alt.point_at(f.hi), 3, 2, 0)]))
    e, f = m.by_id["back-to-wood"]
    out.append(("multi_early_wood", [(np.mean(e["phases"][0]["when"]["box"], axis=0), 2, 2, 1),
                                     (np.array([716.0, 32.0, -268.0]), 2, 0, 0),
                                     (np.mean(e["phases"][-1]["when"]["box"], axis=0), 2, 0, 0)]))
    _, leg = m.by_id["south-loop"]
    seq = []
    for i, st in enumerate(np.linspace(leg.lo - 3, leg.hi + 3, 600)):
        seq.append((leg.alt.point_at(st), leg.sector, 2 if leg.lo <= st <= leg.hi else 1, int(i == 0)))
    out.append(("multi_south_loop", seq))
    # seeded walks: through a member's phases (the right material most of the time), lateral noise, back steps,
    # wrong materials / sectors, jumps back to main, resets
    members = [(e, f) for e, f in m.members]
    for k in range(n_random):
        e, f = members[int(rng.integers(len(members)))]
        seq = []
        sec = f.sector
        if e.get("legacy"):
            spans = [(f.lo, f.hi, f.ball_type)]
        else:
            spans = [(p["local_s"][0], p["local_s"][1], p["when"]["ball"]) for p in e["phases"]]
        d = float(spans[0][0] - rng.uniform(0, 30))
        seq.append((f.main.point_at(max(d, 0)), sec, spans[0][2], 1))
        j = 0
        for i in range(int(rng.integers(150, 700))):
            lo, hi, ball = spans[j]
            r = rng.random()
            if r < 0.01:
                d = float(rng.uniform(0, f.alt.length))
            elif r < 0.05:
                d -= rng.uniform(0, 3)
            else:
                d += rng.uniform(0.1, 0.9)
            if d > hi and j + 1 < len(spans):
                j += 1
                lo, hi, ball = spans[j]
            bt = ball if rng.random() > 0.05 else int(rng.integers(0, 3))
            sc = sec if rng.random() > 0.02 else int(rng.choice([sec - 1, sec + 1]))
            on_main = rng.random() < 0.03
            line = f.main if on_main else f.alt
            pos = line.point_at(float(np.clip(d, 0, line.length))) + rng.normal(0, [0.5, 0.2, 0.5])
            # sometimes inside an entrance / phase box exactly
            if not e.get("legacy") and rng.random() < 0.03:
                box = e["phases"][int(rng.integers(len(e["phases"])))]["when"].get("box")
                if box is not None:
                    pos = rng.uniform(box[0], box[1])
            seq.append((pos, sc, bt, int(rng.random() < 0.004)))
        out.append((f"multi_random_{k}_{e['id']}", seq))
    return out


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
        gen = multi_scenarios if hasattr(f, "by_id") else scenarios
        for name, seq in gen(f, np.random.default_rng(a.seed), a.random):
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

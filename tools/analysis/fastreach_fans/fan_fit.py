"""The fan box against observed pushes in the r8 traced episodes (and the SR02 TAS): per PSI the vertical impulse
v[k+1] - lam v[k] is -20/66 (free flight) or -20/66 + 0.1/m (the fan pushes); contacts give other values.

    .venv/Scripts/python.exe -I tools/analysis/fastreach_fans/fan_fit.py

For every free-flight PSI step near a fan: (1) the exact OBB-OBB box test (ball local box from Balls.nmo, rotated by
the traced ball matrix, against each Kollisionsquader box) at the step's start and end states vs the observed push;
(2) the rotation-free rule "centre within r of the box" for a range of r (the nominal radius fastreach uses for the
key-free profile), and the inner / outer bounds (inscribed / circumscribed sphere of the ball box).
"""
import glob
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("FR_SRC") or Path(__file__).resolve().parents[3])
sys.path.insert(0, str(ROOT))
from ballrain import protocol as P  # noqa: E402
from ballrain.state import GameTracker  # noqa: E402
from ballrain import fastreach as FRM  # noqa: E402

from ballrain import instances  # noqa: E402
DATA = Path(os.environ.get("BALLRAIN_DATA") or instances.runtime_dir())
REC = DATA / "datasets/tas-v1/SR02_1_28_992_264"
TR = DATA / "m2-train-mac/r8/episodes-traced"
MASS = {"Ball_Wood": 1.9, "Ball_Paper": 0.2, "Ball_Stone": 10.0}
DAMP = {"Ball_Wood": 0.9, "Ball_Paper": 1.5, "Ball_Stone": 0.3}
G = 20 / 66
BALL_LO = {"Ball_Paper": np.array([-2.0387, -2.0312, -2.0359]), "Ball_Wood": -2.0 * np.ones(3),
           "Ball_Stone": np.array([-1.989, -2.0, -2.0])}
BALL_HI = {"Ball_Paper": np.array([1.9707, 1.9682, 1.9613]), "Ball_Wood": 2.0 * np.ones(3),
           "Ball_Stone": np.array([1.989, 2.0, 2.0])}

tr = GameTracker(2)
tr.merge_level(P.parse_level_message((REC / "level.msg").read_bytes())[0])
for m in P.parse_level_stream((REC / "level.msg.inc").read_bytes())[0]:
    tr.merge_level(m)
fans = FRM.fan_boxes(tr.level)
OBB = FRM.fan_obbs(fans)  # (k, 18): centre, half-extents, axes (rows), world AABB half-extents
print("fans", len(fans))
for (nm, M), o in zip(fans, OBB):
    print(f"  {nm:32s} centre {np.round(o[:3], 3)} half {np.round(o[3:6], 3)} aabb half {np.round(o[15:18], 3)}")


def obb_obb(ca, Ra, ea, cb, Rb, eb):
    axes = [Ra[i] for i in range(3)] + [Rb[i] for i in range(3)]
    for i in range(3):
        for j in range(3):
            c = np.cross(Ra[i], Rb[j])
            n = np.linalg.norm(c)
            if n > 1e-9:
                axes.append(c / n)
    d = cb - ca
    for ax in axes:
        ra = sum(ea[i] * abs(Ra[i] @ ax) for i in range(3))
        rb = sum(eb[i] * abs(Rb[i] @ ax) for i in range(3))
        if abs(d @ ax) > ra + rb:
            return False
    return True


def box_dist(p):
    """Distance from the ball centre to the nearest fan box (rotation-free)."""
    best = np.inf
    for o in OBB:
        d = p - o[:3]
        U = o[6:15].reshape(3, 3)
        ex = np.maximum(np.abs(U @ d) - o[3:6], 0.0)
        best = min(best, float(np.sqrt(ex @ ex)))
    return best


def exact_in(p, Rm, name):
    lo, hi = BALL_LO[name], BALL_HI[name]
    c = 0.5 * (lo + hi) @ Rm + p
    for o in OBB:
        if np.abs(p - o[:3]).max() > 25:
            continue
        if obb_obb(c, Rm, 0.5 * (hi - lo), o[:3], o[6:15].reshape(3, 3), o[3:6]):
            return True
    return False


import json  # noqa: E402

# The traces replay the TAS from the level start up to the episode's start_state frame and then the agent plays:
# a TAS-prefix state is counted once (the first episode that has it), agent states (after start_state) per episode.
rows = []  # name, frame, observed on (1/0), exact(k), exact(k-1), dist(k), dist(k-1), exact(k+1), dist(k+1), acc
# before / at / after, prefix flag
seen_prefix = set()
for t in sorted(glob.glob(str(TR / "*" / "*" / "trace.txt"))):
    js = list(Path(t).parent.glob("*.json"))
    k0 = int((json.loads(js[0].read_text(encoding="utf-8")).get("start_state") or {}).get("frame") or 0) if js else 0
    fr_, nm, pos, mat = [], [], [], []
    for line in open(t, encoding="utf-8", errors="replace"):
        s = line.split()
        if len(s) >= 14 and s[1].startswith("Ball_"):
            try:
                f = int(s[0])
            except ValueError:
                continue
            if 18800 <= f <= 22500 and f % 2 == 0:
                fr_.append(f); nm.append(s[1]); pos.append([float(x) for x in s[11:14]])
                mat.append([float(x) for x in s[2:11]])
    if len(fr_) < 4:
        continue
    fr_ = np.array(fr_); pos = np.array(pos); mat = np.array(mat).reshape(-1, 3, 3)
    def acc_at(j):
        """Vertical impulse at PSI j (between the steps j-1 -> j and j -> j+1); nan without both."""
        if j < 1 or j + 1 >= len(fr_) or fr_[j] - fr_[j - 1] != 2 or fr_[j + 1] - fr_[j] != 2 or nm[j] not in MASS:
            return float("nan")
        lam_ = 1 - DAMP[nm[j]] / 66
        return (pos[j + 1, 1] - pos[j, 1]) * 66 - lam_ * (pos[j, 1] - pos[j - 1, 1]) * 66

    for k in range(len(fr_) - 2):
        if fr_[k + 1] - fr_[k] != 2 or fr_[k + 2] - fr_[k + 1] != 2 or not (nm[k] == nm[k + 1] == nm[k + 2]):
            continue
        name = nm[k]
        if name not in MASS:
            continue
        f1 = int(fr_[k + 1])
        prefix = f1 <= k0
        if prefix:
            if f1 in seen_prefix:
                continue
        elif f1 <= k0 + 4:
            continue  # the hand-over itself
        p0, p1, p2 = pos[k], pos[k + 1], pos[k + 2]
        if box_dist(p0) > 6.0:
            continue
        v0, v1 = (p1 - p0) * 66, (p2 - p1) * 66
        lam = 1 - DAMP[name] / 66
        acc = v1[1] - lam * v0[1]
        on_acc = -G + 0.1 / MASS[name]
        if abs(acc - on_acc) < 0.004:
            obs = 1
        elif abs(acc + G) < 0.004:
            obs = 0
        else:
            continue  # a contact
        if abs(on_acc + G) < 0.02:
            continue  # stone: indistinguishable
        # the push on the step p1 -> p2 (v0 is the step p0 -> p1's velocity: the impulse between them acts at p1)
        if prefix:
            seen_prefix.add(f1)
        rows.append((name, f1, obs, exact_in(p1, mat[k + 1], name), exact_in(p0, mat[k], name),
                     box_dist(p1), box_dist(p0), exact_in(p2, mat[k + 2], name), box_dist(p2),
                     (acc_at(k), acc, acc_at(k + 2)), prefix, Path(t).parent.name))
print("free-flight PSI steps within 6 of a fan box:", len(rows), "(TAS prefix, once:",
      sum(1 for r in rows if r[10]), "; agent:", sum(1 for r in rows if not r[10]), ")",
      "by ball:", {b: sum(1 for r in rows if r[0] == b) for b in MASS})
obs = np.array([r[2] for r in rows])
ex1 = np.array([r[3] for r in rows]); ex0 = np.array([r[4] for r in rows]); ex2 = np.array([r[7] for r in rows])
d1 = np.array([r[5] for r in rows]); d0 = np.array([r[6] for r in rows]); d2 = np.array([r[8] for r in rows])
print("observed pushes", int(obs.sum()), "free", int((1 - obs).sum()))
for lab, e in (("exact OBB at the step's state (k)", ex1), ("exact at k-1", ex0), ("exact at k+1", ex2),
               ("exact at k or k-1", ex1 | ex0), ("exact at k or k+1", ex1 | ex2)):
    print(f"  {lab:34s}: agree {np.mean(e == obs):.4f}  push&out {int(((obs == 1) & ~e).sum())}  "
          f"free&in {int(((obs == 0) & e).sum())}")
for lab, d in (("dist at k", d1), ("dist at k-1", d0), ("min(dist k, k-1)", np.minimum(d0, d1)),
               ("min(dist k, k+1)", np.minimum(d1, d2))):
    best = max(((np.mean((d <= r) == obs), r) for r in np.arange(1.5, 4.01, 0.05)))
    print(f"  rule {lab:18s} <= r: best r {best[1]:.2f} agree {best[0]:.4f}")
for r_ in (1.95, 2.0, 2.5, 2.7, 3.0, 3.53, 3.6):
    print(f"  r {r_:.2f}: pushes with dist(k) > r {int(((obs == 1) & (d1 > r_)).sum())}, "
          f"free with dist(k) <= r {int(((obs == 0) & (d1 <= r_)).sum())}")
pushed = d1[obs == 1]
print("pushed steps: dist(k) max %.3f; free steps: dist(k) min %.3f" % (pushed.max(), d1[obs == 0].min()))
print("every disagreement of the exact test at k (impulses at k-1, k, k+1; a push is +%.3f, free flight %.3f):"
      % (-G + 0.5, -G))
for r in rows:
    if bool(r[2]) != bool(r[3]):
        print("   ", "push&out" if r[2] else "free&in ", r[11][-26:], "frame", r[1], "prefix" if r[10] else "agent ",
              "dist k-1 %.2f k %.2f k+1 %.2f" % (r[6], r[5], r[8]), "exact k-1/k+1", r[4], r[7],
              "impulses %+.3f %+.3f %+.3f" % r[9])

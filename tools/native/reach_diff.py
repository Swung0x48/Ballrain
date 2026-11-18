"""Differential test of the fastreach query kernel: ballrain/fastreach.py _perceive (numba) against
native/core/src/fastreach.cpp (nc_reach_diff).

Acceptance (run's exit status): after reach_features.scale, continuous entries within 1e-4 max(1, |v|) and no flag
(grounded, doomed, unknown_zone, air_any, air_near_can_stop, air_near_clear) different, on the realistic states and on
at most 0.1 % of the synthetic ones. Bit identity of out, dbg, pdd and the stamp_id sequence (NaN == NaN, -0.0 != 0.0)
is reported too: with the C runtime Python uses (nc_reach_diff --libm system, the default) the port is bit-identical;
with the executable's own (--crt, the kernel's default) pow differs by 1 ulp on a few arguments.

    python tools/native/reach_diff.py tables          # (re)write the embedded tables block of fastreach.cpp
    python tools/native/reach_diff.py libm            # std::log / std::pow of the C++ build vs numba's log / **
    python tools/native/reach_diff.py states --dir D  # realistic (L2 TAS replays) + synthetic states, Python outputs
    python tools/native/reach_diff.py run --dir D     # nc_reach_diff on them, bit comparison, coverage, timing
    python tools/native/reach_diff.py leaf --name sdf --dir D   # one helper on random arguments

Fields: the copies in runtime/nc/fastreach (never the training cache runtime/fastreach). States:
* realistic: every --stride-th stored frame of the L2 TAS dataset records (--dataset) (dataset.OfflineReplay: the tracker as the env sees it;
  the fields file the tracker's static geometry keys to, reach_features.fields_for; the camera quarter of
  FeatureExtractor.action_quarter); every 5th state carries the tracker's unknown zones (reach_features._zone_boxes),
  the others the field's own (empty) uzones.
* synthetic, around the realistic ones on both fields: random velocities, ball types and quarters, raised (airborne),
  at march edges, on rail axes, near the fans (in the air and on the ground nearby), cell-boundary coordinates
  (+- 1 ulp), random points of the field box; a targeted search keeps states that reach the rare fan codes and doom
  reasons.
"""

import argparse
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
FR_DIR = ROOT / "runtime" / "nc" / "fastreach"
os.environ["BALLRAIN_FASTREACH_DIR"] = str(FR_DIR)  # never the training cache runtime/fastreach

from ballrain import fastreach as F  # noqa: E402
from ballrain.reach import BALLS  # noqa: E402

CPP = ROOT / "native" / "core" / "src" / "fastreach.cpp"
NC = ROOT / "build" / "native-Release" / "nc_reach_diff.exe"
FIELDS = ("L02-66e8d101e6cd97d4.npz", "L02-86886f7d56fff7a5.npz")
DATASET = Path(os.environ.get("BALLRAIN_TAS_DATASET", str(ROOT / "runtime" / "datasets" / "tas-v1")))
L2_RECORDS = ("SR02_1_28_992_264", "SR02_1_33_070_240", "HS02_5874_2_02_751", "HS02_5885_1_57_481")
NREC = F.N_FEATURES + F.N_DBG + 8 * F.N_PDD + 2
FAN = F.DBG["fan"]
DOOM = F.DBG["doom_reason"]


# ---------------------------------------------------------------------------------------------- tables
def _hx(x: float) -> str:
    x = float(x)
    if not math.isfinite(x):
        raise ValueError(x)
    return x.hex()


def _arr(name, a, per_line=4):
    a = np.asarray(a, np.float64)
    dims = "".join(f"[{d}]" for d in a.shape)

    def rec(x, ind):
        if x.ndim == 1:
            items = [_hx(v) for v in x]
            lines = [", ".join(items[i:i + per_line]) for i in range(0, len(items), per_line)]
            if len(lines) == 1:
                return "{" + lines[0] + "}"
            return "{\n" + ",\n".join(ind + "    " + ln for ln in lines) + "}"
        return "{" + ", ".join(rec(y, ind + "    ") for y in x) + "}"

    return f"const double {name}{dims} = {rec(a, '')};\n"


def tables_block() -> str:
    """The embedded tables: per ball (BALLS 0..2) lam_a, a_air, lam_g, a_g, r, fan dv and S1 / S2 (air_tables, numpy
    pow); per quarter the key directions (FastReach._directions: x, z, push), camera right / forward; the side probe
    cos / sin (math.cos(q pi / 4) as the numba kernel computes it)."""
    import numba

    @numba.njit
    def probe(q):
        ang = q * (math.pi / 4.0)
        return math.cos(ang), math.sin(ang)

    k, s1, s2 = [], [], []
    for bt in range(3):
        b = BALLS[bt]
        S1, S2 = F.air_tables(b)
        assert len(S1) == F.N_PSI + 1
        k.append([b.air_lambda, b.air_accel, b.ground_lambda, b.ground_accel, b.radius, F.fan_dv(b)])
        s1.append(S1)
        s2.append(S2)
    dirs, rv, fv = [], [], []
    for q in range(4):
        d, r, f = F.FastReach._directions(q)
        dirs.append(d)
        rv.append(r)
        fv.append(f)
    pc = [probe(q) for q in range(8)]
    for q in range(8):  # the plain-Python path computes the same values
        ang = q * (math.pi / 4.0)
        assert pc[q] == (math.cos(ang), math.sin(ang))
    out = "// BEGIN GENERATED TABLES (python tools/native/reach_diff.py tables; do not edit by hand)\n"
    out += "// per ball: air lambda, air accel, ground lambda, ground accel, radius, fan dv (reach.BALLS, fan_dv)\n"
    out += _arr("BALL_K", k, 3)
    out += "// fastreach.air_tables(BALLS[b]) (numpy lam ** k)\n"
    out += _arr("S1T", s1)
    out += _arr("S2T", s2)
    out += "// FastReach._directions(q): key combo world x, z (unit) and push; camera right / forward (reach.key_push)\n"
    out += _arr("DIRS", dirs, 3)
    out += _arr("RIGHTV", rv, 2)
    out += _arr("FWDV", fv, 2)
    out += "// _perceive's side probes: math.cos / math.sin of q * (pi / 4)\n"
    out += _arr("PROBE_COS", [c for c, _ in pc], 4)
    out += _arr("PROBE_SIN", [s for _, s in pc], 4)
    out += "// END GENERATED TABLES\n"
    return out


def cmd_tables(a):
    s = CPP.read_text(encoding="utf-8")
    pat = re.compile(r"// BEGIN GENERATED TABLES.*?// END GENERATED TABLES\n", re.S)
    if not pat.search(s):
        sys.exit("no generated block in fastreach.cpp")
    new = pat.sub(lambda m: tables_block(), s)
    if new != s:
        CPP.write_text(new, encoding="utf-8", newline="\n")
        print("tables rewritten")
    else:
        print("tables up to date")


# ---------------------------------------------------------------------------------------------- helpers
def nc_path(a) -> Path:
    return Path(a.nc) if getattr(a, "nc", None) else NC


def field_paths():
    return [FR_DIR / n for n in FIELDS]


def run_leaf(a, name, args, field=0):
    d = Path(a.dir)
    d.mkdir(parents=True, exist_ok=True)
    fi, fo = d / f"leaf_{name}_in.npz", d / f"leaf_{name}_out.bin"
    np.savez(fi, args=np.ascontiguousarray(args, np.float64))
    subprocess.run([str(nc_path(a)), "leaf", "--fields", str(field_paths()[field]), "--name", name, "--in", str(fi),
                    "--out", str(fo)], check=True)
    raw = fo.read_bytes()
    n, m = np.frombuffer(raw[:16], np.int64)
    return np.frombuffer(raw[16:], np.float64).reshape(int(n), int(m))


def same_bits(x, y):
    """Elementwise bit equality with every NaN equal to every NaN."""
    x = np.ascontiguousarray(x, np.float64)
    y = np.ascontiguousarray(y, np.float64)
    eq = x.view(np.uint64) == y.view(np.uint64)
    return eq | (np.isnan(x) & np.isnan(y))


# ---------------------------------------------------------------------------------------------- libm
def cmd_libm(a):
    """The live libm calls of the kernel: math.log (brake stop, _v_reach) and lam ** k (float k) in numba against the
    C++ build's std::log / std::pow, on the values the kernel feeds them (lambda of the three balls, k from the brake
    stop range) and on random ones."""
    import numba

    @numba.njit
    def nlog(x):
        out = np.empty_like(x)
        for i in range(x.shape[0]):
            out[i] = math.log(x[i])
        return out

    @numba.njit
    def npow(x, k):
        out = np.empty_like(x)
        for i in range(x.shape[0]):
            out[i] = x[i] ** k[i]
        return out

    rng = np.random.default_rng(1)
    n = a.n
    lams = np.array([b.air_lambda for b in BALLS.values()] + [b.ground_lambda for b in BALLS.values()])
    # log: the ratios -A / (v0 - A) in (0, 1) (uniform, log-uniform, near 1), the lambdas, random positives
    xs = np.concatenate([rng.random(n), np.exp(-rng.random(n) * 40), 1 - rng.random(n) * 1e-6, lams,
                         np.exp(rng.normal(0, 20, n)), rng.random(n) * 100])
    xs = xs[xs > 0]
    c = run_leaf(a, "log", xs[:, None])[:, 0]
    p = nlog(xs)
    bad_log = int((~same_bits(c, p)).sum())
    # pow: lam ** k for the lambdas, k = brake stop PSIs (0..1e4 continuous), the Newton iterates, 1e6, 1e9
    lam = rng.choice(lams, 2 * n)
    k = np.concatenate([rng.random(n) * 2000, np.exp(rng.random(n) * 14)])
    lam = np.r_[lam, lams, lams, rng.random(n)]
    k = np.r_[k, np.full(6, 1e6), np.full(6, 1e9), rng.random(n) * 50]
    c = run_leaf(a, "pow", np.c_[lam, k])[:, 0]
    p = npow(lam, k)
    bad_pow = int((~same_bits(c, p)).sum())
    # pure Python agrees with numba?
    pl = np.array([math.log(x) for x in xs[:2000]])
    pp = np.array([x ** y for x, y in zip(lam[:2000], k[:2000])])
    print(f"log: {len(xs)} values, {bad_log} differ (C++ vs numba); python vs numba "
          f"{int((~same_bits(pl, nlog(xs[:2000]))).sum())} differ")
    print(f"pow: {len(lam)} values, {bad_pow} differ (C++ vs numba); python vs numba "
          f"{int((~same_bits(pp, npow(lam[:2000], k[:2000]))).sum())} differ")
    # the executable's static C runtime (std::log / std::pow: the kernel's default without set_libm)
    crt_log = int((~same_bits(run_leaf(a, "log_crt", xs[:, None])[:, 0], nlog(xs))).sum())
    crt_pow = int((~same_bits(run_leaf(a, "pow_crt", np.c_[lam, k])[:, 0], p)).sum())
    print(f"the executable's own CRT: log {crt_log} differ, pow {crt_pow} differ (the kernel default; nc_reach_diff installs the system ones)")
    return bad_log + bad_pow


# ---------------------------------------------------------------------------------------------- Python reference
class Ref:
    """One FastReach per field file (its own stamp_id sequence, as nc_reach_diff's one workspace per field)."""

    def __init__(self):
        if not F.HAVE_NUMBA:
            raise SystemExit("the reference is the numba kernel: numba is not available (or BALLRAIN_NO_NUMBA is set)")
        self.fr = [F.FastReach.load(p) for p in field_paths()]
        self.keys = [f.info.get("key") for f in self.fr]

    def perceive(self, fi, uz, p, v, bt, q):
        """FastReach.perceive's kernel call with the given unknown zones: (out, dbg, pdd, stamp_id, seconds)."""
        fr = self.fr[fi]
        S1, S2, b = fr.tables(int(bt))
        dirs, rv, fv = fr.directions(int(q))
        args = (fr.tiles, fr.Hh, fr.P1, fr.P2, fr.Pd, fr.Poff, fr.Pdim, fr.Rr, fr.Rl, fr.Rn, fr.Rf, fr.bidx,
                fr.bricks, fr.seg, fr.rstart, fr.ritems, fr.ntz, fr.ox, fr.oz, fr.sx0, fr.sy0, fr.sz0, fr.boxes,
                fr.void_y, uz, fr.fans)
        t0 = time.perf_counter()
        fr.stamp_id = F._perceive(*args, S1, S2, b.air_lambda, b.air_accel, b.ground_lambda, b.ground_accel,
                                  b.radius, F.fan_dv(b), float(p[0]), float(p[1]), float(p[2]), float(v[0]),
                                  float(v[1]), float(v[2]), dirs, rv, fv, fr.yy, fr.xf, fr.zf, fr.D, fr.fins, fr.yl,
                                  fr.ya, fr.stack, fr.stamp, fr.stamp_id, fr._nocollect, fr.work, fr.out, fr.dbg,
                                  fr.pdd)
        dt = time.perf_counter() - t0
        return fr.out.copy(), fr.dbg.copy(), fr.pdd.copy(), fr.stamp_id, dt


NOZ = np.zeros((0, 6))


SKIPPED = {}  # realistic frames whose geometry keys to another field file


def realistic_states(stride: int, limit: int, dataset_dir: Path = DATASET):
    """Yields (field index, zones, pos, vel, ball type, quarter, record tag) from the L2 TAS dataset replays."""
    from ballrain import dataset
    from ballrain import reach_features as RF
    from ballrain.features import yaw_of
    file_of = {}
    for i, name in enumerate(FIELDS):
        file_of[name.split("-")[1].split(".")[0]] = i
    count = 0
    for rec in L2_RECORDS:
        d = dataset_dir / rec
        if not (d / "raw.npz").is_file():
            print(f"skip {rec}: no dataset record")
            continue
        rep = dataset.OfflineReplay(d, 2)
        for i, o, tr in rep:
            if i % stride:
                continue
            ball = o.ball
            if ball is None or not ball["found"] or int(ball["type"]) not in (0, 1, 2) or tr.transforming:
                continue
            bt = tr.ball_type()
            if bt not in (0, 1, 2):
                continue
            p = np.asarray(ball["pos"], np.float64)
            v = np.asarray(ball["speed"], np.float64) + np.asarray(ball["speed_change"], np.float64)
            if not (np.isfinite(p).all() and np.isfinite(v).all()) or np.abs(p).sum() < 1e-6:
                continue
            fr, key = RF.fields_for(tr, cache_dir=FR_DIR)
            dig = RF.key_digest(key)
            if dig not in file_of:
                SKIPPED[dig] = SKIPPED.get(dig, 0) + 1
                continue
            cams = o.cams
            q = 0 if cams is None else int(round(yaw_of(cams["cams"][0]["matrix"]) / (math.pi / 2))) % 4
            zones = np.array(fr.uzones, np.float64).reshape(-1, 6) if count % 5 == 4 else NOZ
            yield file_of[dig], zones, p, v, int(bt), q, rec
            count += 1
            if count >= limit:
                return


class States:
    def __init__(self):
        self.pos, self.vel, self.ball, self.quarter, self.field, self.zones, self.tag = [], [], [], [], [], [], []

    def add(self, fi, zones, p, v, bt, q, tag):
        self.field.append(int(fi))
        self.zones.append(np.asarray(zones, np.float64).reshape(-1, 6))
        self.pos.append(np.asarray(p, np.float64))
        self.vel.append(np.asarray(v, np.float64))
        self.ball.append(int(bt))
        self.quarter.append(int(q))
        self.tag.append(tag)

    def __len__(self):
        return len(self.ball)

    def save(self, path):
        zoff = np.r_[0, np.cumsum([len(z) for z in self.zones])].astype(np.int64)
        zones = np.concatenate(self.zones).reshape(-1, 6) if self.zones else NOZ
        np.savez(path, pos=np.array(self.pos).reshape(-1, 3), vel=np.array(self.vel).reshape(-1, 3),
                 ball=np.array(self.ball, np.int64), quarter=np.array(self.quarter, np.int64),
                 field=np.array(self.field, np.int64), zoff=zoff, zones=zones, tag=np.array(self.tag))


def synthetic_states(ref: Ref, seeds: States, n: int, rng):
    """Perturbations of the realistic states and targeted ones (module docstring). Yields add() argument tuples."""
    P = np.array(seeds.pos)
    V = np.array(seeds.vel)
    zone_pool = [z for z in seeds.zones if len(z)]
    kinds = ("vel", "air", "edge", "rail", "fan_air", "fan_ground", "cell", "box")
    weights = np.array([0.22, 0.16, 0.16, 0.06, 0.14, 0.12, 0.08, 0.06])
    produced = 0
    while produced < n:
        kind = kinds[rng.choice(len(kinds), p=weights / weights.sum())]
        fi = int(rng.integers(len(ref.fr)))
        fr = ref.fr[fi]
        bt = int(rng.integers(3))
        q = int(rng.integers(4)) if rng.random() < 0.95 else int(rng.choice([-5, -1, 4, 7]))
        zones = zone_pool[int(rng.integers(len(zone_pool)))] if zone_pool and rng.random() < 0.15 else NOZ
        j = int(rng.integers(len(P)))
        p = P[j].copy()
        v = V[j].copy()
        if kind == "vel":
            v = rng.normal(0, rng.choice([2.0, 8.0, 20.0]), 3)
            if rng.random() < 0.5:
                v[1] = V[j][1] + rng.normal(0, 2)
            p = p + rng.normal(0, 0.3, 3)
        elif kind == "air":
            p[1] += rng.uniform(0.5, 40.0)
            p[[0, 2]] += rng.normal(0, 3, 2)
            v = rng.normal(0, 12, 3)
            v[1] = rng.uniform(-30, 25)
        elif kind == "edge":
            rest = fr.support(p)[0]
            if rest != rest:
                continue
            ang = rng.uniform(0, 2 * np.pi)
            d = np.array([math.cos(ang), math.sin(ang)])
            m = fr.march([p[0], rest, p[2]], [d[0], 0.0, d[1]], y=rest, lmax=20.0, ball_type=bt)
            s = m["edge_s"] if m["edge_s"] < np.inf else m["obst_s"]
            if not s < np.inf:
                continue
            s += rng.uniform(-1.5, 0.8)
            p = np.array([p[0] + d[0] * s, rest, p[2] + d[1] * s])
            r2 = fr.support(p)[0]
            p[1] = r2 if r2 == r2 and rng.random() < 0.7 else rest + rng.uniform(-0.2, 0.3)
            sp = rng.uniform(0, 25)
            v = np.array([d[0] * sp, rng.normal(0, 1), d[1] * sp]) + rng.normal(0, 2, 3)
        elif kind == "rail":
            if not len(fr.seg):
                continue
            sg = fr.seg[int(rng.integers(len(fr.seg)))]
            t = rng.random()
            a_, b_ = sg[:3], sg[3:6]
            p = a_ + t * (b_ - a_)
            p[1] += sg[6] + BALLS[bt].radius + rng.uniform(-0.3, 0.5)
            p[[0, 2]] += rng.normal(0, 0.3, 2)
            u = b_ - a_
            u = u / max(np.linalg.norm(u), 1e-9)
            v = u * rng.uniform(-25, 25) + rng.normal(0, 1.5, 3)
        elif kind in ("fan_air", "fan_ground"):
            if not len(fr.fans):
                continue
            fan = fr.fans[int(rng.integers(len(fr.fans)))]
            c, e = fan[:3], fan[15:18]
            bt = int(rng.choice([0, 1, 1, 2]))
            p = c + rng.uniform(-1, 1, 3) * (e + np.array([7.0, 6.0, 7.0]))
            if kind == "fan_ground":
                lay = fr.rest_layers(p[0], p[2])
                if not len(lay):
                    continue
                p[1] = lay[int(rng.integers(len(lay))), 0] + rng.uniform(-0.1, 0.1)
                v = rng.normal(0, rng.choice([3.0, 10.0, 25.0]), 3)
                v[1] = rng.normal(0, 1)
            else:
                v = rng.normal(0, 8, 3)
                v[1] = rng.uniform(-25, 15)
        elif kind == "cell":
            # coordinates on the cell / tile / voxel boundaries (+- 1 ulp) of the lookups
            for ax, o in ((0, fr.ox), (2, fr.oz)):
                k = math.floor((p[ax] - o) / F.RES + 0.5)
                x = o + (k - 0.5) * F.RES if rng.random() < 0.6 else o + k * F.RES
                step = int(rng.integers(-1, 2))
                p[ax] = float(np.nextafter(x, step * np.inf)) if step else x
            if rng.random() < 0.5:
                rest = fr.support(p)[0]
                if rest == rest:
                    p[1] = rest + rng.choice([0.0, F.GROUND_TOL, -F.GROUND_TOL])
            v = V[j] + rng.normal(0, 1, 3)
        else:  # anywhere in the field box
            lo = np.array([fr.ox, fr.void_y + 20, fr.oz])
            hi = np.array([fr.ox + fr.tiles.shape[0] * 4.0, fr.void_y + 120, fr.oz + fr.tiles.shape[1] * 4.0])
            p = rng.uniform(lo, hi)
            lay = fr.rest_layers(p[0], p[2])
            if len(lay) and rng.random() < 0.6:
                p[1] = lay[int(rng.integers(len(lay))), 0]
            v = rng.normal(0, 10, 3)
        if not (np.isfinite(p).all() and np.isfinite(v).all()):
            continue
        produced += 1
        yield fi, zones, p, v, bt, q, kind


def coverage_search(ref: Ref, rng, want: int, tries: int):
    """Random states near the fans and over the voids, kept when they reach a fan code / doom reason that has fewer
    than `want` states so far (probed on a scout reference, so that the main reference's stamp_id sequence holds only
    the kept states, as nc_reach_diff sees them). Yields add() tuples."""
    scout = Ref()
    have = {}
    for t in range(tries):
        fi = int(rng.integers(len(ref.fr)))
        fr = ref.fr[fi]
        fan = fr.fans[int(rng.integers(len(fr.fans)))]
        bt = int(rng.choice([0, 1]))
        c, e = fan[:3], fan[15:18]
        p = c + rng.uniform(-1, 1, 3) * (e + np.array([10.0, 8.0, 10.0]))
        ground = rng.random() < 0.75  # the grounded takeoff band codes (5, 6, 8, 9) are the rarest
        if ground:
            lay = fr.rest_layers(p[0], p[2])
            if not len(lay):
                continue
            p[1] = lay[int(rng.integers(len(lay))), 0]
        v = rng.normal(0, rng.choice([5.0, 15.0, 30.0]), 3)
        if ground:
            v[1] = rng.normal(0, 1)
        q = int(rng.integers(4))
        _, d, _, _, _ = scout.perceive(fi, NOZ, p, v, bt, q)
        keys = [("fan", d[FAN]), ("doom", d[DOOM])]
        if any(k[1] == k[1] and have.get(k, 0) < want for k in keys):
            for k in keys:
                if k[1] == k[1]:
                    have[k] = have.get(k, 0) + 1
            yield fi, NOZ, p, v, bt, q, "search"
    print("coverage search kept:", dict(sorted(have.items(), key=str)), flush=True)


def cmd_states(a):
    d = Path(a.dir)
    d.mkdir(parents=True, exist_ok=True)
    ref = Ref()
    rng = np.random.default_rng(a.seed)
    st = States()
    res = []
    t0 = time.time()
    for s in realistic_states(a.stride, a.realistic, Path(a.dataset)):
        st.add(*s)
        res.append(ref.perceive(s[0], s[1], s[2], s[3], s[4], s[5]))
    n_real = len(st)
    print(f"realistic: {n_real} states ({time.time() - t0:.0f} s)", flush=True)
    seeds = States()
    for i in range(n_real):
        seeds.add(st.field[i], st.zones[i], st.pos[i], st.vel[i], st.ball[i], st.quarter[i], st.tag[i])
    for s in synthetic_states(ref, seeds, a.synthetic, rng):
        st.add(*s)
        res.append(ref.perceive(s[0], s[1], s[2], s[3], s[4], s[5]))
    for s in coverage_search(ref, rng, a.want, a.search):
        st.add(*s)
        res.append(ref.perceive(s[0], s[1], s[2], s[3], s[4], s[5]))
    print(f"synthetic: {len(st) - n_real} states ({time.time() - t0:.0f} s); realistic frames skipped (other "
          f"geometry keys): {SKIPPED}", flush=True)
    # timing as nc_reach_diff measures it: the best of `repeat` passes (fresh references: the same stamp sequence)
    sec = np.array([r[4] for r in res])
    for _ in range(a.repeat - 1):
        tr = Ref()
        for i in range(len(st)):
            o, dg, pd, sid, dt = tr.perceive(st.field[i], st.zones[i], st.pos[i], st.vel[i], st.ball[i], st.quarter[i])
            if not (same_bits(o, res[i][0]).all() and same_bits(dg, res[i][1]).all() and sid == res[i][3]):
                raise RuntimeError(f"the Python kernel is not repeatable at state {i}")
            sec[i] = min(sec[i], dt)
    st.save(d / "states.npz")
    np.savez(d / "python.npz", out=np.array([r[0] for r in res]), dbg=np.array([r[1] for r in res]),
             pdd=np.array([r[2] for r in res]), stamp=np.array([r[3] for r in res], np.int64),
             sec=sec, n_real=n_real)
    print(f"wrote {d / 'states.npz'} and {d / 'python.npz'}")


# ---------------------------------------------------------------------------------------------- compare
def run_nc(a, d, libm, shared=False):
    out_bin = d / f"cpp_{libm}{'_shared' if shared else ''}.bin"
    cmd = [str(nc_path(a)), "perceive", "--fields", ",".join(str(p) for p in field_paths()), "--in",
           str(d / "states.npz"), "--out", str(out_bin), "--repeat", str(a.repeat), "--libm", libm]
    subprocess.run(cmd + (["--shared"] if shared else []), check=True)
    return np.fromfile(out_bin, np.float64).reshape(-1, NREC)


FLAGS = ("grounded", "doomed", "unknown_zone", "air_any", "air_near_can_stop", "air_near_clear")


def tolerance_report(name, c, py, real):
    """The acceptance rule of the reach block (scaled by reach_features.scale): continuous |d| <= 1e-4 max(1, |v|);
    flag differences and continuous jumps beyond it counted per state (realistic / synthetic), per-field max |d|."""
    from ballrain import reach_features as RF
    n = len(c)
    o = c[:, :F.N_FEATURES]
    so = np.array([RF.scale(x) for x in o]).reshape(n, -1)
    sp = np.array([RF.scale(x) for x in py["out"]]).reshape(n, -1)
    d = np.abs(so - sp)
    fl = [F.FEATURE_NAMES.index(k) for k in FLAGS]
    cont = [k for k in range(F.N_FEATURES) if k not in fl]
    flip = (o[:, fl] != py["out"][:, fl]).any(1)
    jump = (d[:, cont] > 1e-4 * np.maximum(1.0, np.abs(sp[:, cont]))).any(1)
    bits = ~same_bits(o, py["out"]).all(1)
    bad_real = int((flip | jump)[real].sum())
    bad_syn = int((flip | jump)[~real].sum())
    print(f"[{name}] states whose raw out differs in any bit: {int(bits.sum())}/{n}")
    for nm, m in (("realistic", real), ("synthetic", ~real)):
        k = max(int(m.sum()), 1)
        print(f"[{name}]   {nm}: flag flips {int(flip[m].sum())} ({100 * flip[m].sum() / k:.3f} %), continuous jumps "
              f"beyond 1e-4 {int(jump[m].sum())} ({100 * jump[m].sum() / k:.3f} %)")
    print(f"[{name}]   flag flips per flag: " + ", ".join(
        f"{F.FEATURE_NAMES[k]} {int((o[:, k] != py['out'][:, k]).sum())}" for k in fl))
    mx = d.max(0) if n else np.zeros(F.N_FEATURES)
    nz = [(F.FEATURE_NAMES[k], float(mx[k])) for k in range(F.N_FEATURES) if mx[k] > 0]
    print(f"[{name}]   per-field max scaled |d|: " + (", ".join(f"{a} {b:.3g}" for a, b in nz) if nz else "all 0"))
    fail = bad_real > 0 or bad_syn > 0.001 * max(int((~real).sum()), 1)
    print(f"[{name}] acceptance: {'FAIL' if fail else 'pass'}")
    return fail


def cmd_run(a):
    d = Path(a.dir)
    c = run_nc(a, d, "system")
    py = np.load(d / "python.npz")
    st = np.load(d / "states.npz")
    n = len(c)
    n_real = int(py["n_real"])
    no, nd = F.N_FEATURES, F.N_DBG
    c_out, c_dbg, c_pdd = c[:, :no], c[:, no:no + nd], c[:, no + nd:no + nd + 8 * F.N_PDD].reshape(n, 8, F.N_PDD)
    c_stamp, c_us = c[:, -2].astype(np.int64), c[:, -1]
    ok_out = same_bits(c_out, py["out"]).all(1)
    ok_dbg = same_bits(c_dbg, py["dbg"]).all(1)
    ok_pdd = same_bits(c_pdd, py["pdd"]).reshape(n, -1).all(1)
    ok_st = c_stamp == py["stamp"]
    ok = ok_out & ok_dbg & ok_pdd & ok_st
    real = np.arange(n) < n_real
    print(f"states: {n} ({n_real} realistic, {n - n_real} synthetic)")
    for name, m in (("realistic", real), ("synthetic", ~real)):
        print(f"  {name}: {int(ok[m].sum())}/{int(m.sum())} bit-identical (out {int(ok_out[m].sum())}, dbg "
              f"{int(ok_dbg[m].sum())}, pdd {int(ok_pdd[m].sum())}, stamp_id {int(ok_st[m].sum())})")
    tags = st["tag"]
    for t in sorted(set(tags.tolist())):
        m = tags == t
        print(f"    {t}: {int(ok[m].sum())}/{int(m.sum())}")
    bad = np.flatnonzero(~ok)
    for i in bad[:a.show]:
        print(f"  DIFF state {i} ({tags[i]}): field {st['field'][i]} ball {st['ball'][i]} q {st['quarter'][i]} "
              f"pos {st['pos'][i].tolist()} vel {st['vel'][i].tolist()} zones {int(st['zoff'][i + 1] - st['zoff'][i])}")
        for nm, x, y, names in (("out", c_out[i], py["out"][i], F.FEATURE_NAMES),
                                ("dbg", c_dbg[i], py["dbg"][i], F.DBG_NAMES)):
            for k in np.flatnonzero(~same_bits(x, y)):
                print(f"     {nm}.{names[k]}: c++ {x[k]!r} python {y[k]!r}")
        for q, k in zip(*np.nonzero(~same_bits(c_pdd[i], py["pdd"][i]))):
            print(f"     pdd[{F.DIR_NAMES[q]}].{F.PER_DIR_DBG[k]}: c++ {c_pdd[i, q, k]!r} python {py['pdd'][i, q, k]!r}")
        if c_stamp[i] != py["stamp"][i]:
            print(f"     stamp_id: c++ {c_stamp[i]} python {py['stamp'][i]}")
    # coverage (Python's dbg)
    fan = py["dbg"][:, FAN]
    doom = py["dbg"][:, DOOM]
    fc = {int(k): int((fan == k).sum()) for k in range(10)}
    fc["nan"] = int(np.isnan(fan).sum())
    dc = {int(k): int((doom == k).sum()) for k in range(7)}
    dc["nan"] = int(np.isnan(doom).sum())
    print("fan codes:", fc)
    print("doom reasons:", dc)
    print("grounded", int(py["out"][:, 0].sum()), "doomed", int(py["out"][:, 1].sum()), "unknown",
          int(py["out"][:, 2].sum()), "air_any", int(py["out"][:, 3].sum()))
    pu = py["sec"] * 1e6
    for name, m in (("realistic", real), ("synthetic", ~real), ("all", np.ones(n, bool))):
        print(f"time per perceive ({name}): c++ median {np.median(c_us[m]):.1f} us p99 {np.percentile(c_us[m], 99):.1f}"
              f" us; python (numba) median {np.median(pu[m]):.1f} us p99 {np.percentile(pu[m], 99):.1f} us")
    for k in (1, 2, 3, 7):
        m = fan == k
        if m.any():
            print(f"  fan {k}: c++ median {np.median(c_us[m]):.1f} us p99 {np.percentile(c_us[m], 99):.1f} us; "
                  f"python median {np.median(pu[m]):.1f} us p99 {np.percentile(pu[m], 99):.1f} us")
    fail = tolerance_report("system libm", c, py, real)
    if a.crt:
        fail |= tolerance_report("static CRT libm", run_nc(a, d, "crt"), py, real)
    if a.shared:  # the reach block's call pattern: one workspace for both fields, no dbg / pdd (out only)
        cs = run_nc(a, d, "system", shared=True)
        same = same_bits(cs[:, :F.N_FEATURES], py["out"]).all(1)
        print(f"[shared workspace, no dbg / pdd] out bit-identical on {int(same.sum())}/{n}")
        fail |= not same.all()
    return int(fail)


# ---------------------------------------------------------------------------------------------- leaves
def cmd_leaf(a):
    """One helper on random arguments around the field (C++ leaf vs the numba function)."""
    ref = Ref()
    fr = ref.fr[a.field]
    rng = np.random.default_rng(a.seed)
    n = a.n
    lo = np.array([fr.ox, fr.void_y + 20, fr.oz])
    hi = np.array([fr.ox + fr.tiles.shape[0] * 4.0, fr.void_y + 100, fr.oz + fr.tiles.shape[1] * 4.0])
    P = rng.uniform(lo, hi, (n, 3))
    if a.name == "sdf":
        args = P
        py = np.array([F._sdf(fr.bidx, fr.bricks, fr.sx0, fr.sy0, fr.sz0, *p) for p in args])
    elif a.name == "rail_dist":
        sg = fr.seg[rng.integers(len(fr.seg), size=n)]
        args = sg[:, :3] + rng.random((n, 1)) * (sg[:, 3:6] - sg[:, :3]) + rng.normal(0, 3, (n, 3))
        py = np.array([F._rail_dist(fr.seg, fr.rstart, fr.ritems, fr.ntz, fr.ox, fr.oz, *p) for p in args])
    elif a.name == "h_top":
        args = np.c_[P[:, 0], P[:, 2], P[:, 1] - 50, P[:, 1] + 50]
        py = np.array([[F._h_top(fr.tiles, fr.Hh, fr.ox, fr.oz, *x)] for x in args])
    elif a.name == "r_ground":
        args = np.c_[P[:, 0], P[:, 2], P[:, 1], np.full(n, 100.0)]
        py = np.array([F._r_ground(fr.tiles, fr.Rr, fr.Rl, fr.Rn, fr.Rf, fr.ox, fr.oz, *x) for x in args])
    elif a.name == "march":
        ang = rng.uniform(0, 2 * np.pi, n)
        args = np.c_[np.full(n, 2.0), P, np.cos(ang), np.sin(ang), np.full(n, 20.0), np.ones(n), np.full(n, np.nan)]
        py = []
        for x in args:
            o = np.zeros(9)
            F._march(fr.tiles, fr.Hh, fr.Rr, fr.Rl, fr.Rn, fr.Rf, fr.bidx, fr.bricks, fr.seg, fr.rstart, fr.ritems,
                     fr.ntz, fr.ox, fr.oz, fr.sx0, fr.sy0, fr.sz0, x[0], x[1], x[2], x[3], x[4], x[5], x[6], True,
                     x[8], o)
            py.append(o)
        py = np.array(py)
    elif a.name == "brake_axis":
        b = BALLS[0]
        args = np.c_[rng.normal(0, 20, n), rng.normal(0, 0.1, n), np.full(n, b.ground_accel),
                     np.full(n, b.ground_lambda), np.where(rng.random(n) < 0.5, 1e9, rng.random(n) * 300)]
        py = np.array([F._brake_axis(*x) for x in args])
    elif a.name == "v_reach":
        b = BALLS[int(rng.integers(3))]
        args = np.c_[rng.normal(5, 15, n), rng.normal(0, 0.3, n), np.full(n, b.ground_lambda), rng.random(n) * 25]
        py = np.array([[F._v_reach(*x)] for x in args])
    else:
        sys.exit(f"no Python side for leaf {a.name}")
    c = run_leaf(a, a.name, args, a.field)
    bad = ~same_bits(c, py).all(1)
    print(f"{a.name}: {n} calls, {int(bad.sum())} differ")
    for i in np.flatnonzero(bad)[:5]:
        print("  args", args[i].tolist(), "c++", c[i].tolist(), "python", np.asarray(py[i]).tolist())
    return int(bad.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("tables", "libm", "states", "run", "leaf"))
    ap.add_argument("--dir", default=str(ROOT / "runtime" / "nc" / "reach_diff"))
    ap.add_argument("--nc", default=None, help=f"nc_reach_diff executable (default {NC})")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--stride", type=int, default=4, help="realistic: every n-th stored frame")
    ap.add_argument("--realistic", type=int, default=30000)
    ap.add_argument("--synthetic", type=int, default=25000)
    ap.add_argument("--search", type=int, default=60000, help="coverage search tries")
    ap.add_argument("--want", type=int, default=40, help="coverage search: states kept per fan code / doom reason")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--show", type=int, default=10)
    ap.add_argument("--crt", action="store_true", help="run: also measure the static-CRT log / pow variant")
    ap.add_argument("--shared", action="store_true", help="run: also the reach block's call pattern (out only)")
    ap.add_argument("--dataset", default=str(DATASET), help="tas-v1 dataset dir (BALLRAIN_TAS_DATASET)")
    ap.add_argument("--name", default="sdf")
    ap.add_argument("--field", type=int, default=0)
    a = ap.parse_args()
    r = {"tables": cmd_tables, "libm": cmd_libm, "states": cmd_states, "run": cmd_run, "leaf": cmd_leaf}[a.cmd](a)
    sys.exit(1 if r else 0)


if __name__ == "__main__":
    main()

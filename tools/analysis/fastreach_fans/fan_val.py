"""fastreach near the L2 fans (P_Modul_18): before/after validation.

    FR_SRC=<repo root to import ballrain from> FR_OUT=<dir> .venv/Scripts/python.exe -I fan_val.py <tag>

Reads the tas-v1 SR02 dataset and the mirrored r8 traced episodes from BALLRAIN_DATA (default: the runtime dir,
BALLRAIN_RUNTIME or <repo>/runtime). Fields are cached in <runtime>/fastreach/fans-wt/L02-<tag>.npz (a new file per tag; the shared L02.npz is
never written). Writes fan_val_<tag>.json into FR_OUT.
Parts:
  tas      every TAS state (tracked ball type): all doomed flags (frame, reason), unknown_zone count
  pit      the logic of runtime/viz/l2pit/pit_reach.py + pit_gap_stats.py (deaths 17550-17950, lives through the pit)
  traced   every r8 traced death in [18500, 22500] (fans 003-008 and 001 lie on the TAS route at 19050-22300):
           doomed lead, unknown; doomed states not followed by a BallOff within 420 frames (false alarms)
  vert     (only with the fan API) TAS 20800-21600: predicted vertical motion vs the TAS over 0.5 s
  latency  fixed TAS state set, warmed, median of 3 repeats
"""
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

SRC = Path(os.environ.get("FR_SRC") or Path(__file__).resolve().parents[3])
sys.path.insert(0, str(SRC))
from ballrain import protocol as P  # noqa: E402
from ballrain.state import GameTracker  # noqa: E402
from ballrain.dataset import load_observations  # noqa: E402
from ballrain import fastreach as FRM  # noqa: E402
from ballrain.fastreach import FastReach, FEATURE_NAMES, DIR_NAMES  # noqa: E402

TAG = sys.argv[1]
from ballrain import instances  # noqa: E402
DATA = Path(os.environ.get("BALLRAIN_DATA") or instances.runtime_dir())
REC = DATA / "datasets/tas-v1/SR02_1_28_992_264"
TR = DATA / "m2-train-mac/r8/episodes-traced"
CACHE = DATA / "fastreach/fans-wt"
OUT = Path(os.environ.get("FR_OUT") or ".")
F = {n: i for i, n in enumerate(FEATURE_NAMES)}
BT = {"Ball_Wood": 0, "Ball_Paper": 1, "Ball_Stone": 2}
print("fastreach from", FRM.__file__, "BUILD_VERSION", FRM.BUILD_VERSION, "numba", FRM.HAVE_NUMBA)
R = {"tag": TAG, "build_version": FRM.BUILD_VERSION}

tr = GameTracker(2)
tr.merge_level(P.parse_level_message((REC / "level.msg").read_bytes())[0])
for m in P.parse_level_stream((REC / "level.msg.inc").read_bytes())[0]:
    tr.merge_level(m)
tr.level_no = 2
t0 = time.time()
fr = FastReach.for_tracker(tr, cache_dir=CACHE, name=f"L02-{TAG}")
R["fields_s"] = time.time() - t0
R["fields_mb"] = sum(fr.sizes().values()) / 1e6
R["build_s"] = fr.info.get("build_s", {}).get("total")
R["fans"] = fr.info.get("fans")
print(f"fields {R['fields_s']:.1f} s {R['fields_mb']:.0f} MB, build {R['build_s']}")


def q(p, v, bt):
    return fr.perceive(p, v, bt, 0, debug=True)


def reason(info):
    r = info["dbg"]["doom_reason"]
    return int(r) if r == r else -1


# ------------------------------------------------------------------------------------------------ TAS
tas = []  # frame, pos, vel, bt
for i, o in enumerate(load_observations(REC / "raw.npz")):
    tr.update(o)
    b = o.ball
    if b is None:
        continue
    p = np.asarray(b["pos"], float)
    if np.abs(p).sum() < 1e-6:
        continue
    v = np.asarray(b["speed"], float) + np.asarray(b["speed_change"], float)
    bt = tr.ball_type()
    tas.append((i, p, v, 0 if bt is None else int(bt)))
doom, unk, unk_fan = [], 0, []
fan_codes = {}
for i, p, v, bt in tas:
    vec, info = q(p, v, bt)
    if vec[F["doomed"]] > 0.5:
        doom.append((i, reason(info), bt, [round(float(x), 2) for x in p]))
    fc = info["dbg"].get("fan", float("nan"))
    if fc == fc:
        fan_codes[int(fc)] = fan_codes.get(int(fc), 0) + 1
    if vec[F["unknown_zone"]] > 0.5:
        unk += 1
        if 18900 <= i <= 22400:  # the fan stretch, as the latency window (19000 missed the fan states 18992-18997)
            unk_fan.append((i, reason(info), int(fc) if fc == fc else -1, int(vec[F["grounded"]]),
                            [round(float(x), 2) for x in p], [round(float(x), 2) for x in v]))
R["tas"] = {"states": len(tas), "doomed": doom, "n_doomed": len(doom), "unknown": unk,
            "unknown_18900_22400": unk_fan, "fan_codes": fan_codes}
print(f"TAS: {len(tas)} states, doomed {len(doom)} (frames {sorted(set(d[0] // 100 * 100 for d in doom))}); "
      f"unknown_zone {unk}, in 18900-22400: {len(unk_fan)} {unk_fan[:40]}; fan dbg codes {fan_codes}")
TASD = {i: (p, v, bt) for i, p, v, bt in tas}


# ------------------------------------------------------------------------------------------------ traces
def read_trace(path):
    fr_, nm, pos = [], [], []
    for line in open(path, encoding="utf-8", errors="replace"):
        s = line.split()
        if len(s) >= 14 and s[1].startswith("Ball_"):
            try:
                fr_.append(int(s[0])); nm.append(s[1]); pos.append((float(s[11]), float(s[12]), float(s[13])))
            except ValueError:
                pass
    return np.array(fr_), nm, np.array(pos)


def deaths(ev):
    return [int(l.split()[0]) for l in open(ev, encoding="utf-8", errors="replace") if " BallOff" in l]


def series(f, nm, pos, a, b):
    """PSI states in frames [a, b): pos, vel (PSI difference x 66), ball type (pit_reach.py)."""
    m = (f >= a - 2) & (f < b)
    ff, pp, nn = f[m], pos[m], [nm[k] for k in np.nonzero(m)[0]]
    rows = []
    for k in range(1, len(ff)):
        if ff[k] % 2:
            continue
        j = k - 1
        while j >= 0 and ff[k] - ff[j] < 2:
            j -= 1
        if j < 0 or ff[k] - ff[j] != 2:
            continue
        v = (pp[k] - pp[j]) * 66.0
        rows.append((ff[k], pp[k], v, BT.get(nn[k], 0)))
    return rows


EP = []
# the mirror grows while r8 trains: FR_EPISODES names a list file that freezes the set (written on first use)
eplist = Path(os.environ["FR_EPISODES"]) if os.environ.get("FR_EPISODES") else None
if eplist is not None and eplist.is_file():
    dirs_ = [TR / s for s in eplist.read_text(encoding="utf-8").split()]
else:
    dirs_ = [Path(d) for d in sorted(glob.glob(str(TR / "*" / "*")))]
    if eplist is not None:
        eplist.write_text("\n".join(str(d.relative_to(TR)).replace("\\", "/") for d in dirs_), encoding="utf-8")
for d in dirs_:
    d = Path(d)
    tt, ev = d / "trace.txt", d / "events.txt"
    js = list(d.glob("*.json"))
    if not (tt.exists() and ev.exists() and js):
        continue
    meta = json.loads(js[0].read_text(encoding="utf-8"))
    k0 = int((meta.get("start_state") or {}).get("frame") or 0)
    f, nm, pos = read_trace(tt)
    if len(f):
        EP.append((d.name, k0, meta.get("end_reason"), f, nm, pos, deaths(ev)))
print("traced episodes", len(EP))


def death_flags(lo, hi):
    """pit_reach.py's analysis of the deaths in [lo, hi]."""
    res = []
    for name, k0, end, f, nm, pos, dd in EP:
        for x in dd:
            if lo <= x <= hi and x - 30 > k0:
                rows = series(f, nm, pos, max(k0 + 4, x - 396), x)
                flags = []
                for ff, p, v, bt in rows:
                    vec, info = q(p, v, bt)
                    flags.append((ff, vec[F["doomed"]] > .5, vec[F["grounded"]] > .5, vec[F["unknown_zone"]] > .5,
                                  reason(info), bt, p, info["dbg"].get("fan", float("nan"))))
                if not flags:
                    continue
                fa = np.array([r[0] for r in flags]); dm = np.array([r[1] for r in flags])
                gr = np.array([r[2] for r in flags]); un = np.array([r[3] for r in flags])
                first_last = None
                if dm[-1]:
                    k = len(dm) - 1
                    while k > 0 and dm[k - 1]:
                        k -= 1
                    first_last = int(fa[k])
                fanf = [r[7] for r in flags]
                res.append(dict(name=name, k0=k0, death=x, first_last=first_last,
                                first_any=int(fa[dm][0]) if dm.any() else None,
                                last_ground=int(fa[gr][-1]) if gr.any() else None, n=len(flags),
                                unknown_last=bool(un[-1]), unknown_any=int(un.sum()), bt=int(flags[-1][5]),
                                reasons=sorted(set(int(r[4]) for r in flags if r[1])),
                                pos_last=[round(float(c), 1) for c in flags[-1][6]],
                                fan_states=int(sum(1 for z in fanf if z == z and z > 0))))
    return res


def passed_lives(lo, hi):
    out = []
    for name, k0, end, f, nm, pos, dd in EP:
        if k0 < lo and not any(lo <= x <= hi for x in dd) and f.max() > hi:
            prior = [x for x in dd if x < lo]
            a = max(k0 + 4, (max(prior) + 300) if prior else 0, lo - 200)
            rows = series(f, nm, pos, a, hi)
            hit = []
            for ff, p, v, bt in rows:
                vec, info = q(p, v, bt)
                if vec[F["doomed"]] > .5:
                    hit.append((int(ff), reason(info)))
            out.append((name, end, len(rows), hit))
    return out


def summarize(res):
    lead = [(r["death"] - r["first_last"]) / 132 for r in res if r["first_last"] is not None]
    return {"deaths": len(res), "flagged": len(lead),
            "lead_median_s": float(np.median(lead)) if lead else None,
            "unknown_at_death": sum(r["unknown_last"] for r in res)}


# ---- pit (pit_reach.py)
pit = death_flags(17550, 17950)
pp = passed_lives(17550, 17950)
tas_pit = [d for d in doom if 17000 <= d[0] <= 18300]
R["pit"] = dict(summarize(pit), passed=len(pp), passed_with_doomed=sum(1 for x in pp if x[3]),
                passed_states=sum(x[2] for x in pp), tas_17000_18300_doomed=len(tas_pit),
                first_last=[r["first_last"] for r in pit])
print("pit", {k: v for k, v in R["pit"].items() if k != "first_last"})
# ---- pit_gap_stats.py
dirs, _, _ = FastReach.directions(0)
D2 = np.array([[d[0], d[1]] for d in dirs]); D2 /= np.linalg.norm(D2, axis=1, keepdims=True)


def along(p, v, bt):
    vec, _ = q(p, v, bt)
    vh = np.array([v[0], v[2]]); vh /= np.linalg.norm(vh) + 1e-9
    k = int(np.argmax(D2 @ vh))
    return vec[F["grounded"]] > .5, vec[F[DIR_NAMES[k] + "_gap_deficit"]]


EPN = {e[0]: e for e in EP}
gs = {}
for t in (0, 16):
    vals = []
    for r in pit:
        lg = r["last_ground"]
        if lg is None:
            continue
        e = EPN[r["name"]]
        rows = {ff: (p, v, bt) for ff, p, v, bt in series(e[3], e[4], e[5], lg - 140, lg + 2)}
        for f0 in range(lg - t, lg - t - 10, -2):
            if f0 in rows:
                g, gd = along(*rows[f0])
                if g:
                    vals.append(gd); break
    gs[f"deaths_minus_{t}"] = (len(vals), float(np.mean(np.array(vals) > 0)) if vals else None)
tv = []
for i in range(17440, 17545):
    if i in TASD:
        g, gd = along(*TASD[i])
        if g:
            tv.append(gd)
gs["tas_17440_17544"] = (len(tv), float(np.mean(np.array(tv) > 0)) if tv else None)
R["pit"]["gap_stats"] = gs
print("pit gap stats", gs)

# ---- fan stretch deaths
fan = death_flags(18500, 22500)
R["traced_deaths_18500_22500"] = dict(summarize(fan), per_death=[
    {k: r[k] for k in ("name", "death", "first_last", "first_any", "last_ground", "unknown_last", "unknown_any", "bt",
                       "reasons", "pos_last", "fan_states")} for r in fan])
s_ = summarize([r for r in fan if 20500 <= r["death"] <= 22000])
R["traced_deaths_20500_22000"] = s_
print("deaths 18500-22500", summarize(fan), "| 20500-22000", s_)
for r in fan:
    if 20500 <= r["death"] <= 22000:
        print("  ", r["name"][-24:], "death", r["death"], "doomed_from", r["first_last"], "first_any", r["first_any"],
              "last_ground", r["last_ground"], "unknown_last", r["unknown_last"], "unknown_any", r["unknown_any"],
              "bt", r["bt"], "pos", r["pos_last"], "fan_states", r["fan_states"])


# ---- false alarms: doomed states in [lo, hi] not followed by a BallOff within 420 frames
def false_alarms(lo, hi):
    """Agent states only (after each episode's start_state; the replayed TAS prefix is the TAS part above)."""
    fa, n, eps, ex = 0, 0, set(), []
    n_doom, n_unk, codes, reasons = 0, 0, {}, {}
    for name, k0, end, f, nm, pos, dd in EP:
        last = f.max()
        rows = series(f, nm, pos, max(lo, k0 + 4), min(hi, last - 420))
        for ff, p, v, bt in rows:
            if np.linalg.norm(v) > 150 or any(0 <= ff - x <= 300 for x in dd):
                continue  # respawn jumps, the respawn itself
            n += 1
            vec, info = q(p, v, bt)
            fc = info["dbg"].get("fan", float("nan"))
            if fc == fc:
                codes[int(fc)] = codes.get(int(fc), 0) + 1
            n_doom += int(vec[F["doomed"]] > .5)
            n_unk += int(vec[F["unknown_zone"]] > .5)
            rs = reason(info)
            reasons[rs] = reasons.get(rs, 0) + 1
            if vec[F["doomed"]] > .5 and not any(0 <= x - ff <= 420 for x in dd):
                fa += 1
                eps.add(name)
                if len(ex) < 12:
                    ex.append((name[-24:], int(ff), reason(info), bt, [round(float(c), 1) for c in p]))
    return {"states": n, "false_alarm_states": fa, "episodes": len(eps), "examples": ex, "doomed_states": n_doom,
            "unknown_states": n_unk, "fan_codes": codes, "doom_reasons": reasons}


R["false_alarms_18500_22500"] = false_alarms(18500, 22500)
print("false alarms 18500-22500", R["false_alarms_18500_22500"])

# ------------------------------------------------------------------------------------------------ vertical motion
if hasattr(fr, "fan_profile"):
    ks = sorted(i for i in TASD if 20700 <= i <= 21700)
    psi = [ks[0]] + [k for a_, k in zip(ks, ks[1:]) if np.any(TASD[k][1] != TASD[a_][1])]
    errs = {"key_free": [], "no_fan": [], "tas_path": []}
    H = 33
    for j in range(len(psi) - H):
        i0 = psi[j]
        if not (20800 <= i0 <= 21600) or psi[j + H] - psi[j] != 2 * H:
            continue
        p, v, bt = TASD[i0]
        prof = fr.fan_profile(p, v, bt)
        if not prof["inside"][0]:
            continue
        ytas = np.array([TASD[psi[j + n]][0][1] for n in range(H + 1)])
        vy = np.array([TASD[psi[j + n]][1][1] for n in range(H + 1)])
        if np.any(np.abs(np.diff(vy)) > 1.0):
            continue  # a ground contact within the 0.5 s
        path = np.array([TASD[psi[j + n]][0] for n in range(H + 1)])
        errs["key_free"].append(np.abs(prof["y"][:H + 1] - ytas).max())
        errs["no_fan"].append(np.abs(prof["y_nofan"][:H + 1] - ytas).max())
        errs["tas_path"].append(np.abs(fr.fan_profile(p, v, bt, path=path)["y"][:H + 1] - ytas).max())

    def st(e):
        e = np.array(e)
        return {"n": len(e), "median": float(np.median(e)), "p90": float(np.percentile(e, 90)), "max": float(e.max())}
    R["vert"] = {k: st(e) for k, e in errs.items()}
    print("vertical 0.5 s max error", R["vert"])

# ------------------------------------------------------------------------------------------------ latency
def latency(sel):
    LS = [TASD[i] for i in sorted(TASD) if sel(i) and i % 2 == 0]
    for p, v, bt in LS[:50]:
        fr.perceive(p, v, bt, 0)
    med, allt = [], []
    for rep in range(5):
        ts = []
        for p, v, bt in LS:
            s = time.perf_counter()
            fr.perceive(p, v, bt, 0)
            ts.append(time.perf_counter() - s)
        med.append(np.median(ts))
        allt += ts
    return {"states": len(LS), "median_us": float(np.median(med) * 1e6), "mean_us": float(np.mean(allt) * 1e6),
            "p99_us": float(np.percentile(allt, 99) * 1e6)}


R["latency"] = {"pit_17000_18300": latency(lambda i: 17000 <= i <= 18300),
                "fans_18900_22400": latency(lambda i: 18900 <= i <= 22400),
                "whole_tas": latency(lambda i: True)}
print("latency", R["latency"])
OUT.mkdir(parents=True, exist_ok=True)
(OUT / f"fan_val_{TAG}.json").write_text(json.dumps(R, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist")
                                                    else str(o)), encoding="utf-8")

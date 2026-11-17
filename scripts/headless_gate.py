"""Observation gate of the headless backend (docs/headless.md): a TAS played through the headless agent mode
against the retail tas-v1 dataset of the same record, per frame:

  - every parsed OBS1 section, field by field, with the CK ids of the headless engine mapped to the retail ones by
    entity / group / array name (the two engines allocate ids differently); the fields the headless engine cannot give
    (FRIC lt_*, MIND ledge, GLCT group counters, BALL / BODY raw core bit fields beyond the unmoveable bits and the
    movement state, SCOR / SREN bml_hs) are reported apart, as "approximated";
  - the level messages (full + increments);
  - the f5 feature vectors, each side computed by its own GameTracker exactly as a live session feeds it;
  - events, SR (the mod's double sum and the Python recomputation from frame log + events), HS readout.

    uv run python scripts/headless_gate.py --headless DIR --record SR06_1_36_163_132 [--out report.json]

DIR is a headless/tools/replay_stub.py output (obs.bin, obs_index.txt, level.msg[.inc], framelog.bin, events.txt).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ballrain import instances, protocol as P, tas  # noqa: E402
from ballrain.features import FeatureConfig, FeatureExtractor  # noqa: E402
from ballrain.protocol import Ev  # noqa: E402
from ballrain.recording import sr_timer  # noqa: E402
from ballrain.state import GameTracker  # noqa: E402

APPROX = {  # (section, field) the headless engine cannot give; compared apart
    ("FRIC", "lt_normal"), ("FRIC", "lt_point"), ("FRIC", "lt_valid"), ("MIND", "ledge"),
    ("GLCT", "groups"), ("SCOR", "bml_hs"), ("SREN", "bml_hs"), ("BALL", "core_flags_raw"),
    ("BALL", "core_state_raw"), ("BODY", "core_state_raw"), ("TIME", "flags_headless"),
    ("SCOR", "energy_param_ptr"), ("SREN", "energy_param_ptr"), ("MIND", "plane_w"),
}
# --phys2002 (the Windows host running the 2002 physics_RT.dll, headless/README.md): the IVP fields are read from the
# DLL's memory as the retail mod reads them, so only the values that are addresses of one process (MIND ledge, the
# Energy parameter cell's pointer) or unmaintained residue (MIND plane w) stay apart; everything else counts as exact
APPROX_PHYS2002 = {("MIND", "ledge"), ("SCOR", "energy_param_ptr"), ("SREN", "energy_param_ptr"), ("MIND", "plane_w"),
                   ("TIME", "flags_headless")}
_APPROX = {"set": APPROX}


def _approx(sec, name) -> bool:
    return (sec, name) in _APPROX["set"]


GAMEPLAY_ARRAYS = ("Energy", "CurrentLevel", "IngameParameter", "PH", "Checkpoints", "ResetPoints", "AllLevel", "Keyboard")


# --------------------------------------------------------------------------------------------- loading

def load_headless(d: Path):
    data = (d / "obs.bin").read_bytes()
    out = []
    for line in (d / "obs_index.txt").read_text().split("\n"):
        if line.strip():
            fd, off, n = map(int, line.split())
            out.append(data[off:off + n])
    return out


def load_retail(rec_dir: Path):
    z = np.load(rec_dir / "raw.npz")
    off, blob = z["obs_offsets"], z["obs_bytes"]
    return [blob[off[i]:off[i + 1]].tobytes() for i in range(len(off) - 1)]


def level_messages(d: Path):
    full = P.parse_level_message((d / "level.msg").read_bytes())[0] if (d / "level.msg").exists() else None
    inc = P.parse_level_stream((d / "level.msg.inc").read_bytes())[0] if (d / "level.msg.inc").exists() else []
    return full, inc


# --------------------------------------------------------------------------------------------- id map

class IdMap:
    """headless CK id -> retail CK id, by name (entities, groups, arrays; repeated names by occurrence order),
    meshes by (entity, kind, index) of the Physicalize records, contact blocks by LTAB order."""

    def __init__(self, hl, rt):
        self.m: dict = {}
        self.unmapped = Counter()
        hf, hi = hl
        rf, ri = rt

        def by_name(items_h, items_r, key):
            a, b = defaultdict(list), defaultdict(list)
            for x in items_h:
                a[key(x)].append(x.ckid)
            for x in items_r:
                b[key(x)].append(x.ckid)
            for k, ids in a.items():
                for i, ck in enumerate(ids):
                    if i < len(b.get(k, ())):
                        self.m.setdefault(ck, b[k][i])

        by_name([e for e in hf.entities if e.ckid], [e for e in rf.entities if e.ckid], lambda e: e.name)
        by_name(hf.groups, rf.groups, lambda g: g.name)
        by_name(hf.arrays, rf.arrays, lambda a: a.name)
        # meshes: through the Physicalize records, paired by (frame, entity) (the order of the Physicalize calls
        # inside one frame differs between the engines, so record sequence numbers do not pair them)
        pr = defaultdict(list)
        for lm in [rf] + ri:
            for p in lm.phys:
                if not p.unphysicalize:
                    pr[(p.frame, p.ckid)].append(p)
        for lm in [hf] + hi:
            for p in lm.phys:
                q = pr.get((p.frame, self.m.get(p.ckid, -1)))
                if p.unphysicalize or not q:
                    continue
                q = q[0]
                for a, b in ((p.convex, q.convex), (p.concave, q.concave)):
                    for x, y in zip(a, b):
                        self.m.setdefault(x, y)
        th, tr = (hf.tables or [{}])[0], (rf.tables or [{}])[0]
        for x, y in zip(th.get("contact_bbs", []), tr.get("contact_bbs", [])):
            self.m.setdefault(x, y)
        for x, y in zip(th.get("cams", []), tr.get("cams", [])):
            self.m.setdefault(x, y)
        self.m[0] = 0

    def __call__(self, ck) -> int:
        ck = int(ck)
        r = self.m.get(ck)
        if r is None:
            self.unmapped[ck] += 1
            return -ck - 1
        return r


# --------------------------------------------------------------------------------------------- compare

class Report:
    def __init__(self):
        self.fields = {}  # "SEC.field" -> {"mismatch": n, "first": frame, "compared": n, "approx": bool}

    def add(self, key, frame, equal, approx=False):
        f = self.fields.setdefault(key, {"compared": 0, "mismatch": 0, "first": None, "approx": approx})
        f["compared"] += 1
        if not equal:
            f["mismatch"] += 1
            if f["first"] is None:
                f["first"] = frame

    def exact_fields(self):
        return {k: v for k, v in self.fields.items() if not v["approx"]}

    def summary(self):
        ex = self.exact_fields()
        bad = {k: v for k, v in ex.items() if v["mismatch"]}
        first = min((v["first"] for v in bad.values()), default=None)
        return {"fields": len(ex), "mismatching_fields": sorted(bad), "first_mismatch_frame": first}


def _eq(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype.kind == "f":
        return a.tobytes() == b.astype(a.dtype).tobytes()  # bit-exact, NaN == NaN
    return bool(np.array_equal(a, b))


def _cmp_struct(rep, sec, frame, h, r, skip=(), remap=None):
    for name in h.dtype.names:
        if name in skip:
            continue
        hv, rv = h[name], r[name]
        if remap and name in remap:
            hv = remap[name](hv)
        rep.add(f"{sec}.{name}", frame, _eq(hv, rv), _approx(sec, name))


def _cmp_list(rep, sec, frame, h, r, idmap, keys, skip=(), approx_fields=(), sort_by=None):
    rep.add(f"{sec}.count", frame, len(h) == len(r))
    if len(h) != len(r):
        return
    h = h.copy()
    for k in keys:
        h[k] = [idmap(x) for x in h[k]]
    if sort_by:
        h = np.sort(h, order=sort_by)
        r = np.sort(r, order=sort_by)
    for name in h.dtype.names:
        if name in skip:
            continue
        rep.add(f"{sec}.{name}", frame, _eq(h[name], r[name]), _approx(sec, name) or name in approx_fields)


OBJECT_PARAM_TYPES = ("entity", "object", "group", "mesh", "camera", "light", "sprite", "material", "texture")


class ArrayTypes:
    def __init__(self, full: P.LevelMessage):
        self.types = {}
        for a in full.arrays:
            for c, col in enumerate(a.columns):
                self.types[(a.ckid, c)] = (col.type, (col.param_type or "").lower())

    def is_object(self, arr, col, ctype):
        if ctype == P.ARR_OBJECT:
            return True
        t = self.types.get((arr, col))
        return ctype == P.ARR_PARAMETER and t is not None and any(s in t[1] for s in OBJECT_PARAM_TYPES)


def cmp_obs(rep: Report, frame: int, oh: P.Observation, orr: P.Observation, idmap: IdMap, at_h: ArrayTypes):
    rep.add("HDR.frames_done", frame, oh.frames_done == orr.frames_done)
    rep.add("HDR.flags", frame, (oh.flags & ~16) == orr.flags)
    for sec in ("time", "time_ext", "scor", "scor_ext", "sren", "sren_ext", "ball", "cams", "glct"):
        h, r = getattr(oh, sec), getattr(orr, sec)
        S = sec.upper()
        rep.add(f"{S}.present", frame, (h is None) == (r is None))
        if h is None or r is None:
            continue
        if sec == "ball":
            remap = {"ckid": idmap}
            hb, rb = h.copy(), r.copy()
            rep.add("BALL.core_flags", frame, int(hb["core_flags"]) & 0xC == int(rb["core_flags"]) & 0xC)
            rep.add("BALL.core_state", frame, int(hb["core_state"]) & 0xFF == int(rb["core_state"]) & 0xFF)
            rep.add("BALL.core_flags_raw", frame, int(hb["core_flags"]) == int(rb["core_flags"]), _approx("BALL", "core_flags_raw"))
            rep.add("BALL.core_state_raw", frame, int(hb["core_state"]) == int(rb["core_state"]), _approx("BALL", "core_state_raw"))
            _cmp_struct(rep, "BALL", frame, hb, rb, skip=("core_flags", "core_state"), remap=remap)
        elif sec == "cams":
            rep.add("CAMS.ckid", frame, [idmap(x) for x in h["cams"]["ckid"]] == list(r["cams"]["ckid"]))
            rep.add("CAMS.matrix", frame, _eq(h["cams"]["matrix"], r["cams"]["matrix"]))
            rep.add("CAMS.attached", frame, idmap(h["attached"]) == int(r["attached"]))
        elif sec == "glct":
            rep.add("GLCT.bb", frame, idmap(h["bb"]) == int(r["bb"]))
            rep.add("GLCT.pulses", frame, int(h["pulses"]) == int(r["pulses"]))
            rep.add("GLCT.groups", frame, _eq(h["groups"], r["groups"]), _approx("GLCT", "groups"))
        elif sec in ("scor", "sren"):
            # Energy column 4 (Timefactor) is a parameter cell: its raw value is a pointer, not comparable
            ptr = [i for i in range(6) if int(h["types"][i]) == P.ARR_PARAMETER]
            keep = [i for i in range(6) if i not in ptr]
            rep.add(f"{S}.energy", frame, _eq(h["energy"][keep], r["energy"][keep]))
            rep.add(f"{S}.energy_param_ptr", frame, _eq(h["energy"][ptr], r["energy"][ptr]), _approx(S, "energy_param_ptr"))
            _cmp_struct(rep, S, frame, h, r, skip=("energy",))
        else:
            _cmp_struct(rep, S, frame, h, r)
    _cmp_list(rep, "FRIC", frame, oh.fric, orr.fric, idmap, ["partner"])
    hm, rm = oh.mind.copy(), orr.mind.copy()
    if len(hm) == len(rm) and len(hm):
        rep.add("MIND.plane_w", frame, _eq(hm["plane"][:, 3], rm["plane"][:, 3]), _approx("MIND", "plane_w"))  # not maintained (residual)
        hm["plane"][:, 3] = 0
        rm["plane"][:, 3] = 0
    _cmp_list(rep, "MIND", frame, hm, rm, idmap, ["partner"])
    hb = oh.body.copy()
    rb = orr.body.copy()
    rep.add("BODY.core_state_raw", frame, len(hb) == len(rb) and _eq(np.sort(hb["core_state"]), np.sort(rb["core_state"])), _approx("BODY", "core_state_raw"))
    if len(hb) == len(rb):
        hb["core_state"] &= 0xFF
        rb["core_state"] &= 0xFF
    _cmp_list(rep, "BODY", frame, hb, rb, idmap, ["ckid"], sort_by="ckid")
    _cmp_list(rep, "WTCH", frame, oh.wtch, orr.wtch, idmap, ["ckid"], sort_by="ckid")
    rep.add("EVTS", frame, _eq(oh.events, orr.events))
    rep.add("PHYS", frame, _eq(oh.phys, orr.phys))
    rep.add("TRAC.present", frame, (oh.trac is None) == (orr.trac is None))
    if oh.cams is not None and orr.cams is not None:
        for k, name in enumerate(P.CAM_NAMES):
            rep.add(f"CAMS.matrix.{name}", frame, _eq(oh.cams["cams"][k]["matrix"], orr.cams["cams"][k]["matrix"]))

    def cells(o, mapped):
        out = []
        for c in o.arrs:
            arr = idmap(c.array) if mapped else c.array
            data = c.data
            if mapped and len(data) == 4 and at_h.is_object(c.array, c.col, c.col_type):
                v = int.from_bytes(data, "little")
                if v:
                    data = int(idmap(v) & 0xFFFFFFFF).to_bytes(4, "little")
            out.append((arr, c.row, c.col, c.col_type, data))
        return sorted(out)
    rep.add("ARRS", frame, cells(oh, True) == cells(orr, False))


# --------------------------------------------------------------------------------------------- level

def cmp_level(hl, rt, idmap: IdMap) -> dict:
    hf, hi = hl
    rf, ri = rt
    out = {}
    eh = {idmap(e.ckid): e for e in hf.entities if e.ckid and e.name}
    er = {e.ckid: e for e in rf.entities if e.ckid and e.name}
    common = sorted(set(eh) & set(er))
    mat = sorted({eh[k].name for k in common if not _eq(eh[k].matrix, er[k].matrix)})
    attr = sorted({eh[k].name for k in common if (eh[k].contact_id, eh[k].coll_id, eh[k].visible, eh[k].class_id)
                   != (er[k].contact_id, er[k].coll_id, er[k].visible, er[k].class_id)})
    floor = sum(eh[k].floor != er[k].floor for k in common)
    out["entities"] = {"headless": len(eh), "retail": len(er), "only_headless": len(set(eh) - set(er)),
                       "only_retail": len(set(er) - set(eh)), "matrix_mismatch": mat[:40],
                       "matrix_mismatch_count": len(mat), "attr_mismatch": attr[:20],
                       "floor_attr_raw_mismatch": floor}  # Floor: a parameter whose raw bytes differ by representation
    targets = set(idmap.m.values())
    gh = {g.name: sorted(x for x in (idmap(m) for m in g.members) if x >= 0) for g in hf.groups}
    gr = {g.name: sorted(int(m) for m in g.members if int(m) in targets) for g in rf.groups}
    out["groups"] = {"headless": len(gh), "retail": len(gr),
                     "member_mismatch": sorted(k for k in set(gh) & set(gr) if gh[k] != gr[k])[:20],
                     "only_headless": sorted(set(gh) - set(gr))[:20], "only_retail": sorted(set(gr) - set(gh))[:20],
                     "note": "members compared where both ids map (entities, groups, arrays)"}
    ah = {a.name: a for a in hf.arrays}
    ar = {a.name: a for a in rf.arrays}
    at = ArrayTypes(hf)
    norm = lambda s: (s or "").replace(" ", "").lower()  # noqa: E731  ("2dEntity" / "2D Entity")
    res = {"gameplay": [], "other": []}
    for name in sorted(set(ah) & set(ar)):
        a, b = ah[name], ar[name]
        bucket = res["gameplay" if name in GAMEPLAY_ARRAYS else "other"]
        if [(c.name, c.type, norm(c.param_type)) for c in a.columns] != [(c.name, c.type, norm(c.param_type)) for c in b.columns]:
            bucket.append([name, "columns"])
            continue
        if len(a.cells) != len(b.cells):
            bucket.append([name, "rows"])
            continue
        for r_, (rowa, rowb) in enumerate(zip(a.cells, b.cells)):
            for c_, (x, y) in enumerate(zip(rowa, rowb)):
                if len(x) == 4 and at.is_object(a.ckid, c_, a.columns[c_].type) and int.from_bytes(x, "little"):
                    x = int(idmap(int.from_bytes(x, "little")) & 0xFFFFFFFF).to_bytes(4, "little")
                if x != y:
                    bucket.append([name, r_, c_])
    out["arrays"] = {"headless": len(ah), "retail": len(ar), "gameplay_mismatch": res["gameplay"][:20],
                     "other_mismatch_count": len(res["other"]),
                     "other_mismatch_arrays": sorted({x[0] for x in res["other"]}),
                     "only_headless": sorted(set(ah) - set(ar)), "only_retail": sorted(set(ar) - set(ah))}

    def recs(lms, mp):
        d = defaultdict(list)
        for lm in lms:
            for p in lm.phys:
                d[p.frame].append((mp(p.ckid), p.unphysicalize, p.fixed, p.friction, p.elasticity, p.mass,
                                   p.collision_group, p.frozen, p.collision, p.calc_mass_center, p.linear_damping,
                                   p.rot_damping, p.surface, tuple(p.mass_center), tuple(mp(x) for x in p.convex),
                                   tuple(mp(x) for x in p.concave),
                                   tuple((tuple(np.asarray(c).tolist()), r) for c, r in p.balls)))
        return {k: sorted(v, key=repr) for k, v in d.items()}
    ident = lambda x: int(x)  # noqa: E731
    # the full message repeats the records of the increments: the increments alone are the complete history
    rh, rr = recs(hi or [hf], idmap), recs(ri or [rf], ident)
    out["phys"] = {"headless": sum(map(len, rh.values())), "retail": sum(map(len, rr.values())),
                   "frames_mismatch": sorted(f for f in set(rh) | set(rr) if rh.get(f) != rr.get(f))[:20],
                   "note": "records compared per frame as a multiset (the Physicalize order inside a frame differs)"}
    mh = {(idmap(m.ckid), idmap(m.mesh)): m for lm in [hf] + hi for m in lm.meshes}
    mr = {(m.ckid, m.mesh): m for lm in [rf] + ri for m in lm.meshes}
    out["meshes"] = {"headless": len(mh), "retail": len(mr), "only_headless": len(set(mh) - set(mr)),
                     "only_retail": len(set(mr) - set(mh)),
                     "geometry_mismatch": sum(not (_eq(mh[k].vertices, mr[k].vertices) and _eq(mh[k].faces, mr[k].faces))
                                              for k in set(mh) & set(mr)),
                     "matrix_mismatch": sum(not _eq(mh[k].matrix, mr[k].matrix) for k in set(mh) & set(mr))}
    dh = {idmap(c): (lo.tolist(), hi_.tolist()) for c, lo, hi_ in hf.depth}
    dr = {int(c): (lo.tolist(), hi_.tolist()) for c, lo, hi_ in rf.depth}
    out["depth"] = {"headless": len(dh), "retail": len(dr), "mismatch": sum(dh.get(k) != v for k, v in dr.items())}
    wh = sorted((idmap(c), g) for c, g in hf.watch)
    wr = sorted((int(c), g) for c, g in rf.watch)
    out["watch"] = {"headless": len(wh), "retail": len(wr), "equal": wh == wr}
    ih = {(idmap(o.ckid)) for lm in [hf] + hi for o in lm.ivp}
    ir = {o.ckid for lm in [rf] + ri for o in lm.ivp}
    out["ivpo"] = {"headless": len(ih), "retail": len(ir), "equal_entities": ih == ir}
    out["unmapped_ids"] = len(idmap.unmapped)
    return out


# --------------------------------------------------------------------------------------------- features

class Feeder:
    """A GameTracker fed as AgentSession._ingest does: the full level message at StartLevel, increments in order."""

    def __init__(self, level, full, inc, fcfg):
        self.t = GameTracker(level)
        self.full, self.inc = full, list(inc)
        self.k = 0
        self.loaded = False
        self.fx = FeatureExtractor(fcfg)

    def feed(self, o: P.Observation):
        if o.has_event(Ev.StartLevel) and self.full is not None:
            self.t.merge_level(self.full)
            self.loaded = True
        if self.loaded and (len(o.phys) or o.has_event(Ev.StartLevel)):
            while self.k < len(self.inc) and self.inc[self.k].frames_done <= o.frames_done:
                self.t.merge_level(self.inc[self.k])
                self.k += 1
        self.t.update(o)
        if o.has_event(Ev.BallNavActive):
            self.fx.reset_episode()

    def features(self):
        if not self.loaded or self.t.obs is None or self.t.obs.ball is None:
            return None
        return self.fx.compute(self.t)[0]


# --------------------------------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--headless", required=True, type=Path)
    ap.add_argument("--record", required=True)
    ap.add_argument("--dataset", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-features", action="store_true")
    ap.add_argument("--phys2002", action="store_true", help="the 2002-DLL host: the IVP fields the x87-soft engine "
                    "approximates count as exact (APPROX_PHYS2002); the hello must name phys2002")
    ap.add_argument("--feature-every", type=int, default=1)
    ap.add_argument("--features", default="{}", help="FeatureConfig overrides as JSON (e.g. the trainer's: "
                    '{"hs_block": false, "camera": "world"} under --world-actions)')
    a = ap.parse_args(argv)
    if a.phys2002:
        hello = json.loads((a.headless / "summary.json").read_text()).get("hello", "")
        if "phys2002" not in hello:
            ap.error(f"--phys2002: the run's hello {hello!r} does not name phys2002 (the source physics ran?)")
        _APPROX["set"] = APPROX_PHYS2002
    ds = a.dataset or (instances.runtime_dir() / "datasets" / "tas-v1")
    rd = ds / a.record
    meta = json.loads((rd / "meta.json").read_text(encoding="utf-8"))
    level = int(meta["level"])
    hl_raw, rt_raw = load_headless(a.headless), load_retail(rd)
    hl_lvl, rt_lvl = level_messages(a.headless), level_messages(rd)
    idmap = IdMap(hl_lvl, rt_lvl)
    at_h = ArrayTypes(hl_lvl[0])
    rep = Report()
    hobs = [P.parse_observation(b) for b in hl_raw]
    robs = [P.parse_observation(b) for b in rt_raw]
    rmap = {o.frames_done: o for o in robs if not o.final}
    rfinal = robs[-1] if robs and robs[-1].final else None
    frames = []
    rep0 = Report()  # the frame-0 observation: headless builds it after its hidden load ticks (docs/headless.md)
    for o in hobs:
        r = rfinal if o.final else rmap.get(o.frames_done)
        if r is None:
            rep.add("HDR.retail_has_frame", o.frames_done, False)
            continue
        cmp_obs(rep0 if o.frames_done == 0 and not o.final else rep, o.frames_done, o, r, idmap, at_h)
        frames.append(o.frames_done)
    level_cmp = cmp_level(hl_lvl, rt_lvl, idmap)
    # events / SR / HS
    ev_h = [(int(e["frame"]), int(e["code"]), int(e["param"])) for o in hobs for e in o.events]
    ev_r = [(int(e["frame"]), int(e["code"]), int(e["param"])) for o in robs for e in o.events]
    first_ev = next((i for i, (x, y) in enumerate(zip(ev_h, ev_r)) if x != y), None)
    fl_h = tas.load_raw(a.headless / "framelog.bin")
    fl_r = np.load(rd / "raw.npz")["framelog"]
    sren_h, sren_r = hobs[-1].sren, robs[-1].sren

    def sren_view(s):
        if s is None:
            return None
        out = {k: (float(s[k]) if s.dtype[k].kind == "f" else s[k].tolist()) for k in s.dtype.names if k != "bml_hs"}
        out["energy"] = [v for i, v in enumerate(out["energy"]) if int(s["types"][i]) != P.ARR_PARAMETER]  # pointer cell
        return out
    lf_h = next((n for n, c, _ in ev_h if c == Ev.LevelFinish), None)
    sr_py_h = sr_timer(fl_h[:lf_h], ev_h) if lf_h else None
    events = {"headless": len(ev_h), "retail": len(ev_r), "equal": ev_h == ev_r,
              "first_difference": None if first_ev is None else {"index": first_ev, "headless": ev_h[first_ev],
                                                                 "retail": ev_r[first_ev]},
              "counts_headless": dict(Counter(P.EVENT_NAMES.get(c, str(c)) for _, c, _ in ev_h)),
              "levelfinish": lf_h, "framelog_equal": bool(len(fl_h) == len(fl_r) and fl_h.tobytes() == fl_r.tobytes()),
              "sren_equal": sren_view(sren_h) == sren_view(sren_r), "sren_headless": sren_view(sren_h),
              "sr_python_at_levelfinish": sr_py_h}
    # features
    feat = None
    if not a.no_features:
        fcfg = FeatureConfig(**json.loads(a.features))
        fh = Feeder(level, *hl_lvl, fcfg)
        fr = Feeder(level, *rt_lvl, FeatureConfig(**json.loads(a.features)))
        n = bad = 0
        first = None
        worst = 0.0
        names = None
        bad_names = Counter()
        from ballrain.features import feature_names
        names = feature_names(fh.fx)
        for o in hobs:
            r = rfinal if o.final else rmap.get(o.frames_done)
            if r is None:
                continue
            fh.feed(o)
            fr.feed(r)
            if o.frames_done % a.feature_every:
                continue
            vh, vr = fh.features(), fr.features()
            if vh is None or vr is None:
                if (vh is None) != (vr is None):
                    bad += 1
                continue
            n += 1
            if not np.array_equal(vh.view(np.uint32), vr.view(np.uint32)):
                bad += 1
                d = np.abs(vh.astype(np.float64) - vr.astype(np.float64))
                worst = max(worst, float(np.nanmax(d)))
                for j in np.flatnonzero(vh.view(np.uint32) != vr.view(np.uint32)):
                    bad_names[names[j]] += 1
                if first is None:
                    first = o.frames_done
        feat = {"compared": n, "mismatch_frames": bad, "first_mismatch": first, "max_abs_diff": worst,
                "mismatching_entries": dict(bad_names.most_common(20)), "version": fcfg.version,
                "config": json.loads(a.features), "size": int(fh.fx.size)}
    report = {"record": a.record, "headless_dir": str(a.headless), "observations_headless": len(hobs),
              "observations_retail": len(robs), "compared_frames": len(frames), "summary": rep.summary(),
              "fields": rep.fields, "frame0_mismatch": sorted(k for k, v in rep0.fields.items() if v["mismatch"]),
              "level": level_cmp, "events": events, "features": feat,
              "unmapped_ids": len(idmap.unmapped)}
    txt = json.dumps(report, indent=1, default=str)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(txt, encoding="utf-8")
    s = report["summary"]
    print(json.dumps({"record": a.record, "compared_frames": len(frames), **s,
                      "approx_mismatch": {k: v["mismatch"] for k, v in rep.fields.items() if v["approx"] and v["mismatch"]},
                      "events_equal": events["equal"], "framelog_equal": events["framelog_equal"],
                      "sren_equal": events["sren_equal"], "features": feat, "level": level_cmp}, indent=1, default=str))
    return 0 if not s["mismatching_fields"] and events["equal"] and (feat is None or feat["mismatch_frames"] == 0) else 1


if __name__ == "__main__":
    sys.exit(main())

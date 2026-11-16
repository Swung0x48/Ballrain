"""Compare ball-vs-road contact from (1) IVP memory, (2) game-logic contact BB, (3) Python geometry, frame by frame."""
import json
import math
import struct
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, ".")
import geom  # noqa: E402
import pf  # noqa: E402

out_dir = sys.argv[1]
frames, events = pf.parse(out_dir + "/frames.txt", want={"K", "BW", "C", "D", "DT", "CC", "B", "P"})
meshes = geom.load_meshes(out_dir + "/meshes.txt")

# static: contact ids and floor groups
ccid, cdid, groups_of, names = {}, {}, defaultdict(set), {}
with open(out_dir + "/static_start.txt", encoding="latin-1") as f:
    grp = None
    for l in f:
        if l.startswith("GROUP "):
            grp = l.split(" ")[2]
        elif l.startswith("  - ") and grp:
            groups_of[int(l.split(" ")[3])].add(grp)
        elif l.startswith("ENT "):
            grp = None
            p = l.rstrip("\n").split(" ")
            eid = int(p[1])
            names[eid] = p[3]
            s = l
            if "@Continuous Contact ID=" in s:
                ccid[eid] = int(s.split("@Continuous Contact ID=")[1].split(" ")[0])
            if "@Coll Detection ID=" in s:
                cdid[eid] = int(s.split("@Coll Detection ID=")[1].split(" ")[0])
        else:
            grp = grp if l.startswith("  - ") else None

FLOORGROUPS = ("Phys_Floors", "Phys_FloorRails", "Phys_FloorStopper")


def surface(eid):
    g = [x for x in FLOORGROUPS if x in groups_of.get(eid, ())]
    return (g[0] if g else "other") + ":" + str(ccid.get(eid, 0))


world = geom.StaticWorld(meshes, lambda m: True)
mov_mesh = {e: lst for e, lst in meshes.items() if not lst[0]["fixed"]}
paper = [m for lst in meshes.values() for m in lst if m["entname"] == "Ball_Paper"]
r_paper = float(np.linalg.norm(paper[0]["V"], axis=1).max()) if paper else 2.0
r_paper_mean = float(np.linalg.norm(paper[0]["V"], axis=1).mean()) if paper else 2.0
print(f"static tris {len(world.A)}, boundary edges {len(world.S0)}, paper radius max {r_paper:.3f} mean {r_paper_mean:.3f}")


def ang(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


recs = []
for fr in frames:
    K = fr.get("K")
    if not K or "pos" not in K or fr["ball"] == "-":
        continue
    btype = fr["ball"]
    r = 2.0 if btype != "Ball_Paper" else r_paper_mean
    p = np.array(K["pos"])
    pnow = np.array(fr["BW"][9:12]) if "BW" in fr else p
    rec = {"idx": fr["idx"], "ball": btype, "speed": float(np.linalg.norm(K["s"])), "vy": K["s"][1]}
    # (1) IVP friction contacts
    fr_part, lt_n, lt_p, gaps = [], [], [], []
    for c in fr["C"]:
        fr_part.append(int(c["pid"]))
        raw = c["raw"]
        if len(raw) >= 0x5C:
            gaps.append(pf.f32(raw, 0x58))
        lt = c.get("lt")
        if lt and lt != "??":
            b = bytes.fromhex(lt)
            lt_n.append(struct.unpack_from("<3f", b, 0))
            lt_p.append(struct.unpack_from("<3d", b, 0x20))
    rec["nfric"] = len(fr["C"])
    rec["fric_partners"] = fr_part
    # (1b) mindists
    mds = []
    for d in fr["D"]:
        raw = d["raw"]
        if len(raw) < 0x78:
            continue
        mds.append((pf.f32(raw, 0x54), struct.unpack_from("<3f", raw, 0x68), int(d["pid"]), int(d["pst"])))
    mds.sort(key=lambda x: x[0])
    rec["md_n"] = len(mds)
    rec["md_min"] = mds[0][0] if mds else float("inf")
    rec["md_pst"] = mds[0][3] if mds else -1
    rec["md_pid"] = mds[0][2] if mds else 0
    rec["md_plane"] = mds[0][1] if mds else None
    # (2) game logic: active contact BB
    gl = None
    for cc in fr["CC"]:
        t = cc.split(" ")
        d = pf.kv(t)
        if d.get("data", "00000000") != "00000000" and "*10" in d:
            b = bytes.fromhex(d["*10"])
            gl = [(struct.unpack_from("<i", b, 8 * k)[0], struct.unpack_from("<i", b, 8 * k + 4)[0]) for k in range(3)]
            rec["gl_mask"] = int(d["mask"], 16)
    rec["gl"] = gl
    # (3) geometry, static at the PSI position
    g = world.nearest(p, 8.0)
    if g:
        rec["geo_gap"] = g["d"] - r
        rec["geo_ent"] = g["ent"]
        rec["geo_reg"] = g["reg"]
        rec["geo_n"] = (p - g["q"]) / max(g["d"], 1e-9)
        rec["geo_nface"] = g["n_face"]
        rec["edge"] = world.edge_distance(g["q"]) if abs(g["n_face"][1]) > 0.5 else float("nan")
        # touching set: all static triangles within eps of the ball surface
        rec["geo_touch_ents"] = sorted(set(world.E[g["all_idx"][g["all_d"] - r < 0.05]].tolist()))
    else:
        rec["geo_gap"] = float("inf")
    # (3b) movable bodies at the current (interpolated) poses
    best = float("inf"); bent = 0
    for bl in fr["B"]:
        t = bl.split(" ")
        name, eid = t[0], int(t[1])
        if name.startswith("Ball_") or eid not in mov_mesh:
            continue
        Wi = t.index("W")
        W = geom.W12_to_44([float(x) for x in t[Wi + 1:Wi + 13]])
        for m in mov_mesh[eid]:
            Vw = geom.to_world(m["V"], W)
            T = m["T"]
            if len(T) == 0 or np.min(np.linalg.norm(Vw - pnow, axis=1)) > 40:
                continue
            Q, reg = geom.closest_pt_tris(pnow, Vw[T[:, 0]], Vw[T[:, 1]], Vw[T[:, 2]])
            dd = np.linalg.norm(pnow - Q, axis=1).min() - r
            if dd < best:
                best, bent = dd, eid
    rec["mov_gap"] = best
    rec["mov_ent"] = bent
    if lt_n:
        rec["lt_n"] = lt_n[0]
        rec["lt_pdist"] = float(np.linalg.norm(np.array(lt_p[0]) - p))
    rec["cp_gap"] = gaps[0] if gaps else None
    recs.append(rec)

N = len(recs)
print("frames with ball", N)


def conf(a, b):
    c = Counter(zip(a, b))
    return {"TT": c[(True, True)], "TF": c[(True, False)], "FT": c[(False, True)], "FF": c[(False, False)]}


res = {"frames": N, "r_paper_mean": r_paper_mean}
ivp = [r["nfric"] > 0 for r in recs]
for eps in (0.02, 0.05, 0.1, 0.2):
    geo = [min(r["geo_gap"], r["mov_gap"]) <= eps for r in recs]
    md = [r["md_min"] <= eps for r in recs]
    res[f"fric_vs_geo@{eps}"] = conf(ivp, geo)
    res[f"md_vs_geo@{eps}"] = conf(md, geo)
    res[f"fric_vs_md@{eps}"] = conf(ivp, md)
glc = [(r["gl"] is not None and sum(n for _, n in r["gl"]) > 0) for r in recs]
gla = [(r["gl"] is not None and any(a for a, _ in r["gl"])) for r in recs]
res["fric_vs_gl_number"] = conf(ivp, glc)
res["fric_vs_gl_active"] = conf(ivp, gla)

# md length vs geometric gap when the closest mindist partner is static and the geometric nearest is that entity
err = [r["md_min"] - r["geo_gap"] for r in recs if r["md_n"] and r["md_pid"] == r.get("geo_ent") and r["md_min"] < 1.0
       and r["ball"] != "Ball_Paper"]
if err:
    e = np.abs(err)
    res["md_len_minus_geo_gap"] = {"n": len(e), "median": float(np.median(e)), "p95": float(np.percentile(e, 95)), "max": float(e.max())}

# normals
a1 = [ang(r["lt_n"], r["geo_n"]) for r in recs if "lt_n" in r and r["geo_gap"] < 0.1 and r["ball"] != "Ball_Paper"]
a1s = [min(x, 180 - x) for x in a1]
a2 = [ang(r["md_plane"], r["geo_n"]) for r in recs if r["md_plane"] and r["md_min"] < 0.1 and r["md_pid"] == r.get("geo_ent")
      and r["ball"] != "Ball_Paper"]
a3 = [ang(r["lt_n"], r["md_plane"]) for r in recs if "lt_n" in r and r["md_plane"] and r["md_min"] < 0.1]
for k, v in (("lt_vs_geo_deg(sign-free)", a1s), ("lt_vs_geo_deg(raw)", a1), ("mdplane_vs_geo_deg", a2), ("lt_vs_mdplane_deg", a3)):
    if v:
        v = np.array(v)
        res[k] = {"n": len(v), "median": float(np.median(v)), "p95": float(np.percentile(v, 95)), "max": float(v.max())}
fl = [r["lt_pdist"] for r in recs if "lt_pdist" in r and r["ball"] != "Ball_Paper"]
if fl:
    fl = np.array(fl)
    res["lt_contact_point_dist_from_center"] = {"median": float(np.median(fl)), "p5": float(np.percentile(fl, 5)), "p95": float(np.percentile(fl, 95))}

# surface agreement: IVP fric partner vs GL group vs geometric nearest entity
sa = Counter()
for r in recs:
    if r["nfric"] == 0:
        continue
    ivps = sorted(set(ccid.get(e, 0) for e in r["fric_partners"]))
    gls = sorted(k + 1 for k, (_, n) in enumerate(r["gl"] or []) if n > 0)
    ivp_nonzero = [x for x in ivps if x > 0]
    sa["ivp==gl" if ivp_nonzero == gls else "ivp!=gl"] += 1
    if r.get("geo_ent"):
        sa["fricpartner_has_geo_ent" if r["geo_ent"] in r["fric_partners"] else "fricpartner_not_geo_ent"] += 1
res["surface_agreement"] = dict(sa)
res["fric_partner_surface"] = dict(Counter(surface(e) for r in recs for e in r["fric_partners"]))
# feature type at contact: IVP pst (0 vertex,1 edge,2 face) of the closest mindist vs geometric edge distance
feat = defaultdict(list)
for r in recs:
    if r["md_min"] < 0.05 and r["md_pst"] >= 0 and not math.isnan(r.get("edge", float("nan"))):
        feat[r["md_pst"]].append(r["edge"])
res["edge_dist_by_ivp_feature"] = {str(k): {"n": len(v), "median": float(np.median(v)), "p10": float(np.percentile(v, 10)),
                                            "p90": float(np.percentile(v, 90))} for k, v in feat.items()}
eb = [r["edge"] for r in recs if r["nfric"] > 0 and not math.isnan(r.get("edge", float("nan")))]
if eb:
    res["edge_dist_in_contact"] = {"n": len(eb), "p5": float(np.percentile(eb, 5)), "median": float(np.median(eb)),
                                   "share_lt_2": float(np.mean(np.array(eb) < 2.0))}
res["mov_contact_frames"] = int(sum(1 for r in recs if r["mov_gap"] <= 0.1))
res["mov_contact_entities"] = dict(Counter(names.get(r["mov_ent"], str(r["mov_ent"])) for r in recs if r["mov_gap"] <= 0.1))
# by ball type
res["by_ball"] = {}
for bt in sorted(set(r["ball"] for r in recs)):
    rr = [r for r in recs if r["ball"] == bt]
    res["by_ball"][bt] = {"frames": len(rr), "fric": sum(r["nfric"] > 0 for r in rr),
                          "fric_vs_geo@0.05": conf([r["nfric"] > 0 for r in rr], [min(r["geo_gap"], r["mov_gap"]) <= 0.05 for r in rr]),
                          "fric_vs_md@0.05": conf([r["nfric"] > 0 for r in rr], [r["md_min"] <= 0.05 for r in rr]),
                          "md_vs_geostatic@0.05": conf([r["md_min"] <= 0.05 for r in rr if r["md_pid"] in world.names or r["md_n"] == 0], [r["geo_gap"] <= 0.05 for r in rr if r["md_pid"] in world.names or r["md_n"] == 0])}
# disagreement samples
dis = [(r["idx"], r["ball"], r["nfric"], round(r["md_min"], 4), round(r["geo_gap"], 4), round(r["mov_gap"], 4), round(r["vy"], 3))
       for r in recs if (r["nfric"] > 0) != (min(r["geo_gap"], r["mov_gap"]) <= 0.05)]
res["disagree_fric_geo@0.05_samples"] = dis[:40]
res["disagree_count"] = len(dis)
gaps = [r["cp_gap"] for r in recs if r["cp_gap"] is not None]
if gaps:
    res["contact_point_last_gap_len"] = {"median": float(np.median(gaps)), "p95": float(np.percentile(gaps, 95)), "min": float(np.min(gaps))}
md_at_fric = [r["md_min"] for r in recs if r["nfric"] > 0 and r["ball"] != "Ball_Paper"]
if md_at_fric:
    res["md_min_when_fric"] = {"median": float(np.median(md_at_fric)), "p95": float(np.percentile(md_at_fric, 95)), "max": float(np.max(md_at_fric))}
json.dump(res, open(out_dir + "/compare.json", "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))

# ---- extra stats
ex = {}
fl_edge = defaultdict(list)
for r in recs:
    if r["nfric"] > 0 and not math.isnan(r.get("edge", float("nan"))):
        s = surface(r["fric_partners"][0]).split(":")[0]
        fl_edge[s].append(r["edge"])
ex["edge_dist_in_contact_by_surface"] = {k: {"n": len(v), "p5": float(np.percentile(v, 5)), "median": float(np.median(v)), "p95": float(np.percentile(v, 95))} for k, v in fl_edge.items()}
ex["nfric_hist"] = dict(Counter(r["nfric"] for r in recs))
ex["md_n_hist"] = dict(Counter(min(r["md_n"], 10) for r in recs))
mdl = [d for r in recs for d in [r["md_min"]] if d < float("inf")]
ex["md_min_max_seen"] = float(max(mdl)) if mdl else None
ex["frames_without_md"] = sum(1 for r in recs if r["md_n"] == 0)
nfr_rail = Counter(r["nfric"] for r in recs if r["nfric"] and all(surface(e).startswith("Phys_FloorRails") for e in r["fric_partners"]))
ex["nfric_on_rails"] = dict(nfr_rail)
# landing: frames where fric goes 0->1: md_min in previous frames
land = []
for i in range(1, len(recs)):
    if recs[i]["nfric"] > 0 and recs[i - 1]["nfric"] == 0 and recs[i]["idx"] == recs[i - 1]["idx"] + 1:
        land.append(recs[i - 1]["md_min"])
ex["md_min_frame_before_fric_on"] = {"n": len(land), "median": float(np.median(land)) if land else None, "p95": float(np.percentile(land, 95)) if land else None}
json.dump(ex, open(out_dir + "/compare_extra.json", "w"), indent=1, default=float)
print(json.dumps(ex, default=float))
ft = Counter()
for r in recs:
    if r["ball"] != "Ball_Paper" and r["geo_gap"] <= 0.05 and r["md_min"] > 0.05:
        ft[(names.get(r.get("geo_ent"), "?"), surface(r.get("geo_ent", 0)))] += 1
print("FT_geo_entities", ft.most_common(8))
fti = [(r["idx"], r["ball"], round(r["geo_gap"], 4), round(r["md_min"], 3), r["md_n"], r["nfric"], round(r["speed"], 3)) for r in recs if r["ball"] != "Ball_Paper" and r["geo_gap"] <= 0.05 and r["md_min"] > 0.05]
print("FT idx", fti[:5], fti[-5:], len(fti))
mm = Counter()
for r in recs:
    if r["nfric"] == 0: continue
    ivps = sorted(set(ccid.get(e, 0) for e in r["fric_partners"]))
    gls = sorted(k + 1 for k, (_, n) in enumerate(r["gl"] or []) if n > 0)
    if [x for x in ivps if x > 0] != gls:
        for e in r["fric_partners"]:
            mm[(names.get(e, "NOT_IN_STATIC:%d" % e), ccid.get(e, 0), tuple(gls))] += 1
print("IVP_NE_GL", mm.most_common(10))
st = [e for e in world.names if "Stopper" in world.names[e]]
if st:
    sel = np.isin(world.E, st)
    A, B, C = world.A[sel], world.B[sel], world.C[sel]
    best = min(((np.linalg.norm(np.array(fr["K"]["pos"]) - geom.closest_pt_tris(np.array(fr["K"]["pos"]), A, B, C)[0], axis=1).min() - 2.0, fr["idx"]) for fr in frames[::5] if fr.get("K") and "pos" in fr["K"]), default=None)
    print("STOPPER min gap", best, [world.names[e] for e in st])
mmi = [r["idx"] for r in recs if r["nfric"] and [x for x in sorted(set(ccid.get(e, 0) for e in r["fric_partners"])) if x > 0] != sorted(k + 1 for k, (_, n) in enumerate(r["gl"] or []) if n > 0)]
print("IVP_NE_GL_idx", mmi[:3], mmi[-3:], "balls", Counter(r["ball"] for r in recs if r["idx"] in set(mmi)))

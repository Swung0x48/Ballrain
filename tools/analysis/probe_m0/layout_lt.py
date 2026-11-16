import sys, struct, math
sys.path.insert(0, '.')
import pf
# parse C lines with lt
frames, ev = pf.parse(sys.argv[1], want={"C","K"})
cnt = {}; pts = {}; ptsd = {}; n = 0
for fr in frames:
    K = fr.get("K")
    if not K: continue
    for c in fr["C"]:
        lt = c.get("lt")
        if not lt or lt == "??": continue
        raw = bytes.fromhex(lt); n += 1
        for off in range(0, len(raw)-12+1, 4):
            v = struct.unpack_from("<3f", raw, off)
            if all(math.isfinite(x) and abs(x) < 1e6 for x in v):
                L = math.sqrt(sum(x*x for x in v))
                if abs(L-1) < 1e-3: cnt[off] = cnt.get(off,0)+1
                if abs(math.dist(v, K["pos"]) - K["xrad"]) < 0.1: pts[off] = pts.get(off,0)+1
        for off in range(0, len(raw)-24+1, 8):
            v = struct.unpack_from("<3d", raw, off)
            if all(math.isfinite(x) and abs(x) < 1e6 for x in v):
                if abs(math.dist(v, K["pos"]) - K["xrad"]) < 0.1: ptsd[off] = ptsd.get(off,0)+1
print("n", n); print("unit f3", sorted(cnt.items())); print("pt f3", sorted(pts.items())); print("pt d3", sorted(ptsd.items()))

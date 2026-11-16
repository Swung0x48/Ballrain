import sys, struct, math
sys.path.insert(0, '.')
import pf
frames, ev = pf.parse(sys.argv[1], want={"C","CT","K","BW","D","DT"})
# find unit-length float triplets in contact point raw, and points ~ radius from the core position
cnt = {}
pts = {}
n=0
for fr in frames:
    K = fr.get("K")
    if not K: continue
    for c in fr["C"]:
        raw = c["raw"]
        if len(raw) < 0xA0: continue
        n += 1
        for off in range(0, 0xA0-12+1, 4):
            v = struct.unpack_from("<3f", raw, off)
            if all(math.isfinite(x) for x in v):
                L = math.sqrt(sum(x*x for x in v))
                if abs(L-1) < 1e-3: cnt[off] = cnt.get(off,0)+1
                d = math.dist(v, K["pos"])
                if abs(d - K["xrad"]) < 0.2: pts[off] = pts.get(off,0)+1
print("contact points", n)
print("unit triplets", sorted(cnt.items()))
print("near radius", sorted(pts.items()))

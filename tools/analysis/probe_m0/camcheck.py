import sys, math
import numpy as np
sys.path.insert(0, '.')
import pf
out = sys.argv[1]
frames, ev = pf.parse(out + "/frames.txt", want={"M"})
prev = None; segs = []; start = None
names = set()
for fr in frames:
    names |= set(fr["M"].keys())
    m = fr["M"].get("Cam_OrientRef")
    if not m: continue
    z = np.array(m[6:9])  # row 2 = local z axis in world
    yaw = math.degrees(math.atan2(z[0], z[2]))
    q = round(yaw / 90) * 90
    moving = abs(yaw - q) > 0.01
    if prev is not None and moving != prev:
        segs.append((fr["idx"], "rot-start" if moving else "rot-end", round(yaw, 2)))
    prev = moving
print("cams", sorted(names))
print(len(segs), segs[:16])
# rotation durations
d = [segs[i+1][0] - segs[i][0] for i in range(len(segs)-1) if segs[i][1] == "rot-start"]
print("rotation frames", d[:20])
# Cam_Pos height above OrientRef
fr = frames[len(frames)//2]
print({k: [round(x,3) for x in v[9:12]] for k, v in fr["M"].items()})

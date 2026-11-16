import sys, math
import numpy as np
sys.path.insert(0, '.')
import pf
frames, ev = pf.parse(sys.argv[1] + "/frames.txt", want={"M"})
a, b = int(sys.argv[2]), int(sys.argv[3])
for fr in frames:
    if a <= fr["idx"] <= b and (fr["idx"] - a) % int(sys.argv[4]) == 0:
        s = []
        for k in ("Cam_OrientRef", "Cam_Orient", "Cam_Target", "Cam_Pos", "InGameCam"):
            m = fr["M"].get(k)
            if not m: continue
            z = np.array(m[6:9]); x = np.array(m[0:3])
            s.append("%s yawz=%.2f yawx=%.2f" % (k[4:], math.degrees(math.atan2(z[0], z[2])), math.degrees(math.atan2(x[0], x[2]))))
        print(fr["idx"], " | ".join(s))

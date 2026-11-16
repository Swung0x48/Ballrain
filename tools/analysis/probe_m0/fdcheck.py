import sys, math
import numpy as np
sys.path.insert(0, '.')
import pf
frames, ev = pf.parse(sys.argv[1] + "/frames.txt", want={"K", "T", "BW"})
prev = None; errs = []; errs2 = []; rel = []
for fr in frames:
    K = fr.get("K")
    if not K or "pos" not in K: prev = None; continue
    if prev and fr["ball"] == prev[2] and K["tpsi"] != prev[0]:
        dt = K["tpsi"] - prev[0]
        v_fd = (np.array(K["pos"]) - prev[1]) / dt
        v_new = np.array(K["s"]); v_old = prev[3]
        e = np.linalg.norm(v_fd - v_new); e_old = np.linalg.norm(v_fd - v_old)
        errs.append(e); errs2.append(e_old); rel.append(np.linalg.norm(v_new))
    prev = (K["tpsi"], np.array(K["pos"]), fr["ball"], np.array(K["s"]))
errs, errs2, rel = map(np.array, (errs, errs2, rel))
print("n", len(errs), "|fd - speed(new)| median %.4g p95 %.4g ; |fd - speed(prev)| median %.4g p95 %.4g ; speed median %.3g" % (np.median(errs), np.percentile(errs, 95), np.median(errs2), np.percentile(errs2, 95), np.median(rel)))
# psi per frame
tp = [fr["K"]["tpsi"] for fr in frames if fr.get("K") and "tpsi" in fr["K"]]
d = np.diff(tp); print("PSI time steps:", sorted(set(np.round(d[d > 0], 6)))[:5], "frames with new PSI share %.3f" % np.mean(d > 0))

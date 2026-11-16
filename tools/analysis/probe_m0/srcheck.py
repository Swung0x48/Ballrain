import sys, struct
import numpy as np
sys.path.insert(0, '.')
import pf
out = sys.argv[1]
frames, ev = pf.parse(out + "/frames.txt", want={"T","A"})
evs = {}
for i, n in ev: evs.setdefault(i, []).append(n)
running = False; acc = 0.0; acc32 = np.float32(0); worst = 0; first = None; n=0
energy = []
for fr in frames:
    i = fr["idx"]
    for n_ in evs.get(i, []):
        if n_ == "StartLevel": acc = 0.0; acc32 = np.float32(0); running = False
        if n_ in ("CounterActive", "UnpauseLevel"): running = True
        if n_ in ("CounterInactive", "PauseLevel"): running = False
    T = fr.get("T")
    if not T: continue
    # BML+ accumulates in PostProcess of this frame using GetLastDeltaTime
    if running:
        acc += float(np.float32(T["d"])); acc32 = np.float32(acc32 + np.float32(T["d"]))
    sr = T["sr"]
    if acc > 0:
        e = abs(np.float32(acc) - sr); worst = max(worst, e); n += 1
        if first is None and e > 0: first = (i, acc, sr)
    for a in fr["A"]:
        if a.startswith("Energy ") or a.startswith("IngameParameter ") or a.startswith("CurrentLevel 0 4") or a.startswith("CurrentLevel 0 3"):
            energy.append((i, a))
print("frames compared", n, "max |float(acc_double) - GetSRScore()|", worst, "first mismatch", first)
print("final acc ms", acc, "trunc", int(acc), "float32 acc", float(acc32))
print("events with pause:", [(i, e) for i, es in sorted(evs.items()) for e in es if e in ("PauseLevel","UnpauseLevel","CounterActive","CounterInactive","LevelFinish")][:20])
for e in energy[:12]: print(e)
print("...", len(energy))
print([e for e in energy if not e[1].startswith("Energy 0 0")][:30])

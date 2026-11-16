import pickle
res = pickle.load(open("tl.pkl","rb"))
for name, r in res.items():
    m = r["meta"]; lf = m["levelfinish_frame"]
    fd, pts = r["fd"], r["pts"]
    n_main = n_small = 0; odd = []
    for i in range(1, len(fd)):
        a, b = pts[i-1], pts[i]
        if a is None or b is None or b <= a: continue
        dp = b - a
        if fd[i] < 200: continue  # initial 1000
        for tick in (0, 1, 2):
            x = dp + tick
            if x % 20 == 0:
                k = x // 20
                if k >= 5 and (k - 5) % 1 == 0 and x >= 100 and (x - 100) % 20 == 0 and k not in (6,):
                    pass
                break
        else:
            odd.append((fd[i], dp)); continue
        # ambiguity: 120 could be main+1small or 6 smalls. decide via events count on that frame
        nev = sum(1 for f, n, p in r["evs"] if n == "ExtraPoint" and f == fd[i] + 1)
        units = x // 20
        if units % 5 == 0 and units >= 5 and (fd[i] not in [] ):
            pass
        res_name = (fd[i], x)
        r.setdefault("jumps", []).append((fd[i], x))
    # classify: main ball jumps are exactly 100 (ExtraPoint event next frame); everything else small
    js = r.get("jumps", [])
    mains = [f for f, x in js if x == 100]
    smalls = sum(x // 20 for f, x in js if x != 100)
    pre = [f for f in mains if lf is None or f <= lf]
    gap = None if m["hs"] is None else int(name.split("_")[1]) - m["hs"]
    print(f"{name:22s} main={len(mains):2d} small={smalls:3d} expect={6*len(mains):3d} missing={6*len(mains)-smalls:3d} missing*20={20*(6*len(mains)-smalls):5d} gap={gap} odd={odd}")

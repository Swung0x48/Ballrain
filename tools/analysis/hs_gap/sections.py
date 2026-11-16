import pickle
res = pickle.load(open("tl.pkl","rb"))
for name, r in res.items():
    m = r["meta"]; lf = m["levelfinish_frame"]
    fd, pts = r["fd"], r["pts"]
    js = []
    for i in range(1, len(fd)):
        a, b = pts[i-1], pts[i]
        if a is None or b is None or b <= a or fd[i] < 200: continue
        dp = b - a
        x = next(dp + t for t in (0,1,2) if (dp + t) % 20 == 0)
        js.append((fd[i], x))
    evs = r["evs"]
    pause = [f for f, n, p in evs if n == "PauseLevel"]
    unpause = [f for f, n, p in evs if n == "UnpauseLevel"]
    cps = [f for f, n, p in evs if n == "PreCheckpoint"]
    balloff = [f for f, n, p in evs if n == "BallOff"]
    ends = cps + ([lf] if lf else []) + ([m["executed_frames"]] if not lf else [])
    print("="*20, name, "gap", None if m["hs"] is None else int(name.split("_")[1]) - m["hs"], "BallOff", balloff)
    # Esc edges -> pause/unpause
    esc = r["esc_down"]
    ex = m["executed_frames"]
    lines = []
    for e in esc:
        p = [f for f in pause if e <= f <= e + 3]; u = [f for f in unpause if e <= f <= e + 3]
        lines.append(f"{e}:{'P'+str(p[0]-e) if p else ('U'+str(u[0]-e) if u else ('NOEFFECT' if e < ex else 'NOT_EXECUTED'))}")
    print(" esc edges:", " ".join(lines))
    orphan_p = [f for f in pause if not any(e <= f <= e+3 for e in esc)]
    print(" pauses w/o esc:", orphan_p)
    outstanding = 0; start = 0
    for k, end in enumerate(ends):
        mains = [f for f, x in js if start <= f < end and x == 100]
        smalls = sum(x//20 for f, x in js if start <= f < end and x != 100)
        inpause = [(p, u) for p, u in zip(pause, unpause + [10**9]) if start <= p < end]
        outstanding += 6*len(mains) - smalls
        label = "CP@%d" % end if end in cps else ("LF@%d" % end if end == lf else "END@%d" % end)
        print(f"  sec{k+1} [{start},{end}) {label}: mains={len(mains)} at {mains} smalls+={smalls} pauses={inpause} outstanding_after={outstanding}")
        # in this game: do lost balls reset outstanding at checkpoint? keep raw count
        start = end
    tail = [(f, x) for f, x in js if f >= start]
    print("  after last end:", tail)

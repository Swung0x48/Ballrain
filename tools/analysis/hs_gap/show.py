import pickle, sys, json
res = pickle.load(open("tl.pkl","rb"))
NAMEGAP = {}
for name, r in res.items():
    m = r["meta"]; lf = m["levelfinish_frame"]
    fd, pts = r["fd"], r["pts"]
    # point jumps
    jumps = []
    for i in range(1, len(fd)):
        a, b = pts[i-1], pts[i]
        if a is None or b is None: continue
        dp = b - a
        if dp > 0: jumps.append((fd[i], dp))
    evs = [(f, n) for f, n, p in r["evs"] if n in ("ExtraPoint","PauseLevel","UnpauseLevel","PreCheckpoint","PostCheckpoint","LevelFinish","PreLifeUp","PostLifeUp","BallOff","GameOver")]
    items = [(f, "EV " + n) for f, n in evs] + [(f, f"+{d}") for f, d in jumps] + [(f, "ESC_DOWN(input frame idx)") for f in r["esc_down"]]
    items.sort(key=lambda x: (x[0], x[1]))
    name_score = int(name.split("_")[1])
    hs = m["hs"]
    print("=" * 30, name, "nframes", r["nframes"], "LF", lf, "exec", m["executed_frames"], "hs", hs, "gap", None if hs is None else name_score - hs)
    if len(sys.argv) > 1 and sys.argv[1] not in name: continue
    for f, s in items: print(f, s)

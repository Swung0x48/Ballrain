"""Build complete paired references from logged engine centres, not hand-drawn x/z chords."""

import json, shutil, sys, os
from pathlib import Path
import numpy as np
from probe_l2_multi import OUT, REC
from ballrain.path import ReferencePath, file_sha256

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "benchmarks/refpaths-alt"
DEST = BASE / "L02-multi"
MAIN = ROOT / "benchmarks/refpaths-v2/L02.npz"


def dump(p, x):
    p.write_text(json.dumps(x, indent=2) + "\n", encoding="utf8", newline="\n")


def build(section, trace, entry):
    main = ReferencePath.load(MAIN)
    child = DEST / entry
    child.mkdir(parents=True, exist_ok=True)
    z = np.load(trace / "trace.npz")
    a = z["samples"]
    bt = 2 if section == 2 else 1
    ix = np.flatnonzero(a[:, 4] == bt)[0]
    a = a[ix:]
    p = a[:, 1:4]
    t = a[:, 7] - a[0, 7]
    measured = len(p)
    assert a[-1, 6] == section + 1, "refuse partial run"
    switch = np.flatnonzero(a[:, 4] != bt)[0]
    if section == 2:
        end = np.flatnonzero((a[:, 4] == 0) & (p[:, 2] < -328))[0]
        cuts = [np.flatnonzero(a[:, 5] == stage)[0] for stage in (2, 3, 4)] + [switch]
    else:
        end = np.flatnonzero((a[:, 5] >= 17) & (p[:, 2] > -319))[0]
        unlock = np.flatnonzero(a[:, 5] == 11)
        cuts = [
            switch,
            unlock[np.argmin(p[unlock, 0])],
            np.flatnonzero(a[:, 5] == 12)[0],
        ]
    p = p[: end + 1]
    t = t[: end + 1]
    measured = len(p)
    i, j = [np.argmin(np.linalg.norm(main.P - q, axis=1)) for q in (p[0], p[-1])]
    j = int(np.searchsorted(main.cum, main.cum[j] + 2.0))
    assert i < j
    ids = [0]
    for n in range(1, len(p)):
        if n in cuts or np.linalg.norm(p[n] - p[ids[-1]]) > 0.48:
            ids.append(n)
    points = np.vstack([main.P[: i + 1], p[ids], main.P[j:]])
    delta = max(1.0, np.linalg.norm(p[0] - main.P[i]) * 100)
    time = t[ids] + main.t_ms[i] + delta
    time = np.r_[
        main.t_ms[: i + 1],
        time,
        main.t_ms[j:]
        - main.t_ms[j]
        + time[-1]
        + max(1.0, np.linalg.norm(main.P[j] - p[ids[-1]]) * 100),
    ]
    path = ReferencePath(
        points,
        np.r_[
            main.point_sector[: i + 1],
            np.full(len(ids), section),
            main.point_sector[j:],
        ],
        meta=dict(main.meta, generator="scripts/build_l2_multi.py"),
        t_ms=time,
    )
    asset = child / "L02.npz"
    path.save(asset)
    path = ReferencePath.load(asset)
    si = i + 1 + ids.index(switch)
    ej = i + len(ids) + 1
    lo, mid, hi = path.cum[[i, si, ej]].tolist()
    ms = main.cum[[i, j]].tolist()
    mapped = np.interp(mid, [lo, hi], ms)
    desc = dict(
        level=2,
        ball_type=bt,
        sector=section,
        main_sha256=file_sha256(MAIN),
        branch_sha256=file_sha256(asset),
        branch_s=[lo, hi],
        main_s=ms,
        stations=[[mapped, mid]],
        rules={},
    )
    dump(child / "L02.json", desc)
    indices = [n for n in ids if n < measured]
    samples = np.column_stack([a[indices, 0], path.cum[i + 1 : i + 1 + len(indices)]])
    np.savez_compressed(
        child / "drive.npz",
        frames=z["frames"][: int(samples[-1, 0])],
        prefix_frame=z["prefix_frame"],
        samples=samples,
        record_sha256=json.loads((REC / "meta.json").read_text())["record_sha256"],
        branch_sha256=file_sha256(asset),
    )
    # Boxes enclose transform animation and the observed pit, but exclude the far side of the plate.
    box = (
        [[686, 29, -334], [721, 57, -260]]
        if section == 2
        else [[909, 45, -353], [925, 53, -340]]
    )
    gate = lambda b, box: dict(ball=b, sector=section, box=box)
    arcs = [lo, *[float(path.cum[i + 1 + ids.index(n)]) for n in cuts], hi]
    phases = []
    for q, (l, r) in enumerate(zip(arcs, arcs[1:])):
        n = cuts[q - 1] if q else 0
        center = p[n]
        tol = np.array([2.5, 3, 2.5])
        region = [(center - tol).tolist(), (center + tol).tolist()] if q else box
        when = gate(int(a[n, 4]), region)
        if q and n != switch:
            when["s"] = [l - 0.65, l + 0.001]
        phases.append(dict(local_s=[l, r], when=when))
    if section == 2:
        # A stone that landed west of the rails can reach wood without our pit run-up.
        phases[-1]["from"] = list(range(len(phases) - 1))
    merge = main.P[j]
    g = gate(
        int(a[end, 4]), [(merge - [8, 6, 5]).tolist(), (merge + [8, 6, 5]).tolist()]
    )
    g["s"] = [hi - 0.05, hi + 1]
    print(
        entry,
        "prefix",
        z["prefix_frame"],
        "phase arcs",
        arcs,
        "merge",
        merge,
        "main span",
        ms,
    )
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    ax.plot(main.P[i : j + 1, 0], main.P[i : j + 1, 2], "--", label="主线")
    ax.plot(
        p[:switch, 0], p[:switch, 2], label="石球回木" if section == 2 else "纸球去程"
    )
    ax.plot(
        p[switch:measured, 0],
        p[switch:measured, 2],
        label="木球再上坡" if section == 2 else "石球回程",
    )
    ax.axis("equal")
    ax.set(
        xlabel="东西位置 x", ylabel="南北位置 z", title=f"小节 {section} 分支球心轨迹"
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.savefig(OUT / f"line{section}.png")
    plt.close(fig)
    return dict(id=entry, phases=phases, merge=g)


if __name__ == "__main__":
    os.environ["MPLCONFIGDIR"] = str(OUT / "mplcache")
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {"font.family": "Microsoft YaHei", "axes.unicode_minus": False}
    )
    entries = [
        build(s, Path(p), name)
        for s, p, name in zip(
            (2, 3), sys.argv[1:], ("back-to-wood", "east-platform"), strict=True
        )
    ]
    shutil.copytree(BASE / "L02-stone", DEST / "south-loop", dirs_exist_ok=True)
    dump(
        DEST / "branches.json",
        dict(version=2, branches=[*entries, dict(id="south-loop", legacy=True)]),
    )

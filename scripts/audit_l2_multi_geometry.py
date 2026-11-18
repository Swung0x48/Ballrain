"""Audit measured contact/rotation, fan height, pit stop and both installed splices."""

from build_l2_multi import MAIN, DEST, OUT
import json, sys, os
import numpy as np
from ballrain.path import ReferencePath
from scripts.build_l2_stone import geom


def audit(run):
    trace = OUT / run
    z = np.load(trace / "trace.npz")
    a = z["samples"]
    s = int(a[0, 6])
    _, T = geom()
    xyz = a[:, 1:4]
    h, face = T.ray_down(xyz)
    supported = (face >= 0) & (abs(xyz[:, 1] - h - 2) < 0.15)
    print(
        run,
        "static centre samples within vertical tolerance",
        sum(supported),
        "/",
        len(a),
    )
    if s == 2:
        idx = np.flatnonzero(a[:, 5] == 2)[0] - 1
        r = a[idx]
        print(
            "release from pit",
            r[1:4],
            "distance from study stall",
            np.linalg.norm(r[1:4] - [715.98, 32.26, -267.86]),
        )
        return
    d = json.loads((trace / "mechanisms.json").read_text())
    plate = [r for r in d["poses"] if r[1] == "P_Modul_25_Bridge"]
    c = [r for r in d["contacts"] if r[1] == "P_Modul_25_Bridge"]
    angle = np.degrees(
        2 * np.arccos(np.clip(abs(np.dot(plate[0][-4:], plate[-1][-4:])), 0, 1))
    )
    print(
        "bridge first+last sampled friction",
        c[0],
        c[-1],
        "material",
        set(r[-1] for r in c),
        "angle from start deg",
        angle,
    )
    print("paper max height", xyz[a[:, 4] == 1, 1].max())
    print(
        "pusher",
        [(r[0], r[-1]) for r in d["contacts"] if r[1] == "P_Modul_01_Pusher"][:4],
    )


def profiles():
    os.environ["MPLCONFIGDIR"] = str(OUT / "mplcache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False})
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), layout="constrained")
    entries = json.loads((DEST / "branches.json").read_text())["branches"]
    names = [
        ["下坡入坑", "向北蓄速", "回冲上坡", "绕栏回木", "木球再爬坡"],
        ["纸球过双风扇", "石球顶横闸", "折返向东", "南廊回推板"],
    ]
    for ax, e, labels in zip(axes, entries, names):
        path = ReferencePath.load(DEST / e["id"] / "L02.npz")
        offset = e["phases"][0]["local_s"][0]
        for ph, label in zip(e["phases"], labels):
            lo, hi = ph["local_s"]
            mask = (path.cum >= lo) & (path.cum <= hi)
            ax.plot(path.cum[mask] - offset, path.P[mask, 1], label=label)
        ax.set(
            xlabel="距分岔的实际路程",
            ylabel="球心高度",
            title=labels[0] + "至" + labels[-1],
        )
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.savefig(OUT / "profiles.png")
    plt.close(fig)


if __name__ == "__main__":
    for run in sys.argv[1:]:
        audit(run)
    profiles()

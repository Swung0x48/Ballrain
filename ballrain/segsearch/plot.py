"""Top view (whole level + stretch close-up, on the level's real floor / rail outlines) and side view (height along the
TAS line) of a segment-search result: the TAS line and the verified found lines."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
GEOMETRY = REPO / "runtime/dashboard/geometry_l1.json.gz"  # the dashboard's outlines from the TAS record's level geometry


def progress(path, ref) -> np.ndarray:
    """Arc length along `ref` of each point's nearest ref point (top view), made non-decreasing."""
    ref = np.asarray(ref, float)
    cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(ref, axis=0), axis=1))]
    p = np.asarray(path, float)
    out = np.empty(len(p))
    for i in range(0, len(p), 2048):
        d = np.linalg.norm(p[i:i + 2048, None, [0, 2]] - ref[None, :, [0, 2]], axis=2)
        out[i:i + 2048] = cum[np.argmin(d, axis=1)]
    return np.maximum.accumulate(out)


def plot_result(result: Path, title: str = "", out: Path | None = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    res = json.loads(Path(result).read_text(encoding="utf-8"))
    tp = np.asarray(res["tas"]["pos_path"], float)
    found = [r for r in res["results"] if r.get("deterministic") and r.get("pos") and r.get("kind") != "fails_end_check"]
    other = [r for r in res["results"] if r.get("deterministic") and r.get("pos") and r.get("kind") == "fails_end_check"]
    geo = json.loads(gzip.decompress(GEOMETRY.read_bytes())) if GEOMETRY.is_file() else {}
    fig = plt.figure(figsize=(17, 13))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.8, 1.6], height_ratios=[1.2, 0.7])
    ax_all, ax, sx = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    allp = np.concatenate([tp] + [np.asarray(r["pos"], float) for r in found + other])
    x0, x1 = allp[:, 0].min() - 15, allp[:, 0].max() + 15
    z0, z1 = allp[:, 2].min() - 15, allp[:, 2].max() + 15
    for a, box in ((ax_all, None), (ax, (x0, x1, z0, z1))):
        for k, c in (("walk", "#ddd3b8"), ("steep", "#8d99ad"), ("rail", "#c9a36a")):
            for r in geo.get(k, []):
                pts = np.asarray(r, float).reshape(-1, 2)
                if box and (pts[:, 0].max() < box[0] or pts[:, 0].min() > box[1] or pts[:, 1].max() < box[2]
                            or pts[:, 1].min() > box[3]):
                    continue
                a.add_patch(Polygon(pts, closed=True, fc=c, ec="#9a8c6c", lw=0.3, alpha=0.85))
        a.plot(tp[:, 0], tp[:, 2], color="#1f3fff", lw=2.5, zorder=6, label="TAS")
        for i, r in enumerate(found):
            q = np.asarray(r["pos"], float)
            a.plot(q[:, 0], q[:, 2], color=("#e8333a", "#ff9f1a", "#18b84a")[i % 3], lw=1.6, zorder=7,
                   label=f"找到的线 {i + 1}：{r['delta_ms'] / 1000:+.3f} s")
        for i, r in enumerate(other):
            q = np.asarray(r["pos"], float)
            a.plot(q[:, 0], q[:, 2], color="#7a2bd1", lw=1.3, ls="--", zorder=7,
                   label=f"更早但终点不合格 {i + 1}：{r['delta_ms'] / 1000:+.3f} s（{'、'.join(r['check']['fails'])}）")
        a.set_aspect("equal")
        a.invert_yaxis()
    if geo:
        g = np.concatenate([np.asarray(r, float).reshape(-1, 2) for k in geo for r in geo[k]])
        ax_all.set_xlim(g[:, 0].min() - 10, g[:, 0].max() + 10)
        ax_all.set_ylim(g[:, 1].max() + 10, g[:, 1].min() - 10)
    ax_all.add_patch(plt.Rectangle((x0, z0), x1 - x0, z1 - z0, fill=False, ec="k", lw=1.2, zorder=9))
    ax_all.set_title("全关俯视（方框 = 这一段）")
    ax.set_xlim(x0, x1)
    ax.set_ylim(z1, z0)
    for g in res["gates"]:
        p = g["pos"]
        ax.plot(p[0], p[2], "o", color="#1f3fff", ms=4, zorder=8)
    ax.plot(tp[0, 0], tp[0, 2], "s", color="k", ms=8, zorder=9)
    ax.annotate("A", tp[0, [0, 2]], xytext=(6, 6), textcoords="offset points", weight="bold")
    ax.annotate("B", np.asarray(res["gates"][-1]["pos"])[[0, 2]], xytext=(6, 6), textcoords="offset points", weight="bold")
    ax.legend(loc="best", fontsize=11)
    ax.set_title("这一段俯视（蓝点 = 子门）")
    sx.plot(progress(tp, tp), tp[:, 1], color="#1f3fff", lw=2.5, label="TAS")
    for i, r in enumerate(found):
        q = np.asarray(r["pos"], float)
        sx.plot(progress(q, tp), q[:, 1], color=("#e8333a", "#ff9f1a", "#18b84a")[i % 3], lw=1.4)
    for r in other:
        q = np.asarray(r["pos"], float)
        sx.plot(progress(q, tp), q[:, 1], color="#7a2bd1", lw=1.2, ls="--")
    sx.set_xlabel("沿 TAS 线的进度")
    sx.set_ylabel("高度 y")
    sx.grid(alpha=0.3)
    sx.set_title("侧视：沿 TAS 线的高度")
    st = res["stats"]
    best = min((r["delta_ms"] for r in found), default=None)
    head = title or f"TAS 帧 {res['a']} → {res['b']}"
    fig.suptitle(f"{head}\nTAS A→B {res['tas']['sr_ms'] / 1000:.3f} s；最好 "
                 f"{'%+.3f s' % (best / 1000) if best is not None else '没有更快的'}；{st['rollouts']} 次 rollout，"
                 f"{st['engine_s']:.0f} 引擎秒（headless，球序 {res['engine']['ball_order']}）", fontsize=13)
    fig.tight_layout()
    out = out or Path(result).with_name("view.png")
    fig.savefig(out, dpi=90)
    plt.close(fig)
    return out

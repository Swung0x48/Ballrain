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


CATEGORIES = (  # (key, label, colour, draw order)
    ("behind", "落后被剪掉", "#9a9a9a"),
    ("stuck", "卡住（太久没过门）", "#ff9f1a"),
    ("fell", "掉下去", "#e8333a"),
    ("pickup", "少吃分数球 / 生命球被丢掉", "#7a2bd1"),
    ("reached", "到了 B", "#18b84a"),
)


def category(end: str, dropped: int) -> str:
    if dropped >= 0:
        return "pickup"
    if end in ("fell", "ball_off", "game_over"):
        return "fell"
    if end == "last_gate":
        return "reached"
    if end == "stuck":
        return "stuck"
    return "behind"


def plot_rollouts(out_dir: Path, title: str = "", out: Path | None = None) -> Path:
    """Every rollout a search tried (rollouts.npz: each rollout's own part, after the branch it started from), coloured
    by how it ended; top views per category on the level outlines + one side view (height along the TAS line)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Polygon
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    out_dir = Path(out_dir)
    res = json.loads((out_dir / "result.json").read_text(encoding="utf-8"))
    z = np.load(out_dir / "rollouts.npz")
    tp = np.asarray(res["tas"]["pos_path"], float)
    geo = json.loads(gzip.decompress(GEOMETRY.read_bytes())) if GEOMETRY.is_file() else {}
    off, pos = z["off"], z["pos"].astype(float)
    cats = [category(str(e), int(d)) for e, d in zip(z["end"], z["dropped"])]
    segs = {k: [] for k, _, _ in CATEGORIES}
    for i, c in enumerate(cats):
        q = pos[off[i]:off[i + 1]]
        if len(q) >= 2:
            segs[c].append(q)
    tt = z["t"]
    times = {k: [] for k in segs}
    for i, c in enumerate(cats):
        if off[i + 1] - off[i] >= 2:
            times[c].append(tt[off[i]:off[i + 1]])
    # top view: the TAS line's box with a margin (rollouts that fly far off are cut at the edge)
    x0, x1 = tp[:, 0].min() - 25, tp[:, 0].max() + 25
    z0, z1 = tp[:, 2].min() - 25, tp[:, 2].max() + 25
    shown = [c for c in CATEGORIES if segs[c[0]]]
    n_cat = len(shown)
    fig = plt.figure(figsize=(6.4 * n_cat, 12.5))
    gs = fig.add_gridspec(2, n_cat, height_ratios=[1.25, 0.8])
    for j, (k, label, col) in enumerate(shown):
        ax = fig.add_subplot(gs[0, j])
        for kk, c in (("walk", "#ddd3b8"), ("steep", "#8d99ad"), ("rail", "#c9a36a")):
            for r in geo.get(kk, []):
                pts = np.asarray(r, float).reshape(-1, 2)
                if pts[:, 0].max() < x0 or pts[:, 0].min() > x1 or pts[:, 1].max() < z0 or pts[:, 1].min() > z1:
                    continue
                ax.add_patch(Polygon(pts, closed=True, fc=c, ec="#9a8c6c", lw=0.3, alpha=0.85))
        if segs[k]:
            ax.add_collection(LineCollection([q[:, [0, 2]] for q in segs[k]], colors=col, lw=0.45, alpha=0.35, zorder=5))
            if k in ("fell", "stuck", "behind"):
                ends = np.array([q[-1] for q in segs[k]])
                ax.plot(ends[:, 0], ends[:, 2], "x", color=col if k != "behind" else "#555", ms=3, mew=0.8, alpha=0.6, zorder=6)
        ax.plot(tp[:, 0], tp[:, 2], color="#1f3fff", lw=1.6, zorder=7)
        ax.set_xlim(x0, x1)
        ax.set_ylim(z1, z0)
        ax.set_aspect("equal")
        ax.set_title(f"{label}：{len(segs[k])} 条", color=col if k != "behind" else "#444", fontsize=13)
    sx = fig.add_subplot(gs[1, :])
    for k, label, col in shown:
        sx.add_collection(LineCollection([np.c_[t_, q[:, 1]] for t_, q in zip(times[k], segs[k])], colors=col, lw=0.4,
                                         alpha=0.3, label=f"{label}（{len(segs[k])}）"))
    sx.plot(np.arange(len(tp)), tp[:, 1], color="#1f3fff", lw=2, label="TAS")
    sx.autoscale()
    sx.set_xlabel("A 之后的帧数（264 帧 = 1 秒）")
    sx.set_ylabel("高度 y")
    sx.grid(alpha=0.3)
    leg = sx.legend(loc="upper left", fontsize=11)
    for h in leg.legend_handles:
        h.set_alpha(1)
        h.set_linewidth(2)
    sx.set_title("高度随时间：全部 rollout（在 TAS 线下方 = 比 TAS 低；B 处 TAS 在右端）")
    st = res["stats"]
    kinds = {str(x): int((z["kind"] == x).sum()) for x in np.unique(z["kind"])}
    names = {"plain": "照 TAS", "edits": "局部编辑", "random_holds": "随机保持"}
    fig.suptitle(f"{title or ('TAS 帧 %d → %d' % (res['a'], res['b']))}：试过的全部 {len(cats)} 条 rollout（每条只画它自己新走的部分，"
                 f"起点 = 它接手的分支）；蓝 = TAS；× = 停下的位置\n按键来源："
                 + "，".join(f"{names.get(k_, k_)} {v}" for k_, v in kinds.items())
                 + f"；{st['engine_s']:.0f} 引擎秒（headless，球序 {res['engine']['ball_order']}）", fontsize=14)
    fig.tight_layout()
    out = out or out_dir / "rollouts.png"
    fig.savefig(out, dpi=80)
    plt.close(fig)
    return out

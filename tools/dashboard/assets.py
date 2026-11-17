"""Copies a few textures from a local Ballance install into runtime/dashboard/assets (untracked) at start.

Nothing from the game is committed: the page uses these files when they are present and falls back to
plain colours otherwise. Converted with Pillow (in uv.lock): sky, wood planks, beige floor stone, the
menu button frame and the menu bitmap font (Font_1.tga, sliced into an atlas + per-glyph widths).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAMES = ["sky.jpg", "planks.jpg", "stone2.jpg", "button.png", "button_sel.png", "font.png", "font.json", "ball.png"]
VERSION = "12"  # bump when the generated files change; prepare() rebuilds them


def make_planks(grain, np, Image, width: int = 4096, height: int = 96, seed: int = 7):
    """One long continuous board in the look of the game's single-board texture (Wood_Old), generated rather than
    tiled, so it never repeats and has no joints. The colours come from the texture itself: a brightness -> colour
    table and the brightness histogram of Wood_Old. The grain is value noise stretched far along the board at
    several scales, with a slow wave so the grain lines drift a little, as in real wood."""
    rng = np.random.default_rng(seed)
    src = np.asarray(grain.rotate(90, expand=True).convert("RGB"), np.float32)
    lum = src @ np.array([0.299, 0.587, 0.114], np.float32)
    # brightness -> colour table from the texture (average colour per brightness bin)
    nb = 64
    edges = np.quantile(lum, np.linspace(0, 1, nb + 1))
    idx = np.clip(np.searchsorted(edges, lum, side="right") - 1, 0, nb - 1)
    lut = np.stack([np.bincount(idx.ravel(), weights=src[..., c].ravel(), minlength=nb) /
                    np.maximum(np.bincount(idx.ravel(), minlength=nb), 1) for c in range(3)], -1)

    final_h, pad_h = height, 16
    height = final_h + 2 * pad_h  # generated taller and cropped: the drifting grain never runs off an edge
    final_w, wrap = width, 512
    width = final_w + wrap  # generated longer; the extra end is cross-faded into the start so the strip wraps seamlessly

    def noise(nx, ny):  # value noise: a coarse random grid resized up smoothly
        z = rng.random((max(2, ny), max(2, nx)), dtype=np.float32)
        return np.asarray(Image.fromarray(z, "F").resize((width, height), Image.BICUBIC), np.float32)

    yy = np.arange(height, dtype=np.float32)[:, None] * np.ones((1, width), np.float32)
    wave = (noise(width // 700, 2) - 0.5) * height * 0.35          # slow drift of the grain lines
    f = np.zeros((height, width), np.float32)
    for sx, sy, amp in ((900, 16, 1.0), (420, 7, 0.75), (180, 3, 0.55), (60, 1.6, 0.4), (25, 1.0, 0.38),
                        (9, 0.7, 0.34)):
        z = noise(max(2, int(width / sx)), max(2, int(height / sy)))
        # sample along the drifted rows so every octave follows the same grain lines
        rows = np.clip(np.round(yy + wave), 0, height - 1).astype(int)
        f += amp * np.take_along_axis(z, rows, axis=0)
    # short dark checks / cracks along the grain, as in the game texture (tapered at both ends)
    rows_all = np.clip(np.round(yy + wave), 0, height - 1).astype(int)
    lo = float(np.percentile(f, 2))
    for _ in range(width // 45):
        x0 = int(rng.integers(0, width)); n = int(rng.integers(14, 80)); y0 = int(rng.integers(2, height - 2))
        xs = np.arange(x0, min(width, x0 + n))
        if len(xs) < 4:
            continue
        taper = np.sin(np.linspace(0, np.pi, len(xs))) ** 0.6
        ys = np.clip(y0 + (rows_all[y0, xs] - y0) * 0 + np.round(np.cumsum(rng.normal(0, 0.12, len(xs)))).astype(int),
                     0, height - 1)
        depth = rng.uniform(0.8, 1.2)
        f[ys, xs] = f[ys, xs] * (1 - depth * taper) + lo * depth * taper
        f[np.clip(ys + 1, 0, height - 1), xs] += 0.08 * taper  # lit lower lip of the crack
    # seamless wrap: column final_w continues column final_w - 1, so fade the start from it into the original start
    ramp = np.linspace(0.0, 1.0, wrap, dtype=np.float32)[None, :]
    f[:, :wrap] = f[:, final_w:final_w + wrap] * (1 - ramp) + f[:, :wrap] * ramp
    f = f[pad_h:pad_h + final_h, :final_w]
    # match the brightness histogram of the original texture, then colour through the table
    order = np.argsort(f, axis=None)
    q = np.empty(f.size, np.float32)
    q[order] = np.sort(lum.ravel())[np.linspace(0, lum.size - 1, f.size).astype(int)]
    q = q.reshape(f.shape)
    b = np.clip(np.searchsorted(edges, q, side="right") - 1, 0, nb - 1)
    t = (q - edges[b]) / np.maximum(edges[b + 1] - edges[b], 1e-6)
    rgb = lut[b] + (lut[np.minimum(b + 1, nb - 1)] - lut[b]) * t[..., None]
    rgb = rgb * np.array([1.0, 0.86, 0.66], np.float32) * 0.97  # the warm tone of the game's planks
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def _textures_dir() -> Path | None:
    rt = ROOT / "runtime"
    cands = [rt / "train-00" / "Textures"] + sorted(rt.glob("train-*/Textures")) + [rt / "ref" / "Textures"]
    for c in cands:
        if (c / "Font_1.tga").exists():
            return c
    return None


def prepare(out: Path, log=print) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    ver = out / "version.txt"
    if all((out / n).exists() for n in NAMES) and ver.exists() and ver.read_text().strip() == VERSION:
        return NAMES
    tex = _textures_dir()
    if tex is None:
        log("assets: no local game textures found; the page uses plain colours")
        return [n for n in NAMES if (out / n).exists()]
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return []

    def rgb(name):
        return Image.open(tex / name).convert("RGB")

    try:
        sky = Image.open(tex / "Sky" / "Sky_K_Front.bmp").convert("RGB")
        sky.resize((512, 512), Image.BICUBIC).save(out / "sky.jpg", quality=85)
        make_planks(rgb("Wood_Old.bmp"), np, Image).save(out / "planks.jpg", quality=88)
        # seamless beige floor stone (no border strips), 2x so the grain is not tiny; tiles get different offsets
        rgb("Floor_Top_Borderless.bmp").resize((256, 256), Image.BICUBIC).save(out / "stone2.jpg", quality=88)
        for src, dst in (("Button01_deselect.tga", "button.png"), ("Button01_select.tga", "button_sel.png")):
            Image.open(tex / src).convert("RGBA").crop((3, 2, 253, 62)).save(out / dst)
        ball = Image.open(tex / "Ball_Wood.bmp").convert("RGBA").resize((64, 64))
        a = np.array(ball)
        yy, xx = np.mgrid[:64, :64]
        a[..., 3] = np.where((xx - 31.5) ** 2 + (yy - 31.5) ** 2 <= 31 ** 2, 255, 0)
        Image.fromarray(a).save(out / "ball.png")
        # menu font: 16 x 16 grid of 32 px cells, codes 32..255; glyph advance from the ink extent
        f = Image.open(tex / "Font_1.tga").convert("RGBA")
        A = np.array(f)[..., 3]
        glyphs = {}
        for code in range(32, 256):
            r, c = divmod(code, 16)
            cell = A[r * 32:(r + 1) * 32, c * 32:(c + 1) * 32]
            cols = np.flatnonzero(cell.max(0) > 24)
            if code == 32:
                glyphs[code] = [c * 32, 0, 10]
            elif len(cols):
                x0, x1 = int(cols[0]), int(cols[-1])
                glyphs[code] = [c * 32 + x0, r * 32, x1 - x0 + 1]
        rows = np.flatnonzero(A[64:96].max(1) > 24)
        f.save(out / "font.png")
        (out / "font.json").write_text(json.dumps({"cell": 32, "glyphs": glyphs, "top": int(rows[0]) if len(rows) else 0,
                                                    "bottom": int(rows[-1]) if len(rows) else 31}), encoding="utf-8")
        ver.write_text(VERSION)
        log("assets: prepared from the local game textures")
    except Exception as e:  # never fatal: the page falls back to plain colours
        log(f"assets: skipped ({type(e).__name__})")
    return [n for n in NAMES if (out / n).exists()]

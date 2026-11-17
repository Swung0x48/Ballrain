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
VERSION = "3"  # bump when the generated files change; prepare() rebuilds them


def make_planks(grain, np, Image, width: int = 4096, height: int = 96, seed: int = 7):
    """A long strip of boards instead of a repeated 128 px tile: the single-board grain texture (Wood_Old) is
    turned so the grain runs along the strip and stretched 7x along it (no visible period inside a board),
    then boards of random length are laid end to end, each with its own offset, flip and tone, with dark
    butt joints between them."""
    rng = np.random.default_rng(seed)
    g = grain.rotate(90, expand=True)
    g = g.resize((g.width * 7 * height // g.height, height), Image.BICUBIC)
    G = np.asarray(g, np.float32)
    period = G.shape[1]
    out = np.zeros((height, width, 3), np.float32)
    x = 0
    while x < width:
        L = int(rng.integers(height * 4, height * 8))
        off = int(rng.integers(0, period))
        idx = (off + np.arange(L)) % period
        board = G[:, idx]
        if rng.random() < 0.5:
            board = board[::-1]
        tone = rng.uniform(0.82, 1.08)
        warm = np.array([1.0, 0.86, 0.66], np.float32) * tone
        board = board * warm
        # end grain: darker towards both ends, then a dark joint line
        ramp = np.clip(np.minimum(np.arange(L), np.arange(L)[::-1]) / 18.0, 0, 1)
        board = board * (0.72 + 0.28 * ramp)[None, :, None]
        board[:, :2] *= 0.35
        n = min(L, width - x)
        out[:, x:x + n] = board[:, :n]
        x += L
    # the strip wraps: soften the seam between the last and the first board
    out[:, -2:] *= 0.35
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


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

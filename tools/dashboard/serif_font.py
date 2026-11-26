"""The page's title font, served by the page server itself (2026-11-25: the Google Fonts request for Noto Serif SC took
10+ s for viewers who reach Google slowly). A subset of a locally installed Noto Serif SC (OFL), only the characters
the page can show in its title faces: the static page and script, the reports, ASCII and common punctuation; one WOFF
per weight (700, 900) instanced from the variable font. Rebuilt when that character set changes. Characters outside
the subset fall back to the CSS stack (an installed Noto Serif SC / Songti / SimSun).

    build(out_dir, sources) -> version string, or None when no source font is installed"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

WEIGHTS = (700, 900)
CANDIDATES = [os.environ.get("BALLRAIN_SERIF_FONT") or "",
              r"C:\Windows\Fonts\NotoSerifSC-VF.ttf",
              str(Path.home() / "AppData/Local/Microsoft/Windows/Fonts/NotoSerifSC-VF.ttf"),
              "/Library/Fonts/NotoSerifSC-VF.ttf", str(Path.home() / "Library/Fonts/NotoSerifSC-VF.ttf"),
              "/usr/share/fonts/opentype/noto/NotoSerifSC-VF.ttf"]
EXTRA = "".join(chr(c) for c in range(0x20, 0x7F)) + "，。、：；！？（）《》“”‘’—…·「」【】％＋－／～ "


def source_font() -> Path | None:
    for c in CANDIDATES:
        if c and Path(c).is_file():
            return Path(c)
    return None


def _strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _strings(k)
            yield from _strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _strings(v)


def page_chars(files) -> str:
    """Characters in the given text files (.json files are parsed, so escaped characters count too)."""
    chars = set(EXTRA)
    for f in files:
        p = Path(f)
        if not p.is_file():
            continue
        try:
            text = p.read_text(encoding="utf-8")
            chars.update("".join(_strings(json.loads(text))) if p.suffix == ".json" else text)
        except Exception:  # noqa: BLE001  an unreadable file only leaves its characters out
            continue
    return "".join(sorted(c for c in chars if c.isprintable()))


def version(out_dir: Path, sources) -> tuple[str | None, bool]:
    """(version of the subset for the current character set, whether out_dir already holds it); (None, False)
    without a source font. Cheap: no font is opened."""
    src = source_font()
    if src is None:
        return None, False
    st = src.stat()
    ver = hashlib.sha1(f"{page_chars(sources)}|{src.name}|{st.st_size}".encode()).hexdigest()[:12]
    stamp = out_dir / "serif.version"
    done = (stamp.is_file() and stamp.read_text(encoding="ascii").strip() == ver
            and all((out_dir / f"serif-{w}.woff").is_file() for w in WEIGHTS))
    return ver, done


def build(out_dir: Path, sources) -> str | None:
    """Writes out_dir/serif-<weight>.woff for the characters of `sources`; returns the version (hash of the character
    set and the source font), or None without a source font. Does nothing when that version is already built."""
    src = source_font()
    if src is None:
        return None
    chars = page_chars(sources)
    st = src.stat()
    ver = hashlib.sha1(f"{chars}|{src.name}|{st.st_size}".encode()).hexdigest()[:12]
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = out_dir / "serif.version"
    if stamp.is_file() and stamp.read_text(encoding="ascii").strip() == ver and \
            all((out_dir / f"serif-{w}.woff").is_file() for w in WEIGHTS):
        return ver
    from fontTools import subset
    from fontTools.ttLib import TTFont
    from fontTools.varLib import instancer
    for w in WEIGHTS:
        f = TTFont(src)
        if "fvar" in f:
            f = instancer.instantiateVariableFont(f, {"wght": w})
        opt = subset.Options()
        opt.flavor = "woff"
        opt.hinting = False
        opt.desubroutinize = True
        opt.name_IDs = ["*"]
        opt.layout_features = ["kern", "liga", "locl", "vert"]
        s = subset.Subsetter(opt)
        s.populate(text=chars)
        s.subset(f)
        tmp = out_dir / f"serif-{w}.woff.tmp"
        f.flavor = "woff"
        f.save(tmp)
        os.replace(tmp, out_dir / f"serif-{w}.woff")
    stamp.write_text(ver, encoding="ascii")
    return ver


if __name__ == "__main__":  # the page server runs the build in its own process (it takes ~40 s of pure Python)
    import sys
    print(build(Path(sys.argv[1]), sys.argv[2:]) or "")

"""Publish a training checkpoint under models/<name>/ (git LFS for .pt / .npz):

    uv run python tools/export_model.py CKPT NAME --run r7 --update 85025 [--eval DIR] [--title "..."] [--note "..."]

Writes:
  <name>.pt            sanitized PyTorch checkpoint: model, rms, args, n_obs, n_act, env_config (no optimizer state;
                       absolute local paths rewritten to repo-relative ones; refuses to write if any path, user name
                       or private address remains in the non-tensor fields)
  plain/<sha>.npz/.json  pickle-free plain-array export (ballrain/pool/weights.py); not for an f7 checkpoint (the reach
                       block: the pool format and the native client compute f6 only, docs/f7-native-port.md), which
                       gets the .pt and a README note instead
  refpaths/L<nn>.npz   the reference line the policy observes (features.path_source "tas"): copied from the
                       checkpoint's path_dir (or --refpath-dir), checked against its path_sha256, and the checkpoint's
                       path_dir rewritten to models/<name>/refpaths, so the published directory runs on its own. A line
                       whose metadata says it was derived from human play is refused without --allow-human-derived
  *.tas                the exported .tas of the evaluation in --eval (eval_export.json), if any
  README.md            title, note and the evaluation summary from --eval/eval.json, if given
Every check (private strings, the plain export, which runs into a temporary directory first) happens before the
first write: a refusal or error leaves models/<name>/ as it was.
TRUSTED INPUT ONLY: the checkpoint is unpickled.

Backfill a model published before the line was bundled (rewrites its .pt and adds refpaths/, nothing else):

    uv run python tools/export_model.py --add-refpath models/<name> [--refpath-dir DIR] [--allow-human-derived]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

PRIVATE = re.compile(r"[A-Za-z]:[\\/]|/Users/|/home/|\\\\|\b\d{1,3}(\.\d{1,3}){3}\b", re.I)
KEEP = ("model", "rms", "args", "n_obs", "n_act", "env_config", "branch_assets")
HUMAN = re.compile(r"\bhuman\b|人类|玩家", re.I)


def _relpath(s: str) -> str:
    """Absolute path inside some checkout of this repo -> the repo-relative part."""
    t = s.replace("\\", "/")
    for marker in ("/ballrain/repo/", "/repos/Ballrain/", "/Ballrain/", "/ballrain/"):
        i = t.find(marker)
        if i >= 0 and (t[:1] == "/" or (len(t) > 1 and t[1] == ":")):
            return t[i + len(marker):]
    return s


def _fix(o):
    if isinstance(o, dict):
        return {k: _fix(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_fix(v) for v in o]
    if isinstance(o, tuple):
        return tuple(_fix(v) for v in o)
    if isinstance(o, str):
        return _relpath(o)
    return o


def _strings(o):
    if isinstance(o, dict):
        for v in o.values():
            yield from _strings(v)
    elif isinstance(o, (list, tuple)):
        for v in o:
            yield from _strings(v)
    elif isinstance(o, str):
        yield o


def _private_names() -> list[str]:
    import os
    names = {os.environ.get(k, "") for k in ("USERNAME", "USER", "COMPUTERNAME")} | {Path.home().name}
    names |= {x.strip() for x in os.environ.get("BALLRAIN_PRIVATE_NAMES", "").split(",")}
    return [n.lower() for n in names if len(n) >= 3]


def _eval_summary(ev: Path) -> str:
    j = json.loads((ev / "eval.json").read_text(encoding="utf-8"))
    s = j.get("summary") or {}
    lines = [f"- 原版从出生点起评估（温度 {j.get('temperature')}）：通关 {s.get('finished')} / {s.get('sampled_runs')}，"
             f"无死亡 {s.get('finished_no_death')}；用时 最快 {s.get('sr_ms_min', 0) / 1000:.3f} s、"
             f"中位 {(s.get('sr_ms_median') or 0) / 1000:.3f} s、最慢 {(s.get('sr_ms_max') or 0) / 1000:.3f} s"]
    for g in s.get("greedy") or []:
        lines.append(f"- 确定性按键：{g.get('end_reason')}，{(g.get('sr_ms') or 0) / 1000:.3f} s，死亡 {g.get('deaths')}")
    return "\n".join(lines)


def refpath_source(env_config: dict, refpath_dir: str | None):
    """(source file, level, sha256) of the reference line the checkpoint observes, or None when it observes none
    (path_source other than "tas"). Raises SystemExit when the file is missing or is not the pinned one."""
    from ballrain.path import asset_path, resolve_path_dir
    fc = env_config.get("features") or {}
    if fc.get("path_source", "tas") != "tas":
        return None
    level = int(env_config.get("level") or 1)
    src = asset_path(level, Path(refpath_dir) if refpath_dir else resolve_path_dir(fc.get("path_dir")))
    if not src.is_file():
        sys.exit(f"refusing: the reference line {src} is missing (give its directory with --refpath-dir)")
    sha = hashlib.sha256(src.read_bytes()).hexdigest()
    if fc.get("path_sha256") and sha != fc["path_sha256"]:
        sys.exit(f"refusing: {src} has sha256 {sha[:12]}, the checkpoint pins {fc['path_sha256'][:12]}")
    return src, level, sha


def check_refpath(src: Path, allow_human: bool) -> None:
    """Refuses a line that cannot be published: private strings in its metadata, or derived from human play
    without --allow-human-derived."""
    import numpy as np
    z = np.load(src, allow_pickle=False)
    meta = str(z["meta"]) if "meta" in z.files else ""
    names = _private_names()
    if PRIVATE.search(meta) or any(n in meta.lower() for n in names):
        sys.exit(f"refusing: private-looking strings in the metadata of {src}")
    if HUMAN.search(meta) and not allow_human:
        sys.exit(f"refusing: {src} says it was derived from human play; publishing it needs --allow-human-derived")


def bundle_refpath(clean: dict, out: Path, name: str, ref) -> str | None:
    """Points the cleaned checkpoint at models/<name>/refpaths (and pins the sha256); returns the file name to copy."""
    if ref is None:
        return None
    src, level, sha = ref
    fc = dict(clean["env_config"]["features"])
    fc["path_dir"] = f"models/{name}/refpaths"
    fc["path_sha256"] = sha
    clean["env_config"] = dict(clean["env_config"], features=fc)
    return f"L{level:02d}.npz"


def add_refpath(model_dir: Path, refpath_dir: str | None, allow_human: bool) -> None:
    import torch
    pts = sorted(model_dir.glob("*.pt"))
    if len(pts) != 1:
        sys.exit(f"refusing: expected one .pt in {model_dir}, found {len(pts)}")
    ck = torch.load(pts[0], weights_only=False, map_location="cpu")
    ref = refpath_source(ck["env_config"], refpath_dir)
    if ref is None:
        sys.exit("the checkpoint observes no reference line; nothing to add")
    check_refpath(ref[0], allow_human)
    fname = bundle_refpath(ck, model_dir, model_dir.name, ref)
    (model_dir / "refpaths").mkdir(exist_ok=True)
    shutil.copy(ref[0], model_dir / "refpaths" / fname)
    torch.save(ck, pts[0])
    print(f"{model_dir}: refpaths/{fname} ({ref[2][:12]}), path_dir -> {ck['env_config']['features']['path_dir']}")


def is_f7(ck: dict) -> bool:
    fc = (ck.get("env_config") or {}).get("features") or {}
    return bool(fc.get("reach")) or fc.get("version") in ("f7", "f8")  # f8 (contact blocks): no plain/ export either


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?")
    ap.add_argument("name", nargs="?")
    ap.add_argument("--run")
    ap.add_argument("--update", type=int)
    ap.add_argument("--refpath-dir", default=None, help="directory holding the checkpoint's L<nn>.npz (default: its path_dir)")
    ap.add_argument("--allow-human-derived", action="store_true",
                    help="publish a reference line whose metadata says it was derived from human play")
    ap.add_argument("--add-refpath", metavar="MODEL_DIR", help="backfill refpaths/ into an already published model")
    ap.add_argument("--eval", default=None, help="evaluation directory (eval.json [, eval_export.json])")
    ap.add_argument("--title", default="")
    ap.add_argument("--note", action="append", default=[])
    a = ap.parse_args(argv)
    if a.add_refpath:
        return add_refpath(Path(a.add_refpath), a.refpath_dir, a.allow_human_derived)
    if not (a.ckpt and a.name and a.run and a.update is not None):
        ap.error("ckpt, name, --run and --update are required")

    import torch
    from ballrain.pool.weights import export_state

    out = REPO / "models" / a.name
    ck = torch.load(a.ckpt, weights_only=False, map_location="cpu")
    f7 = is_f7(ck)
    clean = {k: _fix(ck[k]) for k in KEEP if k in ck}
    clean["source"] = {"run": a.run, "update": a.update}
    ref = refpath_source(clean["env_config"], a.refpath_dir)
    if ref is not None:
        check_refpath(ref[0], a.allow_human_derived)
    ref_name = bundle_refpath(clean, out, a.name, ref)
    meta = {k: v for k, v in clean.items() if k not in ("model", "rms")}
    names = _private_names()
    public_addr = re.compile(r"\b(0\.0\.0\.0|127\.0\.0\.1)\b")  # bind-all / loopback listen addresses are not private
    bad = [s for s in _strings(meta)
           if PRIVATE.search(public_addr.sub("", s)) or any(n in s.lower() for n in names)]
    if bad:
        sys.exit(f"refusing: private-looking strings remain: {bad[:5]}")
    lines = [f"# {a.title or a.name}", "", f"来源：训练 {a.run}，第 {a.update} 次更新。", ""]
    lines += [f"{n}" for n in a.note] + ([""] if a.note else [])
    copies = []  # (source .tas, name in out)
    if a.eval:
        ev = Path(a.eval)
        lines += ["## 评估", "", _eval_summary(ev), ""]
        ex = ev / "eval_export.json"
        if ex.exists():
            e = json.loads(ex.read_text(encoding="utf-8"))
            for x in e.get("export") or []:
                sr = int(x["sr_ms"])
                src = ev / "export" / "records" / Path(str(x["tas"]).replace("\\", "/")).name
                if src.exists():
                    dst = f"L01_{sr // 1000}_{sr % 1000:03d}_{a.run}-u{a.update}.tas"
                    copies.append((src, dst))
                    lines.append(f"- `{dst}`：{sr / 1000:.3f} s，真实 TASSupport 重放 {x.get('ref_identical_to_agent')} 一致"
                                 f"（{x.get('verdict')}）")
            lines.append("")
    lines += ["## 文件", "",
              f"- `{a.name}.pt`：PyTorch 检查点（网络、观测归一化、训练参数、环境配置；无优化器状态，路径为仓库相对路径），"
              "可直接用于 `scripts/eval_m2.py --ckpt`。"]
    if ref_name:
        lines += [f"- `refpaths/{ref_name}`：模型观测里用到的参考线（sha256 `{ref[2][:12]}`），检查点的 path_dir 已指向这里，"
                  "所以这个目录本身就能运行。"]
    if f7:
        rv = ((ck.get("env_config") or {}).get("features") or {}).get("reach_version")
        lines += [f"- 没有 `plain/` 导出：这是 f7 检查点（观测含 fastreach reach 块，版本 `{rv}`），资源池格式和原生客户端只算 f6"
                  "（`docs/f7-native-port.md`）。", ""]
    else:
        lines += ["- `plain/<sha>.npz` + `.json`：不用 pickle 的纯数组导出（`ballrain/pool/weights.py`），文件名是策略的 sha256。",
                  ""]

    # everything that can refuse has run; the plain export goes to a temporary directory first
    with tempfile.TemporaryDirectory() as tmp:
        plain = None
        if not f7:
            try:
                export_state(ck, Path(tmp) / "plain", a.run)
            except ValueError as e:
                sys.exit(f"refusing: the plain export fails ({e}); nothing written")
            plain = Path(tmp) / "plain"
        out.mkdir(parents=True, exist_ok=True)
        torch.save(clean, out / f"{a.name}.pt")
        if plain is not None:
            (out / "plain").mkdir(exist_ok=True)
            for f in plain.iterdir():
                shutil.copy(f, out / "plain" / f.name)
    if ref_name:
        (out / "refpaths").mkdir(exist_ok=True)
        shutil.copy(ref[0], out / "refpaths" / ref_name)
    for src, dst in copies:
        shutil.copy(src, out / dst)
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")
    if f7:
        print("f7 checkpoint (reach block): no plain/ export (the pool format and the native client compute f6 only)")
    print("wrote", out)


if __name__ == "__main__":
    main()

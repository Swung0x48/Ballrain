"""Publish a training checkpoint under models/<name>/ (git LFS for .pt / .npz):

    uv run python tools/export_model.py CKPT NAME --run r7 --update 85025 [--eval DIR] [--title "..."] [--note "..."]

Writes:
  <name>.pt            sanitized PyTorch checkpoint: model, rms, args, n_obs, n_act, env_config (no optimizer state;
                       absolute local paths rewritten to repo-relative ones; refuses to write if any path, user name
                       or private address remains in the non-tensor fields)
  plain/<sha>.npz/.json  pickle-free plain-array export (ballrain/pool/weights.py)
  *.tas                the exported .tas of the evaluation in --eval (eval_export.json), if any
  README.md            title, note and the evaluation summary from --eval/eval.json, if given
TRUSTED INPUT ONLY: the checkpoint is unpickled.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

PRIVATE = re.compile(r"[A-Za-z]:[\\/]|/Users/|/home/|\\\\|\b\d{1,3}(\.\d{1,3}){3}\b", re.I)
KEEP = ("model", "rms", "args", "n_obs", "n_act", "env_config")


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("name")
    ap.add_argument("--run", required=True)
    ap.add_argument("--update", type=int, required=True)
    ap.add_argument("--eval", default=None, help="evaluation directory (eval.json [, eval_export.json])")
    ap.add_argument("--title", default="")
    ap.add_argument("--note", action="append", default=[])
    a = ap.parse_args()

    import torch
    from ballrain.pool.weights import export_state

    out = REPO / "models" / a.name
    out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(a.ckpt, weights_only=False, map_location="cpu")
    clean = {k: _fix(ck[k]) for k in KEEP if k in ck}
    clean["source"] = {"run": a.run, "update": a.update}
    meta = {k: v for k, v in clean.items() if k not in ("model", "rms")}
    names = _private_names()
    public_addr = re.compile(r"\b(0\.0\.0\.0|127\.0\.0\.1)\b")  # bind-all / loopback listen addresses are not private
    bad = [s for s in _strings(meta)
           if PRIVATE.search(public_addr.sub("", s)) or any(n in s.lower() for n in names)]
    if bad:
        sys.exit(f"refusing: private-looking strings remain: {bad[:5]}")
    torch.save(clean, out / f"{a.name}.pt")
    export_state(ck, out / "plain", a.run)

    lines = [f"# {a.title or a.name}", "", f"来源：训练 {a.run}，第 {a.update} 次更新。", ""]
    lines += [f"{n}" for n in a.note] + ([""] if a.note else [])
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
                    shutil.copy(src, out / dst)
                    lines.append(f"- `{dst}`：{sr / 1000:.3f} s，真实 TASSupport 重放 {x.get('ref_identical_to_agent')} 一致"
                                 f"（{x.get('verdict')}）")
            lines.append("")
    lines += ["## 文件", "",
              f"- `{a.name}.pt`：PyTorch 检查点（网络、观测归一化、训练参数、环境配置；无优化器状态，路径为仓库相对路径），"
              "可直接用于 `scripts/eval_m2.py --ckpt`。",
              "- `plain/<sha>.npz` + `.json`：不用 pickle 的纯数组导出（`ballrain/pool/weights.py`），文件名是策略的 sha256。", ""]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()

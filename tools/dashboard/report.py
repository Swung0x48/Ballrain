"""Short progress reports shown at the top of the dashboard ("最新进展").

    uv run python tools/dashboard/report.py add --title "标题" --item "要点 1" --item "要点 2" [--next "下一步"]
    uv run python tools/dashboard/report.py list

Reports live in runtime/dashboard/reports.json (untracked). The page is public, so every report is checked
for local details (paths, hosts, IPs, user names, process ids, copy / instance names, operational trivia)
and refused if any are found.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILE = ROOT / "runtime" / "dashboard" / "reports.json"
KEEP = 60
MAX_ITEMS = 8
MAX_LEN = 240

# Anything that looks like a local detail. Keep this strict: a false positive just means rewording.
BANNED = [
    (r"[A-Za-z]:[\\/]", "盘符路径"),
    (r"(?<![\w.])/(?:Users|home|mnt|tmp|var|etc|opt)/", "文件路径"),
    (r"\\\\|[\\/][\w.-]+\.(?:py|exe|dll|tas|npz|json|ckpt|pt|log|bat|sh)\b", "文件路径或文件名"),
    (r"\b\d{1,3}(?:\.\d{1,3}){3}\b", "IP 地址"),
    (r"\b(?:localhost|127\.0\.0\.1|0\.0\.0\.0)\b", "本机地址"),
    (r"\bssh\b|@[\w-]+\.(?:local|lan)\b|\.local\b", "主机 / SSH"),
    (r"\bpid\b|进程号", "进程号"),
    (r"\btrain-\d+|\btest-nowin|\bremote[_ -]?worker|\bbridge\b|\bsupervisor\b|watchdog|看门狗", "实例 / 运维细节"),
    (r"miner|挖矿|krig", "本机琐事"),
    (r"焦点|静音|弹窗|窗口|无头客户端|断线|重启了|worker", "本机运维细节"),
    (r"bmmo|公测|黑匣子|人类录像|human[- ]?demo", "人类录像数据"),
]


def _local_names() -> set[str]:
    """This machine's user / host names, read at run time so none are written into the repo."""
    names = {os.environ.get(k, "") for k in ("USERNAME", "USER", "LOGNAME", "COMPUTERNAME", "HOSTNAME")}
    names.add(Path.home().name)
    extra = os.environ.get("BALLRAIN_PRIVATE_NAMES", "")  # e.g. remote user names, comma-separated
    names.update(x.strip() for x in extra.split(","))
    return {n.lower() for n in names if len(n) >= 3}


def check(text: str) -> list[str]:
    bad = [why for pat, why in BANNED if re.search(pat, text, re.IGNORECASE)]
    low = text.lower()
    if any(n in low for n in _local_names()):
        bad.append("用户名 / 主机名")
    return bad


def tz_label() -> str:
    off = dt.datetime.now().astimezone().utcoffset()
    h = int(off.total_seconds() // 3600) if off else 0
    m = int(abs(off.total_seconds()) % 3600 // 60) if off else 0
    return f"UTC{h:+d}" + (f":{m:02d}" if m else "")


def _ts(at: str, tz: str) -> int | None:
    """Unix seconds of the report time, so the page can show it in the viewer's time zone."""
    m = re.fullmatch(r"UTC([+-])(\d{1,2})(?::(\d{2}))?", tz)
    try:
        t = dt.datetime.strptime(at, "%Y-%m-%d %H:%M")
    except ValueError:
        return None
    if not m:
        return None
    off = dt.timedelta(hours=int(m.group(2)), minutes=int(m.group(3) or 0)) * (1 if m.group(1) == "+" else -1)
    return int(t.replace(tzinfo=dt.timezone(off)).timestamp())


def load() -> list[dict]:
    try:
        data = json.loads(FILE.read_text(encoding="utf-8"))
        return [r for r in data if isinstance(r, dict)] if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def public(reports: list[dict], n: int = 12) -> list[dict]:
    """Newest first, only the fields the page shows, re-checked so a hand-edited file cannot leak."""
    out = []
    for r in reversed(reports[-n:]):
        title = str(r.get("title", ""))[:MAX_LEN]
        items = [str(x)[:MAX_LEN] for x in (r.get("items") or [])][:MAX_ITEMS]
        nxt = str(r.get("next", ""))[:MAX_LEN]
        if check(" ".join([title, nxt, *items])):
            continue
        at, tz = str(r.get("at", ""))[:16], str(r.get("tz", ""))[:10]
        out.append({"at": at, "tz": tz, "ts": _ts(at, tz), "title": title, "items": items, "next": nxt})
    return out


# A time inside report text is written as {{t:<epoch seconds>}} (or {{t:<ISO 8601 with offset>}}, converted to epoch
# seconds when the report is added); the page shows it in the reader's own time zone.
_T_TOKEN = re.compile(r"\{\{t:([^}]+)\}\}")


def _norm_times(text: str) -> str:
    def one(m):
        v = m.group(1).strip()
        if re.fullmatch(r"\d{9,11}", v):
            return "{{t:%s}}" % v
        try:
            d = dt.datetime.fromisoformat(v)
        except ValueError:
            sys.exit(f"时间标记无法解析：{v}（用 epoch 秒，或带时区的 ISO 8601，例如 2026-11-18T16:28+08:00）")
        if d.tzinfo is None:
            sys.exit(f"时间标记缺少时区：{v}")
        return "{{t:%d}}" % int(d.timestamp())
    return _T_TOKEN.sub(one, text)


def add(title: str, items: list[str], nxt: str) -> dict:
    title, nxt = _norm_times(title), _norm_times(nxt)
    items = [_norm_times(i.strip()) for i in items if i.strip()]
    if not title.strip() or not items:
        sys.exit("需要 --title 和至少一个 --item")
    if len(items) > MAX_ITEMS:
        sys.exit(f"最多 {MAX_ITEMS} 条要点")
    for t in [title, nxt, *items]:
        if len(t) > MAX_LEN:
            sys.exit(f"单条超过 {MAX_LEN} 字：{t[:40]}…")
    bad = check(" ".join([title, nxt, *items]))
    if bad:
        sys.exit("报告含有不能公开的内容：" + "、".join(sorted(set(bad))))
    rep = {"at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "tz": tz_label(),
           "title": title.strip(), "items": items, "next": nxt.strip()}
    reports = (load() + [rep])[-KEEP:]
    FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(reports, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, FILE)
    return rep


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("--title", required=True)
    a.add_argument("--item", action="append", default=[])
    a.add_argument("--next", default="")
    sub.add_parser("list")
    args = p.parse_args()
    if args.cmd == "add":
        print(json.dumps(add(args.title, args.item, args.next), ensure_ascii=False, indent=1))
    else:
        print(json.dumps(public(load()), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

"""Read-only access to the training outputs, turned into the dashboard's JSON payloads.

Runs: runtime/m2-train-mac/r<N> (trained on the Mac, pulled to this PC) and runtime/m2-train/r<N>
(older runs trained on the PC). Files that the pull job may be rewriting are read tolerantly (bad lines
skipped) and the caller keeps the last good result.

Privacy: the payloads may be published. They never carry file paths, host names, IPs, user names,
process ids, run ids or game-copy names, nor operational log lines; free text that does get through
(curriculum log lines, error messages) passes `sanitize`. No human-demo data is read.
"""
from __future__ import annotations

import datetime as dt
import glob
import io
import json
import math
import pickle
import os
import re
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MAC_DIR = ROOT / "runtime" / "m2-train-mac"
PC_DIR = ROOT / "runtime" / "m2-train"
DATASET = ROOT / "runtime" / "datasets" / "tas-v1"
REFPATHS = ROOT / "benchmarks" / "refpaths"
RECORDS = ROOT / "benchmarks" / "records.json"
# default start record per level: mirrors scripts/train_m2.py RECORDS (checked by tests/test_dashboard_levels.py)
LEVEL_RECORDS = {1: "SR01_1_14_457_264", 2: "SR02_1_28_992_264"}
# SR records SR<level>_<m>_<ss>_<mmm>[_<fps>], HS records HS<level>_<score>_<m>_<ss>_<mmm>: group 1 is the level
RECORD_RE = re.compile(r"^(?:SR(?=\d\d_\d+_\d\d_\d{3}_\d{3}$)|HS(?=\d\d_\d+_\d+_\d\d_\d{3}$))(\d\d)_")

SHORTCUT_EXTRA_S = 3.0  # level 1 (鬼躲石变) when a run does not record its split_extra
SR_WINDOW_UPDATES = 1500
SR_BIN_FRAMES = 2000
SR_MIN_N = 10
SR_MIN_SEG_S = 10.0  # bins whose remaining TAS segment is shorter are too noisy for the range
DEFAULT_MAC_OFFSET_H = 12.0  # Mac log clock minus PC clock, used when it cannot be measured
MAX_TRAJ_POINTS = 2000
MIN_STEP = 1.0  # world units between kept trajectory points

BACKEND_LABEL = {"headless": "x87-soft headless", "retail": "2002 retail", "headless2002": "2002 headless"}


BEST_NEAR_FRONTIER = 500  # TAS frames after the frontier still counted as "near" for the best-time tile


def machine_of(backend, host, sources: dict) -> str:
    """Generic machine name of an episode for the data-source tile. `sources` maps the generic host labels the
    workers send (--host-label) to machine names, plus "learner" (episodes from the learner's own workers, which
    carry no host) and "retail" (retail episodes whose host is the machine's own name, never shown)."""
    h = "" if host is None or (isinstance(host, float) and host != host) else str(host)
    if not h:
        return sources.get("learner", "其他")
    if h in sources:
        return sources[h]
    if str(backend) == "retail" and "retail" in sources:
        return sources["retail"]
    return "其他"

LOG_RE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) (.*)$")
UPD_RE = re.compile(r"u(\d+) steps (\d+) sps (\d+) frontier (\d+) band ([\d.]+)/(\d+).*? ent ([\d.]+) kl ([\d.]+)")
FRONTIER_RE = re.compile(r"curriculum: frontier (?:->|set \d+ ->) (\d+)")
CURR_OK_RE = re.compile(r"^curriculum: frontier (?:->|set \d+ ->) \d+(?: \(sector \d+\))?(?: after rate [\d.]+)?"
                        r"(?: \(--set-frontier\))?$")

_SAN = [
    (re.compile(r"(?:[A-Za-z]:)?(?:[\\/][^\s\\/,;:()'\"]+)+[\\/]?"), "<…>"),          # absolute paths
    (re.compile(r"\b[\w.-]+(?:[\\/][\w.-]+)+"), "<…>"),                                # relative paths
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b"), "<…>"),                       # IPv4
    (re.compile(r"\b[\w-]+\.(?:local|lan|home|internal)\b", re.I), "<…>"),              # host names
    (re.compile(r"\b(?:hl|tr|train|ref|copy)-\d+\b", re.I), "<…>"),                     # game copies
    (re.compile(r"\bL\d\d-[\w-]+-\d{8}-\d{6}[\w-]*"), "<…>"),                           # run ids
    (re.compile(r"\bpids?\s*[=:]?\s*\d+", re.I), "<…>"),
    (re.compile(r"\b(?:Users|home)\b", re.I), "<…>"),
]


def sanitize(s: str) -> str:
    s = str(s)
    for rx, rep in _SAN:
        s = rx.sub(rep, s)
    return s[:300]


# ------------------------------------------------------------------------------------ helpers

def _sig(p: Path):
    try:
        st = p.stat()
        return (st.st_size, st.st_mtime_ns)
    except OSError:
        return None


_CSV_MEMO: dict = {}


def read_csv(p: Path, usecols=None) -> pd.DataFrame | None:
    """Tolerant CSV read, memoised on (size, mtime)."""
    s = _sig(p)
    if s is None:
        return None
    key = (str(p), tuple(usecols) if usecols else None)
    hit = _CSV_MEMO.get(key)
    if hit and hit[0] == s:
        return hit[1]
    try:
        if usecols:
            cols = pd.read_csv(p, nrows=0).columns
            usecols = [c for c in usecols if c in cols]
        d = pd.read_csv(p, usecols=usecols, on_bad_lines="skip", low_memory=False)
    except Exception:
        return hit[1] if hit else None
    _CSV_MEMO[key] = (s, d)
    return d


def read_many(files, usecols=None, dedup="update") -> pd.DataFrame | None:
    parts = [d for d in (read_csv(Path(f), usecols) for f in files) if d is not None and len(d)]
    if not parts:
        return None
    d = pd.concat(parts, ignore_index=True)
    for c in ("update", "steps", "frontier", "start_frame", "deaths", "sr_ms"):
        if c in d and d[c].dtype == object:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    if dedup and dedup in d:
        d = d.dropna(subset=[dedup]).drop_duplicates(dedup, keep="last").sort_values(dedup)
    return d.reset_index(drop=True)


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def tz_label() -> str:
    off = dt.datetime.now().astimezone().utcoffset() or dt.timedelta(0)
    h = off.total_seconds() / 3600
    return f"UTC{h:+g}"


# --------------------------------------------------------------------------------------- runs

class Run:
    def __init__(self, name: str, path: Path, remote: bool):
        self.name, self.path, self.remote = name, path, remote

    @property
    def num(self):
        return int(re.match(r"r(\d+)", self.name).group(1))

    def files(self, pattern):
        return sorted(glob.glob(str(self.path / pattern)))


def discover_runs() -> list[Run]:
    runs: dict[str, Run] = {}
    for base, remote in ((PC_DIR, False), (MAC_DIR, True)):
        if not base.is_dir():
            continue
        for p in base.iterdir():
            if p.is_dir() and re.fullmatch(r"r\d+(?:-[a-z0-9]+)?", p.name):  # r10, r10-hs
                runs[p.name] = Run(p.name, p, remote)  # a Mac run wins over a same-named PC dir
    return sorted(runs.values(), key=lambda r: r.num)


def mac_runs(runs):
    return [r for r in runs if r.remote]


def is_active(run: Run, max_age_s=1800) -> bool:
    if (run.path / "STOP").exists():
        return False
    s = _sig(run.path / "train.log")
    return bool(s) and (dt.datetime.now().timestamp() - s[1] / 1e9) < max_age_s


def default_run(runs) -> str | None:
    m = mac_runs(runs)
    if not m:
        return None
    act = [r for r in m if is_active(r)]
    return (act or m)[-1].name


def run_config(run: Run) -> dict:
    try:
        c = json.loads((run.path / "config.json").read_text(encoding="utf-8"))
        return c if isinstance(c, dict) else {}
    except Exception:
        return {}


# ------------------------------------------------------------------------------------- levels

def record_ok(record, level: int | None = None) -> bool:
    """A TAS record name as the dataset spells it (SR<level>_...), of `level` when given, present on this machine."""
    m = RECORD_RE.match(str(record or ""))
    if not m or (level is not None and int(m.group(1)) != int(level)):
        return False
    return (DATASET / str(record) / "starts.npz").is_file()


def run_level(run: Run) -> dict:
    """{"level", "record"} of a run from its config (args --level / --record; runs before r8 have neither: level 1).
    The config is mirrored from the training machine, so the record must be a dataset record of that level."""
    c = run_config(run)
    try:
        level = int(c.get("level") or 1)
    except (TypeError, ValueError):
        level = 1
    if not 1 <= level <= 13:
        level = 1
    rec = c.get("record") or LEVEL_RECORDS.get(level)
    if not record_ok(rec, level):
        rec = LEVEL_RECORDS.get(level) if record_ok(LEVEL_RECORDS.get(level), level) else None
    return {"level": level, "record": rec}


def _local_dir(path_dir) -> Path | None:
    """A config's path_dir on this checkout: repo-relative as stored since r7, or another machine's absolute path
    mapped by its runtime/... suffix (r5, r6)."""
    if not path_dir:
        return None
    p = str(path_dir).replace("\\", "/")
    m = re.search(r"(?:^|/)(runtime/.+|benchmarks/.+)$", p)
    if m:
        return ROOT / m.group(1)
    q = Path(p)
    return None if q.is_absolute() else ROOT / q


def ref_line(run: Run) -> Path:
    """The run's reference line: <path_dir>/L0N.npz, else benchmarks/refpaths/L0N.npz."""
    name = f"L{run_level(run)['level']:02d}.npz"
    d = _local_dir(run_config(run).get("path_dir"))
    if d is not None and (d / name).is_file():
        return d / name
    return REFPATHS / name


def run_uses_alt(run: Run) -> bool:
    """The run follows another reference line than the TAS one (L1: runtime/refpaths-alt, 鬼躲石变)."""
    p = ref_line(run)
    return p.is_file() and p.resolve() != (REFPATHS / p.name).resolve()


PLACEMENTS = ROOT / "benchmarks" / "placements"


def placements_path(level: int) -> Path:
    return PLACEMENTS / f"L{int(level):02d}.json"


def placements(level: int) -> dict:
    """The level's placed objects for the maps (benchmarks/placements/L<nn>.json, from the level file): mechanisms,
    transformers, prop balls, score balls and life balls as [x, z, kind, label, sector]."""
    try:
        doc = json.loads(placements_path(level).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"items": []}
    return {"items": [[round(float(it["position"][0]), 2), round(float(it["position"][2]), 2), it["kind"], it["label"],
                       it.get("sector")] for it in doc.get("items", [])]}


def alt_name(level: int) -> str:
    return "鬼躲石变" if int(level) == 1 else "替代路线"


def shortcut_extra_s(run: Run) -> float:
    """Extra seconds of the alternative route over the TAS line, from the run's own split_extra setting
    ([[from_frame, to_frame, seconds], ...]); SHORTCUT_EXTRA_S (level 1) only if the run does not record it."""
    dflt = SHORTCUT_EXTRA_S if run_level(run)["level"] == 1 else 0.0
    v = run_config(run).get("split_extra")
    try:
        v = json.loads(v) if isinstance(v, str) else v
        return round(sum(float(x[2]) for x in v), 2) if v else dflt
    except Exception:
        return dflt


class TasClock:
    """The TAS record's own SR timer by frame (starts.npz: frames_done -> sr_double, BML+ 0.3.12 sum), its final
    time and LevelFinish frame, and the dashboard's start-frame bins for it."""

    def __init__(self, record: str):
        z = np.load(DATASET / record / "starts.npz", allow_pickle=False)
        o = np.argsort(z["frames_done"])
        self.record, self.level = record, int(record[2:4])
        self.fd = np.asarray(z["frames_done"], np.float64)[o]
        self.sr = np.asarray(z["sr_double"], np.float64)[o]
        self.total_ms = int(math.trunc(float(z["sr_final"])))
        self.end_frame = int(z["levelfinish_frame"])
        self.bins = list(range(0, self.end_frame // SR_BIN_FRAMES * SR_BIN_FRAMES, SR_BIN_FRAMES)) + [self.end_frame + 1]

    def ms(self, k):
        """TAS SR timer at frame(s) k (0 before the spawn; past the last start frame it runs on at 1000/264 ms)."""
        k = np.asarray(k, dtype=float)
        v = np.interp(k, self.fd, self.sr, left=0.0)
        return np.where(k > self.fd[-1], self.sr[-1] + (k - self.fd[-1]) * 1000.0 / 264.0, v)


_CLOCKS: dict = {}


def tas_clock(record: str | None = None) -> TasClock:
    record = record or LEVEL_RECORDS[1]
    if record not in _CLOCKS:
        _CLOCKS[record] = TasClock(record)
    return _CLOCKS[record]


def bin_edges(record: str | None) -> list[int]:
    """Start-frame bin edges of a TAS record; level 1's when the dataset is not on this machine."""
    try:
        return tas_clock(record).bins
    except OSError:
        return list(range(0, 20000, SR_BIN_FRAMES)) + [20588]


def tas_trace(record: str) -> Path:
    return DATASET / record / "trace.txt"


def n_actions(c: dict) -> int:
    """Size of the policy's action set (ballrain.env: 9 arrows, +2 camera turns, +1 Esc)."""
    return 9 + (2 if c.get("camera_actions") else 0) + (1 if c.get("esc") else 0)


# ---------------------------------------------------------------------------------- clocks

def mac_offset_hours(runs) -> float:
    """Mac log wall clock minus PC local clock, rounded to 30 min (log strings are Mac local time)."""
    for r in reversed(mac_runs(runs)):
        for j in reversed(r.files("episodes-traced/*/*/*.json")[-20:]):
            m = re.search(r"-(\d{8}-\d{6})-", Path(j).name)
            try:
                started = json.loads(Path(j).read_text(encoding="utf-8")).get("started_at")
            except Exception:
                continue
            if m and started:
                mac = dt.datetime.strptime(m.group(1), "%Y%m%d-%H%M%S")
                pc = dt.datetime.fromtimestamp(float(started))
                return round((mac - pc).total_seconds() / 1800) * 0.5
        lp = r.path / "train.log"
        s = _sig(lp)
        if s:
            last = None
            for line in log_lines(r)[-50:]:
                m = LOG_RE.match(line)
                if m:
                    last = m.group(1)
            if last:
                mac = dt.datetime.strptime(last, "%Y-%m-%d %H:%M:%S")
                pc = dt.datetime.fromtimestamp(s[1] / 1e9)
                return round((mac - pc).total_seconds() / 1800) * 0.5
    return DEFAULT_MAC_OFFSET_H


_LOG_MEMO: dict = {}


def log_lines(run: Run) -> list[str]:
    p = run.path / "train.log"
    s = _sig(p)
    if s is None:
        return []
    hit = _LOG_MEMO.get(str(p))
    if hit and hit[0] == s:
        return hit[1]
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    _LOG_MEMO[str(p)] = (s, lines)
    return lines


def to_pc(ts: str, run: Run, off_h: float) -> dt.datetime:
    t = dt.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
    return t - dt.timedelta(hours=off_h) if run.remote else t


def fmt(t: dt.datetime) -> str:
    return t.strftime("%m-%d %H:%M")


def epoch(t: dt.datetime) -> int:
    """Unix seconds of a naive PC-local time: the page formats every time in the viewer's own time zone."""
    return int(t.timestamp())


def ts_from_label(at, tz) -> int | None:
    """Unix seconds of "YYYY-MM-DD HH:MM" written in time zone "UTC+h[:mm]" (reports, highlight json)."""
    try:
        t = dt.datetime.strptime(str(at)[:16], "%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return None
    m = re.fullmatch(r"UTC([+-])(\d{1,2})(?::(\d{2}))?", str(tz or ""))
    if not m:
        return None
    off = dt.timedelta(hours=int(m.group(2)), minutes=int(m.group(3) or 0)) * (1 if m.group(1) == "+" else -1)
    return int(t.replace(tzinfo=dt.timezone(off)).timestamp())


# ---------------------------------------------------------------------------------- status

def run_status(run: Run, off_h: float, L, machines: dict) -> dict:
    """L: the run's Level (geometry.Level; None while it is not loaded yet: no covered share then)."""
    lines = log_lines(run)
    st: dict = {"run": run.name, "machine": machines["train" if run.remote else "eval"], "level": run_level(run)["level"]}
    cfg = run_config(run)
    if cfg.get("advance_at") is not None:
        st["advance_at"] = float(cfg["advance_at"])
    st["random_entropy"] = round(math.log(n_actions(cfg)), 2)
    last_ts = None
    curr = []
    first_ts = None
    for line in lines:
        m = LOG_RE.match(line)
        if not m:
            continue
        ts, msg = m.groups()
        first_ts = first_ts or ts
        last_ts = ts
        if CURR_OK_RE.match(msg):
            curr.append((ts, msg))
    now = dt.datetime.now()
    if last_ts:
        t = to_pc(last_ts, run, off_h)
        st["last_log"] = fmt(t)
        st["last_log_ts"] = epoch(t)
        st["last_log_age_min"] = round((now - t).total_seconds() / 60, 1)
    if first_ts:
        st["started"] = fmt(to_pc(first_ts, run, off_h))
        st["started_ts"] = epoch(to_pc(first_ts, run, off_h))
    st["curriculum"] = [f"{fmt(to_pc(t, run, off_h))}  {curriculum_text(msg)}" for t, msg in curr[-6:]]
    st["curriculum_items"] = [{"ts": epoch(to_pc(t, run, off_h)), "text": curriculum_text(msg)} for t, msg in curr[-6:]]

    u = read_many(run.files("updates*.csv"))
    if u is not None and len(u):
        r = u.iloc[-1]
        tail = u.tail(20)
        for k in ("update", "steps", "frontier", "frontier_sector", "band_success", "band_n", "u_entropy",
                  "u_approx_kl", "finish_tas", "success_cp_tas", "best_sr_any", "best_sr_k0"):
            if k in u:
                st[k] = _num(r[k])
        st["steps_per_s"] = _num(r.get("steps_per_s"))
        st["steps_per_s_avg20"] = _num(tail["steps_per_s"].mean()) if "steps_per_s" in u else None
        if "finish_tas" in u:
            st["finish_tas_avg20"] = _num(tail["finish_tas"].mean())
        if "band_success" in u:
            st["band_success_avg20"] = _num(tail["band_success"].mean())
        if st.get("frontier") is not None and L is not None:
            st["covered_pct"] = round(100 * (L.n - st["frontier"]) / (L.n - L.first), 1)
    e = read_many(run.files("episodes*.csv"), usecols=["update", "backend", "steps", "host", "finish", "sr_ms_end",
                                                       "start_frame"], dedup=None)
    # best full-level time among finishes that started near the current curriculum frontier: the trainer's own
    # "best SR" mostly comes from takeovers a few frames before the airship, i.e. it is TAS's time, not the agent's
    st.pop("best_sr_any", None)
    fr = st.get("frontier")
    if e is not None and len(e) and fr is not None and {"finish", "sr_ms_end", "start_frame"} <= set(e.columns):
        near = e[(e["finish"] == 1) & (e["start_frame"] <= fr + BEST_NEAR_FRONTIER) & e["sr_ms_end"].notna()]
        if len(near):
            i = near["sr_ms_end"].idxmin()
            st["best_sr_near"] = _num(near.loc[i, "sr_ms_end"])
            st["best_sr_near_k"] = int(near.loc[i, "start_frame"])
    if e is not None and len(e) and "backend" in e:
        e = e[e["update"] >= e["update"].max() - 100]
        # share of training steps (what the learner consumes), not of episodes: retail episodes are much shorter
        w = e["steps"] if "steps" in e else pd.Series(1, index=e.index)
        vc = w.groupby(e["backend"]).sum().sort_values(ascending=False)
        tot = float(vc.sum())
        if tot > 0:
            st["backend_share"] = {BACKEND_LABEL.get(str(k), "其他"): round(100 * float(v) / tot, 1) for k, v in vc.items()}
            src = (machines or {}).get("sources") or {}
            hosts = e["host"] if "host" in e else pd.Series(None, index=e.index, dtype=object)
            labels = [machine_of(b, h, src) for b, h in zip(e["backend"], hosts)]
            mv = w.groupby(pd.Series(labels, index=e.index)).sum().sort_values(ascending=False)
            st["source_share"] = {str(k): round(100 * float(v) / tot, 1) for k, v in mv.items()}
            lab = pd.Series(labels, index=e.index)
            st["source_detail"] = {str(m): [BACKEND_LABEL.get(str(b), "其他") for b in
                                            w[lab == m].groupby(e["backend"][lab == m]).sum().sort_values(ascending=False).index]
                                   for m in mv.index}
    age = st.get("last_log_age_min")
    # a STOP file only means "ended" if nothing was logged after it: the Mac -> PC pull copies files but never
    # deletes them, so a STOP from an earlier stop / resume stays in the mirrored run directory
    stop = run.path / "STOP"
    stopped = stop.exists() and (age is None or dt.datetime.fromtimestamp(stop.stat().st_mtime)
                                 >= now - dt.timedelta(minutes=age))
    if stopped or (age is not None and age > 120):
        st["state"] = "已结束"
    elif age is not None and age < 20:
        st["state"] = "训练中"
    else:
        st["state"] = "暂无新数据"
    try:
        mi = model_info(run)
        if mi:
            st["model"] = mi
    except Exception:  # a half-copied checkpoint: show nothing rather than fail the status
        pass
    return st


# observation blocks grouped for the model-structure panel (ballrain.features.make_layout names)
_MODEL_GROUPS = (
    ("ball", ("valid", "ball_type", "vel_cam", "vel_path", "speed", "angvel", "cam_yaw_ref", "cam_yaw", "cam_rotating",
              "pos_world", "geom_contact", "fric_contact", "n_fric", "n_mind", "min_gap", "normal_cam", "surface",
              "partner_moving", "partner_rail", "rel_vel_partner")),
    ("terrain", ("edge_mesh", "edge_fan", "rail_dist", "rail_dir", "heights", "height_hit")),
    ("reach", ("reach",)),
    ("route", ("path_progress", "path_remaining", "path_lateral", "path_tangent_cam", "waypoints", "far_waypoints",
               "ref_speed")),
    ("bodies", ("bodies",)),
    ("game", ("lives", "points", "sr_s", "sector", "transforming", "ball_nav", "depth_margin", "followers",
              "follower_sources", "follower_near", "paused", "since_checkpoint")),
    ("contacts", ("contacts", "contact_agg", "rails_lr")),
)
_MODEL_MEMO: dict = {}
_CK_MEMO: dict = {}  # run name -> (checkpoint key, slim checkpoint)


def latest_ckpt(run: Run):
    """(key, slim checkpoint) of a run's newest checkpoint (latest.pt, else the last ckpt_*.pt), or None: only the
    parts the dashboard reads (args, env_config, n_obs, n_act, parameter count, branch_assets). Cached by mtime/size;
    a checkpoint that fails to load (half-copied by the pull) keeps the last good one."""
    cands = [run.path / "latest.pt"] if (run.path / "latest.pt").is_file() else sorted(run.path.glob("ckpt_*.pt"))
    if not cands:
        return None
    ck_path = cands[-1]
    try:
        st = ck_path.stat()
    except OSError:
        return _CK_MEMO.get(run.name)
    key = (str(ck_path), st.st_mtime_ns, st.st_size)
    hit = _CK_MEMO.get(run.name)
    if hit and hit[0] == key:
        return hit
    import torch
    try:
        ck = torch.load(ck_path, weights_only=False, map_location="cpu")
    except Exception:
        if hit:
            return hit
        raise
    slim = {k: ck.get(k) for k in ("args", "env_config", "n_obs", "n_act", "branch_assets")}
    slim["params"] = int(sum(v.numel() for v in (ck.get("model") or {}).values() if hasattr(v, "numel")))
    _CK_MEMO[run.name] = (key, slim)
    return _CK_MEMO[run.name]


def model_info(run: Run) -> dict | None:
    """Network structure of a run's newest checkpoint for the model panel: observation blocks (sizes from the
    checkpoint's own feature config), hidden sizes, actor heads, parameter count. Cached by checkpoint mtime/size.
    No paths or other machine details: the payload is public."""
    got = latest_ckpt(run)
    if got is None:
        return None
    key, ck = got
    if _MODEL_MEMO.get(run.name, (None,))[0] == key:
        return _MODEL_MEMO[run.name][1]
    import dataclasses
    from ballrain.features import FeatureConfig, make_layout
    a = ck.get("args") or {}
    f = dict((ck.get("env_config") or {}).get("features") or {})
    names = {x.name for x in dataclasses.fields(FeatureConfig)}
    layout = make_layout(FeatureConfig(**{k: v for k, v in f.items() if k in names}))
    sizes = dict(layout)
    groups = []
    for g, members in _MODEL_GROUPS:
        n = sum(sizes.get(m, 0) for m in members)
        if n:
            groups.append({"key": g, "size": int(n)})
    other = sum(n for m, n in layout if not any(m in mm for _, mm in _MODEL_GROUPS))
    if other:
        groups.append({"key": "other", "size": int(other)})
    params = int(ck.get("params") or 0)
    info = {"n_obs": int(ck.get("n_obs") or sum(n for _, n in layout)), "n_act": int(ck.get("n_act") or 9),
            "hidden": [int(x) for x in str(a.get("hidden", "256,256")).split(",") if x.strip()],
            "activation": str(a.get("activation", "tanh")), "heads": str(a.get("actor_heads") or "single"),
            "action_repeat": int(a.get("action_repeat") or 1), "features": str(f.get("version") or ""),
            "groups": groups, "params": params, "init_from": bool(a.get("init_from"))}
    _MODEL_MEMO[run.name] = (key, info)
    return info


# ------------------------------------------------------------------------------- route bundle
# The route a run follows comes from its own checkpoint (branch_assets: the learner-owned route bundle, pinned by
# env_config.features.branch_sha256), never from hardcoded lines. Each member line is a whole-level line that leaves
# the run's main (TAS) line somewhere and rejoins it; the stretch where it differs is drawn, the TAS stretch it
# replaces is the part of the TAS line it overrides. Display names only: ids not in the table are shown as they are.
ROUTE_NAMES = {"east-primary": "东台线", "east-platform": "东台平台线", "back-to-wood": "回木球线",
               "south-loop": "南侧绕行线", "stone-arrival": "石球到达线"}
# what the TAS line does in (level, sector) where a branch of that sector overrides it
TAS_ROUTE_NAMES = {(2, 3): "鬼躲纸变"}
ROUTE_ID_RE = re.compile(r"^[a-z][a-z0-9-]{0,31}$")
ROUTE_FAR = 3.0       # world units: a member line farther than this from the main line has left it
ROUTE_NEAR = 8.0      # a trace point farther than this from a line is not on it
ROUTE_MARGIN = 2.0    # ... and must be this much closer to it than to the competing line
ROUTE_MIN_LEN = 10.0  # world units of a life on a fork before the life is classified by that fork
ROUTE_FAMILY_SHARE = 0.5  # two members whose diverging stretches overlap this much are one route family
ROUTE_STEP = 1.0      # world units between the points used for classification
_ROUTE_MEMO: dict = {}
_PIN_MEMO: dict = {}  # checkpoint path -> (sig, (pin, guidance key))


def _nearest(A, B, chunk=256):
    """Distance from each point of A to the nearest point of B, and that point's index (3D, brute force)."""
    A = np.asarray(A, np.float32)
    B = np.asarray(B, np.float32)
    d = np.full(len(A), np.inf, np.float32)
    j = np.zeros(len(A), np.int64)
    if not len(A) or not len(B):
        return d, j
    for i in range(0, len(A), chunk):
        q = ((A[i:i + chunk, None, :] - B[None, :, :]) ** 2).sum(-1)
        j[i:i + chunk] = q.argmin(1)
        d[i:i + chunk] = np.sqrt(q[np.arange(len(q)), j[i:i + chunk]])
    return d, j


def _resample(P, step=ROUTE_STEP):
    """Points of a polyline at least `step` apart (first and last kept)."""
    P = np.asarray(P, np.float64)
    if len(P) < 3:
        return P
    keep, acc = [0], 0.0
    for i, di in enumerate(np.linalg.norm(np.diff(P, axis=0), axis=1), start=1):
        acc += di
        if acc >= step:
            keep.append(i)
            acc = 0.0
    if keep[-1] != len(P) - 1:
        keep.append(len(P) - 1)
    return P[keep]


def _runs_of(mask):
    """[lo, hi) index ranges where mask is True."""
    m = np.r_[False, np.asarray(mask, bool), False]
    e = np.flatnonzero(np.diff(m.astype(np.int8)))
    return list(zip(e[::2], e[1::2]))


def _members(bundle):
    """[(manifest entry, member blobs)] of a route bundle: the v2 members in manifest order, or the single v1 line."""
    if bundle.get("version") == 2:
        from ballrain.multi_branch import manifest
        return list(zip(manifest(bundle["manifest"]), bundle["members"]))
    return [({"id": "branch"}, bundle)]


def _guidance_key(bundle) -> str:
    """What the bundle tells a life to follow: the manifest entries and each member's line and description. Moving-
    handover tables (drive data) are left out: an update that only adds them changes no guidance."""
    import hashlib
    h = hashlib.sha256()
    for e, m in _members(bundle):
        h.update(json.dumps(e, sort_keys=True).encode())
        for k in ("line_npz", "desc_json"):
            h.update(hashlib.sha256(m.get(k) or b"").digest())
    return h.hexdigest()


def run_pin(run: Run) -> str | None:
    """The route-bundle pin of a run's newest checkpoint (None without a bundle). Only used server-side."""
    try:
        got = latest_ckpt(run)
    except Exception:
        return None
    if got is None or not got[1].get("branch_assets"):
        return None
    pin = (((got[1].get("env_config") or {}).get("features") or {}).get("branch_sha256"))
    return str(pin) if pin else "unpinned"


def run_routes(run: Run) -> dict | None:
    """The run's route bundle as geometry (server-side; 3D, world units), or None when its checkpoint has none:
    {"pin", "main" (the run's main line), "families": [{"id", "name", "guidance", "members": [{"id", "name",
    "guidance", "line", "div" (diverging stretches), "over" (mask of main points it overrides)}], "div", "over"}]}.
    Members whose diverging stretches overlap are one family (named after its guidance member, else its first)."""
    try:
        got = latest_ckpt(run)
    except Exception:
        return None
    if got is None or not got[1].get("branch_assets"):
        return None
    bundle = got[1]["branch_assets"]
    pin = run_pin(run)
    try:
        main_p = ref_line(run)
        msig = (str(main_p), _sig(main_p))
    except Exception:
        return None
    hit = _ROUTE_MEMO.get(run.name)
    if hit and hit[0] == (pin, msig):
        return hit[1]
    try:
        main = np.asarray(np.load(main_p)["points"], np.float64)
        members = []
        for k, (e, m) in enumerate(_members(bundle)):
            P = np.asarray(np.load(io.BytesIO(m["line_npz"]))["points"], np.float64)
            far = _nearest(P, main)[0] > ROUTE_FAR
            over = _nearest(main, P)[0] > ROUTE_FAR
            if not far.any():
                continue
            mid = str(e.get("id") or "")
            mid = mid if ROUTE_ID_RE.match(mid) else f"branch-{k + 1}"
            div = [P[a:b] for a, b in _runs_of(far) if b - a >= 2]
            try:
                sector = int(json.loads(m["desc_json"]).get("sector"))
            except (TypeError, ValueError):
                sector = None
            members.append({"id": mid, "name": ROUTE_NAMES.get(mid, mid), "guidance": e.get("entry_context") == "ordinary",
                            "line": P, "div": div, "over": over, "sector": sector})
    except Exception:
        return hit[1] if hit else None
    # route families: members whose diverging stretches lie on each other's line
    parent = list(range(len(members)))

    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i
    for i, a in enumerate(members):
        for j in range(i + 1, len(members)):
            b = members[j]
            da, db = np.vstack(a["div"]), np.vstack(b["div"])
            sa = float(np.mean(_nearest(da, b["line"])[0] <= ROUTE_FAR)) if len(da) else 0.0
            sb = float(np.mean(_nearest(db, a["line"])[0] <= ROUTE_FAR)) if len(db) else 0.0
            if max(sa, sb) >= ROUTE_FAMILY_SHARE:
                parent[root(j)] = root(i)
    fams: dict = {}
    for i, m in enumerate(members):
        fams.setdefault(root(i), []).append(m)
    families = []
    for ms in fams.values():
        head = next((m for m in ms if m["guidance"]), ms[0])
        families.append({"id": head["id"], "name": head["name"], "guidance": any(m["guidance"] for m in ms), "members": ms,
                         "sector": head["sector"],
                         "div": np.vstack([np.vstack(m["div"]) for m in ms]),
                         "over": np.any([m["over"] for m in ms], axis=0)})
    families.sort(key=lambda f: not f["guidance"])  # the current guidance first
    for f in families:  # the classification's reference points, ROUTE_STEP apart
        f["div_r"] = _resample(f["div"])
        f["over_r"] = _resample(main[f["over"]]) if f["over"].any() else main[:0]
    region = np.vstack([f["div"] for f in families] + [main[f["over"]] for f in families])
    res = {"pin": pin, "main": main, "families": families, "level": run_level(run)["level"],
           "main_r": _resample(main), "region_r": _resample(region, 2.0)}
    _ROUTE_MEMO[run.name] = ((pin, msig), res)
    return res


def tas_class_name(R, f) -> str:
    """Name of the TAS stretch family f overrides: the level's name for it when the table has one, else generic."""
    tas = TAS_ROUTE_NAMES.get((int(R.get("level") or 0), f.get("sector")))
    return f"TAS 线（{tas}）" if tas else f"TAS 线（未走{f['name']}）"


def route_classes(R) -> list[dict]:
    """The route classes of a bundle in display order (public): its families, then per family the TAS line where
    that family overrides it ("tas-<family id>"), then lives that never reached a fork."""
    if not R:
        return []
    return ([{"key": f["id"], "name": f["name"], "guidance": f["guidance"]} for f in R["families"]]
            + [{"key": f"tas-{f['id']}", "name": tas_class_name(R, f), "guidance": False, "tas": True,
                "fork_guidance": f["guidance"]} for f in R["families"] if f["over"].any()]
            + [{"key": "shared", "name": "未经过分岔段", "guidance": False}])


def routes_public(R, L, cur: int | None = None) -> dict | None:
    """Map layers of a route bundle on level L (public: XZ coordinates, display names, ids only): each family's
    member lines where they leave the TAS line, the TAS line split into the part no branch overrides and the part a
    branch overrides, and with `cur` (the curriculum frontier) the covered stretch along the current guidance."""
    if not R or L is None:
        return None
    xz = lambda P: [[round(float(p[0]), 2), round(float(p[2]), 2)] for p in P]  # noqa: E731
    ok = L.ok[::3]
    T = L.pos[ok]
    _, near = _nearest(T, R["main"])
    over_any = np.zeros(len(T), bool)
    over_g = np.zeros(len(T), bool)
    for f in R["families"]:
        o = f["over"][near]
        over_any |= o
        if f["guidance"]:
            over_g |= o
    split = lambda mask: [xz(T[a:min(b + 1, len(T))]) for a, b in _runs_of(mask) if b - a >= 1]  # noqa: E731
    tas = [TAS_ROUTE_NAMES[(int(L.level), f["sector"])] for f in R["families"]
           if (int(L.level), f["sector"]) in TAS_ROUTE_NAMES and f["over"].any()]
    out = {"tas_name": f"TAS 参考线（{tas[0]}）" if tas else "TAS 参考线",
           "tas_main": split(~over_any), "tas_over": split(over_any),
           "families": [{"id": f["id"], "name": f["name"], "guidance": f["guidance"],
                         "members": [{"id": m["id"], "name": m["name"], "guidance": m["guidance"],
                                      "segs": [xz(_resample(d, 2.0)) for d in m["div"]]} for m in f["members"]]}
                        for f in R["families"]],
           "classes": route_classes(R)}
    if cur is not None:
        # the covered stretch along the current guidance: the TAS line from the frontier on, minus what the guidance
        # overrides, plus the guidance members' own stretches that lie ahead of the frontier
        ahead = ok >= cur
        segs = split(ahead & ~over_g)
        for f in R["families"]:
            if not f["guidance"]:
                continue
            for m in f["members"]:
                for d in m["div"]:
                    j = int(_nearest(d[:1], L.pos[L.ok])[1][0])  # where the stretch leaves the TAS line
                    if L.ok[j] >= cur:
                        segs.append(xz(_resample(d, 2.0)))
        out["covered"] = segs
    return out


def classify_life(tr, R, info=None) -> tuple[str, str]:
    """(route class, source) of one traced life against a route bundle: the branch member the start state records
    (rest starts on a branch) when there is one, else where the life went: per point the nearer of a family's own
    stretch and the main line (by ROUTE_MARGIN, within ROUTE_NEAR), summed as length per family and per overridden TAS
    stretch ("tas-<family>"). The fork of the current guidance decides first; then the longest; under ROUTE_MIN_LEN:
    "shared"."""
    if not R:
        return "", ""
    rec = (info or {}).get("branch_id")
    if rec:
        for f in R["families"]:
            if any(m["id"] == rec for m in f["members"]):
                return f["id"], "recorded"
    pts, w = tr.get("p3"), tr.get("w3")
    if pts is None or not len(pts):
        return "shared", "spatial"
    pts = np.asarray(pts, np.float64)
    w = np.asarray(w, np.float64)
    reg = R["region_r"]
    box = np.all((pts >= reg.min(0) - ROUTE_NEAR - 2.0) & (pts <= reg.max(0) + ROUTE_NEAR + 2.0), axis=1)
    pts, w = pts[box], w[box]
    cand = _nearest(pts, reg)[0] <= ROUTE_NEAR + 2.0
    if not cand.any():
        return "shared", "spatial"
    P, W = pts[cand], w[cand]
    d_main = _nearest(P, R["main_r"])[0]
    score: dict = {}
    for f in R["families"]:
        d_f = _nearest(P, f["div_r"])[0]
        on_f = (d_f <= ROUTE_NEAR) & (d_f < d_main - ROUTE_MARGIN)
        if len(f["over_r"]):
            d_o = _nearest(P, f["over_r"])[0]
            on_t = (d_o <= ROUTE_NEAR) & (d_o < d_f - ROUTE_MARGIN)
        else:
            on_t = np.zeros(len(P), bool)
        score[f["id"]] = (float(W[on_f].sum()), float(W[on_t].sum()))
    for f in R["families"]:  # the current guidance's fork first
        if f["guidance"]:
            g, t = score[f["id"]]
            if max(g, t) >= ROUTE_MIN_LEN:
                return (f["id"] if g >= t else f"tas-{f['id']}"), "spatial"
    best, label = 0.0, "shared"
    for f in R["families"]:
        g, t = score[f["id"]]
        if g > best:
            best, label = g, f["id"]
        if t > best:
            best, label = t, f"tas-{f['id']}"
    return (label if best >= ROUTE_MIN_LEN else "shared"), "spatial"


def _pin_of_ckpt(p: Path):
    """(pin, guidance key) of a checkpoint file, memoised by size/mtime; None when it cannot be read."""
    s = _sig(p)
    hit = _PIN_MEMO.get(str(p))
    if hit and hit[0] == s:
        return hit[1]
    try:
        import torch
        ck = torch.load(p, weights_only=False, map_location="cpu")
        b = ck.get("branch_assets")
        pin = (((ck.get("env_config") or {}).get("features") or {}).get("branch_sha256"))
        res = (str(pin) if pin else None, _guidance_key(b) if b else None)
    except Exception:
        return None
    _PIN_MEMO[str(p)] = (s, res)
    return res


def _pin_keys(run: Run, pins) -> dict:
    """Guidance keys of route-bundle pins from the run's checkpoints (newest first, stopping once all are found)."""
    want, out = set(pins), {}
    try:
        got = latest_ckpt(run)
        if got and got[1].get("branch_assets"):
            out[run_pin(run)] = _guidance_key(got[1]["branch_assets"])
    except Exception:
        pass
    files = sorted(run.path.glob("*.pt"), key=lambda p: (_sig(p) or (0, 0))[1], reverse=True)
    for p in files:
        if want <= set(out):
            break
        r = _pin_of_ckpt(p)
        if r and r[0] in want and r[0] not in out and r[1]:
            out[r[0]] = r[1]
    return out


def _resume_times(run: Run, off_h: float) -> list[int]:
    out = []
    for line in log_lines(run):
        m = LOG_RE.match(line)
        if m and m.group(2).startswith("resumed from "):
            out.append(epoch(to_pc(m.group(1), run, off_h)))
    return out


def guidance_since(run: Run, idx: list[dict], off_h: float) -> dict | None:
    """When the run's route guidance last changed: {"ts" (Unix s), "pins" (server-side only)} or None. The pin
    sequence is branch-updates.jsonl in the run directory, else the pins its traced lives carry in time order; the
    last step that changed the guidance (manifest, member lines or descriptions: an update that only adds moving
    handover tables does not) starts the window. Its time: the trainer's resume between the last life on the old
    bundle and the first on the new one, else that first life."""
    seq: list = []
    try:
        for line in (run.path / "branch-updates.jsonl").read_text(encoding="utf-8").splitlines():
            try:
                j = json.loads(line)
            except ValueError:
                continue
            if not isinstance(j, dict) or not j.get("to"):
                continue
            if not seq and j.get("from"):
                seq.append(str(j["from"]))
            seq.append(str(j["to"]))
    except OSError:
        pass
    if not seq:
        for i in idx:
            if i.get("pin") and (not seq or seq[-1] != i["pin"]):
                seq.append(i["pin"])
    if len(seq) < 2:
        return None
    keys = _pin_keys(run, set(seq))
    at = None
    for j in range(len(seq) - 1, 0, -1):
        a, b = keys.get(seq[j - 1]), keys.get(seq[j])
        if a is None or b is None or a != b:
            at = j
            break
    if at is None:
        return None
    post = set(seq[at:])
    new = [i["t"] for i in idx if i.get("pin") in post and i["t"]]
    if not new:
        return None
    t_first = min(new)
    old = [i["t"] for i in idx if i.get("pin") not in post and 0 < i["t"] < t_first]
    t_old = max(old, default=None)
    res = [t for t in _resume_times(run, off_h) if t <= t_first + 1]
    t_res = max(res, default=None)
    ts = t_res if t_res is not None and (t_old is None or t_res >= t_old) else t_first
    return {"ts": int(ts), "pins": post}


_STEPS_MEMO: dict = {}
_STEPS_RE = re.compile(r" u(\d+) steps (\d+)")
_INIT_RE = re.compile(r"initialised from .*?update (\d+)")


def run_own_steps(run: Run) -> tuple[int, int]:
    """(steps this run trained itself, its last step counter) from its train.log. The counter carries on from the
    checkpoint a run was initialised from; a rollback inside the run makes it drop and the stretch trained again
    after the drop counts too (sum of the counter's increases), plus the updates before the first log line."""
    lp = run.path / "train.log"
    sig = _sig(lp)
    if sig is None:
        return 0, 0
    hit = _STEPS_MEMO.get(str(lp))
    if hit and hit[0] == sig:
        return hit[1]
    own, prev, prev_u, last, init_u, first, spu = 0, None, None, 0, 0, None, []
    for line in lp.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _INIT_RE.search(line)
        if m and first is None:
            init_u = int(m.group(1))
        m = _STEPS_RE.search(line)
        if not m:
            continue
        u, st = int(m.group(1)), int(m.group(2))
        if first is None:
            first = (u, st)
        if prev is not None:
            if st > prev:
                own += st - prev
                if u > prev_u:
                    spu.append((st - prev) / (u - prev_u))
        prev, prev_u, last = st, u, st
    if first is not None:
        rate = float(np.median(spu)) if spu else 0.0
        # before the first log line: the updates since the parent checkpoint, or from zero for a fresh run
        own += int(max(0, first[0] - init_u) * rate) if init_u else first[1]
    res = (own, last)
    _STEPS_MEMO[str(lp)] = (sig, res)
    return res


_CKPT_U_RE = re.compile(r"(?:ckpt_|-u)(\d+)\.pt$")


def _counter_at(run: Run, update: int) -> int | None:
    """The run's step counter at its last logged update <= update (from train.log)."""
    lp = run.path / "train.log"
    if not lp.is_file():
        return None
    best = None
    for line in lp.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _STEPS_RE.search(line)
        if m and int(m.group(1)) <= update:
            best = int(m.group(2))
    return best


def _lineage_steps(runs, current: str) -> int:
    """Steps in the current model's lineage: its own counter, plus, when a run was initialised from another run's
    checkpoint with a fresh counter (e.g. a new level), the parent's counter at that checkpoint (recursively)."""
    by = {r.name: r for r in runs}
    total, name, seen = 0, current, set()
    while name in by and name not in seen:
        seen.add(name)
        r = by[name]
        own, last = run_own_steps(r)
        total += last
        try:
            cfg = json.loads((r.path / "config.json").read_text(encoding="utf-8"))
            init = str((cfg.get("args") or cfg).get("init_from") or "")
        except (OSError, ValueError):
            break
        parts = init.replace("\\", "/").split("/")
        m = _CKPT_U_RE.search(init)
        parent = parts[-2] if len(parts) >= 2 else None
        if not parent or parent not in by or not m:
            break
        at = _counter_at(by[parent], int(m.group(1)))
        first = _counter_at(r, 10 ** 12)
        # the counter carried on from the parent: the parent's steps are already in `last`
        if at is None or (first is not None and last >= at):
            break
        total += at
        # continue with the parent's own lineage before that checkpoint is already inside `at` (carried counters)
        break
    return total


def discarded_steps(runs, current: str) -> dict:
    """All steps trained across the runs minus the steps in the current model's lineage: abandoned runs, the part of a
    parent trained after the checkpoint a later run branched from, and rollbacks. A run initialised with a fresh step
    counter (a new level) still counts its parent's steps up to the branch checkpoint as lineage."""
    total = sum(run_own_steps(r)[0] for r in runs)
    cur = _lineage_steps(runs, current)
    return {"total": total, "current": cur, "discarded": max(0, total - cur)}


def curriculum_text(msg: str) -> str:
    """'curriculum: frontier -> 4886 (sector 1) after rate 0.60' -> Chinese, without CLI flags."""
    m = re.match(r"curriculum: frontier (?:->|set (\d+) ->) (\d+)(?: \(sector (\d+)\))?(?: after rate ([\d.]+))?", msg)
    if not m:
        return sanitize(msg)
    old, new, sector, rate = m.groups()
    if old:
        return f"开局点 {old} → {new}（手动设定）"
    out = f"开局点 → {new}"
    if sector:
        out += f"（第 {sector} 小节）"
    if rate:
        out += f" · 区间成功率 {float(rate):.0%}"
    return out


# ---------------------------------------------------------------------------- frontier history

def frontier_history(runs, off_h, record: str | None = None) -> list[dict]:
    """Curriculum frontier moves of the runs (only those on TAS record `record` when given: frames of one level)."""
    out = []
    for r in runs:
        if r.num < 2:
            continue
        if record is not None and run_level(r)["record"] != record:
            continue
        for line in log_lines(r):
            m = LOG_RE.match(line)
            if not m:
                continue
            mf = FRONTIER_RE.search(m.group(2))
            if mf:
                t = to_pc(m.group(1), r, off_h)
                out.append({"run": r.name, "t": fmt(t), "ts": epoch(t), "frame": int(mf.group(1))})
    return out


def frontier_payload(L, runs, off_h, cur_run, cur, show_alt: bool, routes=None) -> dict:
    """The curriculum map of level L: its TAS line, checkpoints, the frontier moves of the runs on its record and
    the current frontier of `cur_run` (the level's latest run); with `routes` (run_routes of cur_run) also its
    route bundle's lines and the covered stretch along the current guidance ("routes")."""
    pos, ok, n = L.pos, L.ok, L.n
    tas = ok[::3]
    xz = lambda i: [round(float(pos[i, 0]), 2), round(float(pos[i, 2]), 2)]  # noqa: E731
    out = {
        "tas": [xz(i) for i in tas] + [xz(n - 1)],
        "tas_pts": [xz(i) + [int(i)] for i in ok[::150]],
        "covered": [xz(i) for i in tas if i >= cur] + [xz(n - 1)],
        "checkpoints": [xz(i) + [int(i), f"盘点 {k + 1}"] for k, i in enumerate(L.checkpoints)],
        "start": xz(ok[0]) + [int(ok[0]), "出生点"],
        "end": xz(n - 1) + [int(n - 1), "飞船"],
        "history": [dict(h, x=xz(min(h["frame"], n - 1))[0], z=xz(min(h["frame"], n - 1))[1])
                    for h in frontier_history(runs, off_h, L.record)],
        "current": {"run": cur_run, "frame": int(cur), "x": xz(min(cur, n - 1))[0], "z": xz(min(cur, n - 1))[1],
                    "covered_pct": round(100 * (n - cur) / (n - L.first), 1)},
        "alt": None, "alt_name": alt_name(L.level),
        "level": L.level, "n": int(n), "end_frame": int(L.clock.end_frame),
    }
    if routes:
        out["routes"] = routes_public(routes, L, cur)
    cr = next((r for r in runs if r.name == cur_run), None)
    if show_alt and cr is not None and run_uses_alt(cr):
        try:
            A = np.load(ref_line(cr))["points"]  # line geometry only
            T = np.load(REFPATHS / f"L{L.level:02d}.npz")["points"]
            d = np.min(np.linalg.norm(A[:, None, [0, 2]] - T[None, ::4, [0, 2]], axis=2), axis=1)
            idx = np.flatnonzero(d > 1.0)
            if len(idx):
                out["alt"] = [[round(float(A[i, 0]), 2), round(float(A[i, 2]), 2)] for i in idx]
        except Exception:
            pass
    return out


def curves_payload(runs, machines) -> dict:
    cols = ["update", "steps", "frontier", "band_success", "band_finish", "finish_tas", "u_entropy",
            "steps_per_s"]
    out = {}
    for r in runs:
        d = read_many(r.files("updates*.csv"), usecols=cols)
        if d is None or not len(d):
            continue
        w = max(3, min(25, len(d) // 40))
        sm = lambda c: d[c].rolling(w, min_periods=1).mean() if c in d else None  # noqa: E731
        step = max(1, math.ceil(len(d) / 2500))
        sel = np.arange(0, len(d), step)
        if sel[-1] != len(d) - 1:
            sel = np.append(sel, len(d) - 1)

        def col(s, nd):
            if s is None:
                return None
            v = s.to_numpy(dtype=float)[sel]
            return [None if math.isnan(x) else round(float(x), nd) for x in v]

        out[r.name] = {
            "machine": machines["train" if r.remote else "eval"], "smooth": w, "level": run_level(r)["level"],
            "update": [int(x) for x in d["update"].to_numpy()[sel]],
            "steps": col(d["steps"] / 1e6, 4),
            "frontier": col(d["frontier"], 0),
            "band": col(sm("band_success"), 3),
            "finish": col(sm("finish_tas"), 3),
            "band_finish": col(sm("band_finish"), 3),
            "entropy": col(sm("u_entropy"), 3),
            "sps": col(sm("steps_per_s"), 0),
        }
    return out


# ----------------------------------------------------------------------------------- SR estimate

def tas_ms(k, record: str | None = None):
    """TAS SR timer at frame k of the record (level 1 by default): the record's own sr_double by frame."""
    return tas_clock(record).ms(k)


# used when benchmarks/records.json lacks a value (same numbers as the table; TAS = BML+ 0.3.12 timer)
BENCH_DEFAULTS = {1: {"tas": 74446, "human_wr": 75853, "featured": 83499, "threshold": 92999},
                  2: {"tas": 88977, "human_wr": 96382, "featured": 143499, "threshold": 158499}}


def benchmarks(level: int = 1) -> dict:
    """SR thresholds of a level from benchmarks/records.json: TAS (current record, BML+ 0.3.12 timer), human record,
    精品线 and 上榜线 (values only, no player names)."""
    level = int(level)
    b = dict(BENCH_DEFAULTS.get(level) or {"tas": None, "human_wr": None, "featured": None, "threshold": None})
    try:
        for r in json.loads(RECORDS.read_text(encoding="utf-8")):
            if r.get("level") != level or r.get("category") != "SR":
                continue
            k = r.get("kind")
            if k == "tas" and r.get("is_current_record"):
                b["tas"] = r.get("sr_ms_bmlp0312") or r.get("value_ms")
            elif k == "human":
                b["human_wr"] = r["value_ms"]
            elif k == "human_featured":
                b["featured"] = r["value_ms"]
            elif k == "human_threshold":
                b["threshold"] = r["value_ms"]
    except Exception:
        pass
    return b


def sr_estimate(runs, sel: str) -> dict:
    """Full-level SR estimate of run `sel` (pooled with the run before it when that one ran the same TAS record)."""
    m = mac_runs(runs)
    names = [r.name for r in m]
    if sel not in names:
        return {"error": "无数据"}
    i = names.index(sel)
    lv = run_level(m[i])
    if lv["record"] is None:
        return {"error": "无数据", "level": lv["level"]}
    use = [r for r in m[max(0, i - 1): i + 1] if run_level(r)["record"] == lv["record"]]
    clk = tas_clock(lv["record"])
    cols = ["update", "end_reason", "sr_ms", "start_frame", "deaths", "invalid", "copy"]
    parts = [d for d in (read_many(r.files("runs*.csv"), usecols=cols, dedup=None) for r in use) if d is not None]
    out: dict = {"runs_used": [r.name for r in use], "window": SR_WINDOW_UPDATES, "bins": [], "level": lv["level"],
                 "benchmarks": benchmarks(lv["level"]), "alt": run_uses_alt(m[i]), "alt_name": alt_name(lv["level"]),
                 "tas_record_ms": clk.total_ms, "end_frame": clk.end_frame,
                 "shortcut_extra_s": shortcut_extra_s(m[i]), "min_n": SR_MIN_N, "min_seg_s": SR_MIN_SEG_S}
    if not parts:
        out["error"] = "无数据"
        return out
    d = pd.concat(parts, ignore_index=True)
    for c in ("update", "sr_ms", "start_frame", "deaths"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    umax = d["update"].max()
    if "invalid" in d:
        inv = d["invalid"]
        valid = inv.isna() | inv.astype(str).str.strip().str.lower().isin(["", "nan", "none", "false", "0", "0.0"])
    else:
        valid = True
    ok = ((d["update"] >= umax - SR_WINDOW_UPDATES) & (d["end_reason"] == "LevelFinish") & (d["start_frame"] > 0)
          & (d["deaths"] == 0) & d["sr_ms"].notna() & valid)
    d = d[ok].copy()
    out["update_range"] = [int(max(0, umax - SR_WINDOW_UPDATES)), int(umax)] if not math.isnan(umax) else None
    out["n"] = int(len(d))
    if "copy" in d:  # game copies "hl-*" are the headless backend, the rest the original game
        out["n_retail"] = int((~d["copy"].astype(str).str.startswith("hl")).sum())
    if not len(d):
        return out
    k = d["start_frame"].to_numpy(float)
    tk = clk.ms(k)
    seg_tas = clk.total_ms - tk
    seg_agent = d["sr_ms"].to_numpy(float) - tk
    good = seg_tas > 200
    d["ratio"] = np.where(good, seg_agent / np.where(good, seg_tas, 1), np.nan)
    keep = []
    for a, b in zip(clk.bins[:-1], clk.bins[1:]):
        r = d[(d["start_frame"] >= a) & (d["start_frame"] < b) & d["ratio"].notna()]["ratio"].to_numpy()
        mid = min((a + b) / 2, clk.end_frame)
        row = {"lo": a, "hi": min(b, clk.end_frame), "n": int(len(r)),
               "tas_seg_s": round(float(clk.total_ms - clk.ms(mid)) / 1000, 1)}
        if len(r):
            row["median"] = round(float(np.median(r)), 3)
            row["p10"] = round(float(np.percentile(r, 10)), 3)
            row["p90"] = round(float(np.percentile(r, 90)), 3)
            if len(r) >= SR_MIN_N and row["tas_seg_s"] >= SR_MIN_SEG_S:
                keep.append(row["median"])
                row["in_range"] = True
        out["bins"].append(row)
    if keep:
        lo, hi = min(keep), max(keep)
        extra = out["shortcut_extra_s"] if out["alt"] else 0.0
        tas_s = out["benchmarks"]["tas"] / 1000
        out["estimate_s"] = [round(tas_s * lo + extra * lo, 1), round(tas_s * hi + extra * hi, 1)]
        out["ratio_range"] = [lo, hi]
    return out


def _level_of_name(name: str) -> int:
    for base in (MAC_DIR, PC_DIR):
        if (base / name / "config.json").is_file():
            return run_level(Run(name, base / name, base == MAC_DIR))["level"]
    return 1


def k0_evals() -> list[dict]:
    """Evaluations from the very start of the level (k = 0) on the original game."""
    rows = []
    # results sit in <run>-sim2real/ (r3) or <run>-sim2real/pairs/ (r4 on); one-off check folders are not evaluations
    files = glob.glob(str(PC_DIR / "*-sim2real" / "*.jsonl")) + glob.glob(str(PC_DIR / "*-sim2real" / "pairs" / "*.jsonl"))
    for f in sorted(set(files)):
        run = next(part for part in Path(f).parts if part.endswith("-sim2real")).split("-")[0]
        try:
            lines = Path(f).read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                j = json.loads(line)
            except Exception:
                continue
            p = (j.get("per_start") or {}).get("0")
            if not p or j.get("invalid"):  # invalid: e.g. r5/r6 retail evals without the reference line
                continue
            if str(j.get("backend")) != "retail":  # this table is the original game only
                continue
            rows.append({"run": run, "update": j.get("update"), "level": _level_of_name(run),
                         "backend": BACKEND_LABEL.get(str(j.get("backend")), "其他"),
                         "greedy": bool(j.get("greedy")), "temperature": j.get("temperature"),
                         "runs": p.get("runs"), "success": p.get("success"),
                         "finished_no_death": p.get("finished_no_death"), "deaths_per_run": p.get("deaths_per_run"),
                         "sr_ms_median": p.get("sr_ms_median"), "sr_ms_min": p.get("sr_ms_min"),
                         "t": dt.datetime.fromtimestamp(j["t"]).strftime("%m-%d %H:%M") if j.get("t") else None,
                         "ts": int(j["t"]) if isinstance(j.get("t"), (int, float)) else None})
    return rows


# -------------------------------------------------------------------------------- traced runs

OUTCOME = {"LevelFinish": "finish", "GameOver": "gameover"}
BALLS = ("Ball_Wood", "Ball_Stone", "Ball_Paper")


class Traces:
    """Index of traced episodes (small json) + a disk cache of parsed, downsampled agent paths."""

    CACHE_VERSION = 5  # 4: any Ball_* line (level 2 switches to the paper ball), TAS line per record; 5: 3D points for routes

    def __init__(self, cache_file: Path):
        self.cache_file = cache_file
        self.meta: dict = {}     # json path -> (sig, info)   (paths stay server-side)
        self.parsed: dict = {}   # json path -> (sig, parsed)
        self._tas: dict = {}
        try:
            with open(cache_file, "rb") as f:
                c = pickle.load(f)
            if c.get("_v") == self.CACHE_VERSION:
                self.parsed = c["parsed"]
        except Exception:
            self.parsed = {}

    def tas(self, record: str | None = None):
        record = record or LEVEL_RECORDS[1]
        if record not in self._tas:
            fr, P = self._load(tas_trace(record))
            o = np.argsort(fr)
            self._tas[record] = (fr[o], P[o])
        return self._tas[record]

    @staticmethod
    def _load(p: Path):
        fr, P = [], []
        with open(p, encoding="utf-8", errors="replace") as f:
            for line in f:  # the player ball, whichever kind it is (wood / stone / paper)
                if " Ball_" not in line:
                    continue
                s = line.split()
                if len(s) >= 14 and s[1] in BALLS:
                    fr.append(int(s[0]))
                    P.append((float(s[11]), float(s[12]), float(s[13])))
        return np.asarray(fr, dtype=np.int64), np.asarray(P, dtype=np.float64).reshape(-1, 3)

    def index(self, run: Run) -> list[dict]:
        out = []
        rec = run_level(run)["record"]
        for j in run.files("episodes-traced/*/*/*.json"):
            p = Path(j)
            if not p.with_name("trace.txt").exists():
                continue
            s = _sig(p)
            hit = self.meta.get(j)
            if hit and hit[0] == s and hit[1].get("record") == rec:
                out.append(hit[1])
                continue
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            h = zlib.crc32(p.stem.encode())
            # the route bundle the life ran with and the branch member a rest start was placed on (server-side only)
            pin = (d.get("feature_config") or {}).get("branch_sha256")
            bid = str((d.get("start_state") or {}).get("branch_id") or "")
            info = {"key": j, "id": f"{h:08x}", "run": run.name, "end": OUTCOME.get(d.get("end_reason"), "stopped"),
                    "sr": d.get("sr_ms"), "t": float(d.get("started_at") or 0), "h": h / 2 ** 32,
                    "backend": BACKEND_LABEL.get(str(d.get("backend")), "其他"), "record": rec,
                    "pin": str(pin) if isinstance(pin, str) and pin else None,
                    "branch_id": bid if ROUTE_ID_RE.match(bid) else None}
            self.meta[j] = (s, info)
            out.append(info)
        out.sort(key=lambda r: r["t"])
        return out

    def get(self, info) -> dict | None:
        j = info["key"]
        tp = Path(j).with_name("trace.txt")
        s = _sig(tp)
        rec = info.get("record")
        if rec is None:
            return None
        hit = self.parsed.get(j)
        if hit and hit[0] == (s, rec):
            return hit[1]
        try:
            fr, P = self._load(tp)
        except OSError:
            return None
        if len(fr) < 100:
            return None
        tfr, tP = self.tas(rec)
        idx = np.clip(np.searchsorted(tfr, fr), 0, len(tfr) - 1)
        same = (tfr[idx] == fr) & np.all(tP[idx] == P, axis=1)
        bad = np.flatnonzero(~same)
        k = int(fr[bad[0]]) if len(bad) else int(fr[-1])
        m = fr >= k
        Q, F = P[m], fr[m]
        if len(Q) < 2:
            return None
        cut = np.flatnonzero(np.linalg.norm(np.diff(Q, axis=0), axis=1) > 4) + 1
        segs = [s_[:, [0, 2]].astype(np.float32) for s_ in np.split(Q, cut) if len(s_)]
        # 3D points ROUTE_STEP apart for the route classification, each weighted by its length along the path
        # (jumps between lives / respawns count nothing)
        p3 = np.vstack([_resample(s_) for s_ in np.split(Q, cut) if len(s_)]).astype(np.float32)
        step = np.linalg.norm(np.diff(p3, axis=0), axis=1)
        w3 = np.r_[np.where(step > 4, 0.0, step), 0.0].astype(np.float32)
        res = {"k": k, "segs": segs, "deaths": [(float(Q[i - 1, 0]), float(Q[i - 1, 2]), int(F[i - 1])) for i in cut],
               "start": (float(Q[0, 0]), float(Q[0, 2])), "end_frame": int(F[-1]), "p3": p3, "w3": w3}
        self.parsed[j] = ((s, rec), res)
        return res

    def save(self, keep_keys):
        self.parsed = {k: v for k, v in self.parsed.items() if k in keep_keys}
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_file.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump({"_v": self.CACHE_VERSION, "parsed": self.parsed}, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, self.cache_file)


def _thin(segs, budget):
    """Distance-based thinning of a trajectory's segments to <= budget points in total."""
    tot = sum(len(s) for s in segs)
    out = []
    for s in segs:
        if len(s) < 2:
            out.append(s)
            continue
        d = np.linalg.norm(np.diff(s, axis=0), axis=1)
        keep = [0]
        acc = 0.0
        for i, di in enumerate(d, start=1):
            acc += di
            if acc >= MIN_STEP:
                keep.append(i)
                acc = 0.0
        if keep[-1] != len(s) - 1:
            keep.append(len(s) - 1)
        out.append(s[keep])
    tot = sum(len(s) for s in out)
    if tot > budget:
        step = math.ceil(tot / budget)
        out = [np.vstack([s[::step], s[-1:]]) if len(s) > 2 else s for s in out]
    return out


def traj_payload(info, tr, R=None, memo: dict | None = None) -> dict:
    """One traced life for a map (public); with a route bundle R also its route class ("route") and whether that
    came from the life's recorded branch member or from where it went ("route_src")."""
    segs = _thin(tr["segs"], MAX_TRAJ_POINTS)
    route = {}
    if R:
        k = (info["key"], R["pin"])
        got = memo.get(k) if memo is not None else None
        if got is None:
            got = classify_life(tr, R, info)
            if memo is not None:
                memo[k] = got
        route = {"route": got[0], "route_src": got[1]}
    return {**route, 
        "id": info["id"], "run": info["run"], "end": info["end"], "backend": info["backend"],
        "t": dt.datetime.fromtimestamp(info["t"]).strftime("%m-%d %H:%M") if info["t"] else None,
        "ts": round(info["t"]), "sr": info["sr"], "k": tr["k"], "end_frame": tr["end_frame"],
        "start": [round(tr["start"][0], 1), round(tr["start"][1], 1)],
        "deaths": [[round(x, 1), round(z, 1), f] for x, z, f in tr["deaths"]],
        "segs": [[[round(float(p[0]), 1), round(float(p[1]), 1)] for p in s] for s in segs],
    }


def nested_sample(items: list[dict], target: int):
    """A hash-based subset whose membership is stable as runs are added (so cached parses are reused),
    then thinned evenly in time to at most `target`."""
    frac = 1.0
    sub = items
    while len(sub) > 2 * target:
        frac /= 2
        sub = [r for r in items if r["h"] < frac]
    thin = sub
    if len(sub) > target:
        idx = np.linspace(0, len(sub) - 1, target).astype(int)
        thin = [sub[i] for i in idx]
    return sub, thin

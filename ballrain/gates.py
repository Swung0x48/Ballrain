"""Pure gate logic for scripts/m0_gates.py: verdicts computed from files already on disk.

Nothing here launches the game or needs a game copy, so every rule can be unit-tested on
synthetic traces and re-applied to stored results. All trace comparisons are
ballrain.trace.compare with require_same_end=True (byte-exact, same terminal frame).
"""
from __future__ import annotations

import math
import os
from pathlib import Path

from . import tas, trace

# Bumped whenever a gate's definition changes, so results judged under an older definition
# are not reused by the driver's resume logic.
GATE_DEF_VERSION = 2

# G1 records left out of the strict set by user decision. They are still run and reported
# (including the framelog-export supplement), but count as neither pass nor fail.
_GAMEOVER_REASON = ("the reference itself reaches GameOver on this record; after GameOver the game must "
                    "start a new game, so playback past it is meaningless")
G1_EXCLUDED = {
    "SR08_1_59_626_264": _GAMEOVER_REASON,
    "SR09_1_29_321_264": _GAMEOVER_REASON,
    "HS13_8484_4_17_833": "the reference itself is not repeatable on this record",
}

# Every non-excluded G1 record is a completing record (a finished speedrun / high-score run).
G1_EXPECTED_STOP = "LevelFinish"

MIN_DISPLACEMENT = 1.0  # units; G0: the ball must move by more than this somewhere in the trace

G3_FALLBACK = "fallback: one process per episode"


# ------------------------------------------------------------------------------------- inputs

def read_events(path) -> list[tuple[int, str]]:
    p = Path(path)
    if not p.is_file():
        return []
    ev = []
    for line in p.read_text(encoding="latin-1").splitlines():
        idx, _, name = line.partition(" ")
        ev.append((int(idx), name))
    return ev


def summary_of(events) -> dict | None:
    for idx, name in events:
        if name.startswith("Summary"):
            kv = dict(x.split("=") for x in name.split()[1:])
            return {"frame": idx, "frames": int(kv["frames"]), "wall_ms": int(kv["wall_ms"])}
    return None


def stop_reason(events) -> str | None:
    names = [n for _, n in events]
    for r in ("RecordEnd", "LevelFinish", "GameOver"):
        if r in names:
            return r
    return None


def has_event(events, name: str) -> bool:
    return any(n == name for _, n in events)


def diff_dict(d: trace.TraceDiff) -> dict:
    return {
        "status": d.status, "pass": d.ok, "lines_a": d.lines_a, "lines_b": d.lines_b,
        "common_prefix": d.common_prefix, "first_diff_line": d.first_diff_index,
        "first_diff_frame_a": d.first_diff_frame_a, "first_diff_frame_b": d.first_diff_frame_b,
        "line_a": d.line_a.decode("latin-1").rstrip() if d.line_a else None,
        "line_b": d.line_b.decode("latin-1").rstrip() if d.line_b else None,
        "eol_only": d.eol_only, "first_frame_a": d.first_frame_a, "first_frame_b": d.first_frame_b,
        "last_frame_a": d.last_frame_a, "last_frame_b": d.last_frame_b, "same_end": d.same_end,
    }


def compare(a, b) -> dict:
    out = diff_dict(trace.compare(a, b, require_same_end=True))
    out["a"], out["b"] = str(a), str(b)
    out["cmd"] = f'uv run python -m ballrain.trace "{a}" "{b}"'
    return out


# --------------------------------------------------------------------------------- G0 sanity

def positions(path_or_bytes) -> list[tuple[float, float, float]]:
    """Ball world position per trace line: the last 3 of the 12 matrix values (row 3 = translation)."""
    out = []
    for line in trace.read_lines(path_or_bytes):
        body = line.split(b"|", 1)[0].split()
        vals = body[2:]
        if len(vals) < 12:
            continue
        try:
            out.append((float(vals[9]), float(vals[10]), float(vals[11])))
        except ValueError:
            continue
    return out


def max_displacement(path_or_bytes) -> float | None:
    """Largest distance of any traced position from the first one (None for an empty trace)."""
    pos = positions(path_or_bytes)
    if not pos:
        return None
    x0, y0, z0 = pos[0]
    return max(math.dist((x0, y0, z0), p) for p in pos)


def sanity(trace_path, events=None) -> dict:
    """G0 checks on one trace: non-empty, ball displacement > MIN_DISPLACEMENT, and (when an events
    list is given, i.e. the BallrainIO side) at least one BallNavActive. The reference copy writes
    no events file, so its BallNavActive check is reported as None (not applicable)."""
    p = Path(trace_path) if not isinstance(trace_path, (bytes, bytearray)) else None
    exists = p.is_file() if p is not None else True
    lines = len(trace.read_lines(trace_path)) if exists else 0
    disp = max_displacement(trace_path) if lines else None
    nav = None if events is None else has_event(events, "BallNavActive")
    checks = {"non_empty": lines > 0, "displacement": disp is not None and disp > MIN_DISPLACEMENT}
    if nav is not None:
        checks["ball_nav_active"] = nav
    return {"pass": all(checks.values()), "checks": checks, "lines": lines,
            "max_displacement": None if disp is None else round(disp, 6), "ball_nav_active": nav,
            "trace": str(trace_path) if p is not None else None}


# ------------------------------------------------------------------------------------ framelog

def check_framelog(framelog, frames, events) -> dict:
    """The frame log must equal the input (RecordEnd) or be a non-empty exact prefix of it ending
    at the stop (the Summary frame count equals its length)."""
    fl = tas.load_raw(framelog) if Path(framelog).is_file() else None
    reason = stop_reason(events)
    if fl is None:
        return {"pass": False, "relation": "MISSING", "why": "no frame log", "stop": reason}
    n = len(fl)
    prefix_equal = 0 < n <= len(frames) and tas.frames_to_bytes(fl) == tas.frames_to_bytes(frames[:n])
    full = prefix_equal and n == len(frames)
    summ = summary_of(events)
    if reason == "RecordEnd":
        ok, kind = full, "equal"
    else:
        ok, kind = prefix_equal and summ is not None and summ["frames"] == n, "prefix"
    return {"pass": bool(ok), "relation": kind if ok else "MISMATCH", "framelog_frames": n,
            "input_frames": len(frames), "stop": reason}


# ------------------------------------------------------------------------------------------ G1

def stored_check(stored_trace, ref_trace) -> dict | None:
    """Compare a stored trace-orig with the fresh reference trace.

    The stored run counts as COMPLETE when its terminal frame number equals the reference's
    (frame number only: a divergent last line on the same terminal frame is a complete run that
    differs, never an incomplete one). Only a complete stored run is gated; an incomplete one is
    kept as information."""
    if stored_trace is None or not Path(stored_trace).is_file():
        return None
    cmp = compare(stored_trace, ref_trace)  # a = stored, b = ref
    la, lb = cmp["last_frame_a"], cmp["last_frame_b"]
    complete = la is not None and lb is not None and la == lb
    return {"stored_trace": str(stored_trace), "complete": complete, "gated": complete,
            "pass": cmp["pass"] if complete else None,
            "relation": ("identical" if cmp["pass"] else "DIFFERS") if complete else
            f"not complete: different terminal frame (stored ends at frame {la}, ref at {lb}; "
            f"comparison {cmp['status']}, common prefix {cmp['common_prefix']} lines)",
            "cmp": cmp}


def judge_g1(name: str, ref_trace, bio_trace, bio_events_path, bio_framelog, record_frames,
             stored_trace=None) -> dict:
    """All G1 checks for one record from files on disk. Returns a dict with 'pass' (bool, or None
    for an excluded record), 'verdict' ('PASS' / 'FAIL: ...' / 'excluded: <reason>') and details."""
    events = read_events(bio_events_path)
    out = {
        "sanity_ref": sanity(ref_trace),
        "sanity_bio": sanity(bio_trace, events),
        "cmp_bio_vs_ref": compare(ref_trace, bio_trace),
        "framelog": check_framelog(bio_framelog, record_frames, events),
        "stop": stop_reason(events),
        "expected_stop": G1_EXPECTED_STOP,
        "ball_nav_active": has_event(events, "BallNavActive"),
        "stored": stored_check(stored_trace, ref_trace),
    }
    failed = []
    if not out["sanity_ref"]["pass"]:
        failed.append("ref sanity")
    if not out["sanity_bio"]["pass"]:
        failed.append("bio sanity")
    if not out["cmp_bio_vs_ref"]["pass"]:
        failed.append(f"bio vs ref {out['cmp_bio_vs_ref']['status']}")
    st = out["stored"]
    if st is not None and st["gated"] and not st["pass"]:
        failed.append(f"ref vs stored {st['cmp']['status']}")
    if not out["ball_nav_active"]:
        failed.append("no BallNavActive")
    if out["stop"] != G1_EXPECTED_STOP:
        failed.append(f"stop {out['stop']} != {G1_EXPECTED_STOP}")
    if not out["framelog"]["pass"]:
        failed.append(f"framelog {out['framelog']['relation']}")
    out["failed_checks"] = failed
    out["main_conditions_pass"] = not failed
    if name in G1_EXCLUDED:
        out["pass"] = None
        out["verdict"] = f"excluded: {G1_EXCLUDED[name]}"
    else:
        out["pass"] = not failed
        out["verdict"] = "PASS" if not failed else "FAIL: " + "; ".join(failed)
    return out


# ------------------------------------------------------------------------------------------ G3

def g3_verdict(cmp_load2_vs_load1: dict, cmp_load2_vs_fresh_ref: dict) -> tuple[bool, str]:
    """Same-process reload: exact only if load 2 equals load 1 and the fresh-process reference."""
    ok = bool(cmp_load2_vs_load1["pass"] and cmp_load2_vs_fresh_ref["pass"])
    return ok, ("EQUAL: same-process reload is exact" if ok else G3_FALLBACK)


def g3_verdict_loads(cmp_load1_vs_fresh_ref: dict, cmps_vs_load1: list[dict],
                     cmps_vs_fresh_ref: list[dict]) -> tuple[bool, str]:
    """LOADS=n reload: exact only if load 1 equals the fresh-process reference and every load k > 1
    equals load 1 and the fresh-process reference (one entry per k in each list)."""
    if not cmps_vs_load1 or len(cmps_vs_load1) != len(cmps_vs_fresh_ref):
        raise ValueError("need one load-1 and one fresh-reference comparison per load k > 1")
    ok = bool(cmp_load1_vs_fresh_ref["pass"]) and all(
        g3_verdict(a, b)[0] for a, b in zip(cmps_vs_load1, cmps_vs_fresh_ref))
    return ok, ("EQUAL: same-process reload is exact" if ok else G3_FALLBACK)


# ---------------------------------------------------------------------------- record cleanup

def _same_file(a, b) -> bool:
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def remove_installed(src, dst) -> bool:
    """Remove a record the driver installed into the reference TASRecords (dst), but only when it is
    a copy of a different source file with identical bytes. A record whose source IS dst (the G1
    set lives in TASRecords itself) is never removed."""
    src, dst = Path(src), Path(dst)
    if _same_file(src, dst) or not dst.is_file() or not src.is_file():
        return False
    try:
        # Different spellings of one file (8.3 short names, symlinks, hard links) must never count
        # as "a copy": compare the files themselves, not their path strings.
        if os.path.samefile(src, dst) or _same_file(src.resolve(), dst.resolve()):
            return False
    except OSError:
        return False
    if dst.read_bytes() != src.read_bytes():
        return False
    dst.unlink()
    return True


# ------------------------------------------------------------------------------------- summary

def summarize(cases: dict) -> dict:
    """Per-gate counts from results.json cases (keys 'G1/<name>' ...)."""
    out: dict = {}
    for key, c in sorted(cases.items()):
        gate = key.split("/", 1)[0]
        s = out.setdefault(gate, {"pass": 0, "fail": 0, "excluded": 0, "error": 0, "failed": [], "excluded_cases": []})
        name = key.split("/", 1)[1]
        if "error" in c:
            s["error"] += 1
            s["failed"].append(f"{name}: error {c['error']}")
        elif str(c.get("verdict", "")).startswith("excluded"):
            s["excluded"] += 1
            s["excluded_cases"].append(name)
        elif c.get("pass"):
            s["pass"] += 1
        else:
            s["fail"] += 1
            s["failed"].append(f"{name}: {c.get('verdict', 'FAIL')}")
        if gate == "G1" and "export_supplement" in c:
            e = s.setdefault("export_supplement", {"pass": 0, "fail": 0, "error": 0, "failed": []})
            sup = c["export_supplement"]
            if "error" in sup:
                e["error"] += 1
                e["failed"].append(f"{name}: error {sup['error']}")
            elif sup.get("pass"):
                e["pass"] += 1
            else:
                e["fail"] += 1
                e["failed"].append(f"{name}: {sup.get('cmp', {}).get('status')}")
    if "G3" in out:
        g3 = out["G3"]
        g3["verdict"] = ("error" if g3["error"] else
                         G3_FALLBACK if g3["fail"] else "EQUAL: same-process reload is exact")
    return out

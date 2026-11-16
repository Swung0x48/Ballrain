"""M0 fidelity gates: BallrainIO (train copy) against the real TASSupport (reference copy).

Usage (from the repo root):
    uv run python scripts/m0_gates.py [--gates G0,G1,G2,G3,G4,G5] [--ref-traces DIR] [--fresh]
                                      [--only NAME ...] [--ignore-g0] [--out DIRNAME]

--ref-traces (or the BALLRAIN_REF_TRACES environment variable) is a directory holding stored
reference traces named trace-orig_<record>.txt (read only). Without it, G1 skips that comparison.
--out names the results directory under <runtime> (default m0-results).
Results: <runtime>/<out>/results.json (rewritten after every case, with a per-gate "summary")
and all traces / logs under <runtime>/m0-results/<gate>/<case>/. Re-running resumes: a case already
in results.json for the same installed mod hash and gate-definition version is skipped unless
--fresh is given.

All comparisons are ballrain.trace.compare with require_same_end=True: byte-exact, and both traces
must end on the same terminal frame. Before a mismatch is judged, the reference is re-run three
times on the same record and the trace hashes are recorded. The verdict rules live in
ballrain/gates.py (unit-tested without the game).

G0 sanity (run first; later gates are skipped if it fails, unless --ignore-g0): for a few records,
the reference trace and the BallrainIO trace are each non-empty with a ball displacement above
1 unit, and the BallrainIO events contain BallNavActive (the reference writes no events file).

G1 (per record): PASS only if the BallrainIO trace is identical to the fresh reference trace; the
fresh reference is identical to the stored trace-orig when that stored run is complete (ends on
the same terminal frame); the BallrainIO events contain BallNavActive and stop with LevelFinish;
the frame log is a non-empty prefix of the record ending at the stop; both traces pass the G0
sanity checks. The records in ballrain.gates.G1_EXCLUDED are run and reported as
"excluded: <reason>" (neither pass nor fail). For EVERY G1 record the frame log is also exported
to a legacy .tas, replayed by the real TASSupport and compared with the BallrainIO trace
("export_supplement", reported separately from the G1 verdict).

G3 (same-process reload): if load 2 is not identical to load 1 and to the fresh-process reference,
the verdict is "fallback: one process per episode".

Records the driver installs into the reference copy's TASRecords are removed after each run;
records that live there (the G1 set) are never touched.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from ballrain import gates, instances, synth, tas, trace  # noqa: E402
from ballrain.gates import check_framelog, compare, read_events, stop_reason, summary_of  # noqa: E402

instances.LAUNCH_RECIPE = instances.launch_direct  # Popen, cwd = <copy>/Bin (as Start-Process)

RUNTIME = instances.runtime_dir()
OUT = RUNTIME / "m0-results"
RESULTS = OUT / "results.json"
SYNTH_RECORDS = OUT / "records"
REF = instances.get_copy(instances.REFERENCE_COPY)
TRAIN = instances.get_copy(instances.TRAINING_COPY)
RECORDS = REF.records_dir
BUILD_MOD = REPO / "BallrainIO" / "build" / "BallrainIO.bmodp"
INSTALLED_MOD = TRAIN.mods_dir / "BallrainIO.bmodp"

SR06_FINISH = 13205  # LevelFinish frame of SR06_1_36_163_132 (BallrainIO and reference)
D1_CUT = 40
D1_K = SR06_FINISH - D1_CUT
D2_N, D2_SEED, D2_HOLD = 100000, 51, (5, 30)
G2_EXPECT_LEVELFINISH = {"d1_SR06_finish_tail": D1_K}  # case -> prefix length the stop must come after

G0_RECORDS = ["SR07_2_05_840_264", "SR06_1_36_163_132"]
G1_FIRST = ["SR07_2_05_840_264", "SR06_1_36_163_132", "SR11_2_00_382_132", "HS01_4736_1_32_258"]
REF_REPEATS = 3
TIMEOUT = 180


# ------------------------------------------------------------------------------------ helpers

def sha256(path) -> str | None:
    p = Path(path)
    return instances.sha256_file(p) if p.is_file() else None


def level_of(name: str) -> int:
    m = re.match(r"(?:SR|HS)(\d\d)_", name)
    if not m:
        raise ValueError(f"no level in record name {name}")
    return int(m.group(1))


def ball_names(path) -> list[str]:
    names = []
    with open(path, "rb") as f:
        for line in f:
            n = line.split(b" ", 2)[1].decode("latin-1")
            if not names or names[-1] != n:
                names.append(n)
    return names


class Results:
    def __init__(self, fresh: bool):
        self.data = {"cases": {}}
        if RESULTS.is_file() and not fresh:
            self.data = json.loads(RESULTS.read_text(encoding="utf-8"))
        self.data["meta"] = {
            "updated": None,
            "mod_sha256": sha256(INSTALLED_MOD),
            "build_sha256": sha256(BUILD_MOD),
            "physics_ref": sha256(REF.physics_dll),
            "physics_train": sha256(TRAIN.physics_dll),
            "tassupport_sha256": sha256(REF.mods_dir / "TASSupport.bmodp"),
        }

    def done(self, key) -> bool:
        c = self.data["cases"].get(key)
        return (bool(c) and c.get("mod_sha256") == self.data["meta"]["mod_sha256"]
                and c.get("gate_def") == gates.GATE_DEF_VERSION and "error" not in c)

    def put(self, key, value):
        value["mod_sha256"] = self.data["meta"]["mod_sha256"]
        value["gate_def"] = gates.GATE_DEF_VERSION
        self.data["cases"][key] = value
        self.save()

    def save(self):
        self.data["meta"]["updated"] = dt.datetime.now().isoformat(timespec="seconds")
        self.data["meta"]["gate_def"] = gates.GATE_DEF_VERSION
        # Only cases judged under the current definition and mod count in the summary.
        current = {k: c for k, c in self.data["cases"].items()
                   if c.get("gate_def") == gates.GATE_DEF_VERSION
                   and c.get("mod_sha256") == self.data["meta"]["mod_sha256"]}
        self.data["summary"] = gates.summarize(current)
        self.data["summary"]["stale_cases"] = sorted(set(self.data["cases"]) - set(current))
        OUT.mkdir(parents=True, exist_ok=True)
        tmp = RESULTS.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1, default=str), encoding="utf-8")
        os.replace(tmp, RESULTS)


remove_installed = gates.remove_installed


def set_out(name: str) -> None:
    """Point every output path (results.json, per-gate dirs, synthetic records) at <runtime>/<name>."""
    global OUT, RESULTS, SYNTH_RECORDS
    out = (RUNTIME / name).resolve()
    if out.parent != RUNTIME.resolve() or not out.name.startswith("m0-results"):
        raise SystemExit(f"--out must be a directory name under {RUNTIME} starting with m0-results: {name!r}")
    OUT, RESULTS, SYNTH_RECORDS = out, out / "results.json", out / "records"


def cleanup_stale_installs() -> list[str]:
    """Remove m0_*.tas left in the reference TASRecords by earlier runs that are byte-identical to a
    driver copy in <runtime>/m0-results*/records (any results directory; the copies are only read)."""
    removed = []
    sources = [SYNTH_RECORDS] + [d for d in sorted(RUNTIME.glob("m0-results*/records")) if d != SYNTH_RECORDS]
    for dst in sorted(RECORDS.glob("m0_*.tas")):
        for src_dir in sources:
            if remove_installed(src_dir / dst.name, dst):
                removed.append(dst.name)
                break
    return removed


def run_ref(tas_path, level, trace_path) -> dict:
    src = Path(tas_path).resolve()
    dst = RECORDS / src.name
    try:
        r = instances.run_reference(tas_path, level, trace_path, timeout=TIMEOUT)
    finally:
        removed = remove_installed(src, dst)  # instances.install_record copied src to dst
    res = r["result"]
    return {"trace": str(trace_path), "sha256": sha256(trace_path), "wall_s": round(res.wall_s, 3),
            "exit_code": res.exit_code, "timed_out": res.timed_out, "ok": res.ok,
            "modloader_log": str(r["modloader_log"]) if r["modloader_log"] else None,
            "installed_record_removed": removed}


def run_bio(frames_or_tas, level, out_dir, loads=1) -> dict:
    r = instances.run_ballrain(frames_or_tas, level, out_dir, loads=loads, timeout=TIMEOUT * loads + 60)
    res = r["result"]
    loads_out = []
    for d in r["loads"]:
        ev = read_events(d["events"])
        loads_out.append({k: str(v) for k, v in d.items()} | {
            "trace_sha256": sha256(d["trace"]), "summary": summary_of(ev), "stop": stop_reason(ev),
            "event_list": [f"{i} {n}" for i, n in ev if not n.startswith("Summary")]})
    log = out_dir / "modloader.log"
    if TRAIN.modloader_log.is_file() and TRAIN.modloader_log.stat().st_mtime >= res.started_at - 2:
        log.write_bytes(TRAIN.modloader_log.read_bytes())
    return {"frames": str(r["frames"]), "wall_s": round(res.wall_s, 3), "exit_code": res.exit_code,
            "timed_out": res.timed_out, "missing": res.missing_outputs, "ok": res.ok, "loads": loads_out}


def ref_repeats(tas_path, level, out_dir, first_sha) -> dict:
    """The reference's own repeat rate on this record (run before judging a mismatch)."""
    shas = [first_sha]
    for k in range(1, REF_REPEATS + 1):
        shas.append(run_ref(tas_path, level, out_dir / f"ref_repeat{k}.txt")["sha256"])
    c = Counter(shas)
    return {"sha256": shas, "distinct": len(c), "all_equal": len(c) == 1}


def record_frames(name) -> np.ndarray:
    return tas.load(RECORDS / f"{name}.tas")


def synth_record(frames) -> Path:
    """A legacy .tas with a content-hashed name, for replay by the real TASSupport."""
    h = hashlib.sha256(tas.frames_to_bytes(frames)).hexdigest()[:16]
    path = SYNTH_RECORDS / f"m0_{h}.tas"
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        tas.save(path, frames)
    return path


# -------------------------------------------------------------------------------------- gates

def g0(R: Results, only) -> bool:
    """Sanity block, run first: each trace non-empty, ball displacement > 1 unit, BallNavActive."""
    ok = True
    for name in G0_RECORDS:
        key = f"G0/{name}"
        if (only and name not in only) or (not only and R.done(key)):
            ok = ok and bool(R.data["cases"].get(key, {}).get("pass", True))
            continue
        level = level_of(name)
        d = OUT / "G0" / name
        d.mkdir(parents=True, exist_ok=True)
        tas_path = RECORDS / f"{name}.tas"
        case = {"gate": "G0", "record": name, "level": level}
        try:
            case["ref"] = run_ref(tas_path, level, d / "ref_trace.txt")
            case["bio"] = run_bio(tas_path, level, d / "bio")
            b = case["bio"]["loads"][0]
            case["sanity_ref"] = gates.sanity(d / "ref_trace.txt")
            case["sanity_bio"] = gates.sanity(b["trace"], read_events(b["events"]))
            case["pass"] = case["sanity_ref"]["pass"] and case["sanity_bio"]["pass"]
            case["verdict"] = "PASS" if case["pass"] else "FAIL: " + ", ".join(
                f"{side} {k}" for side in ("ref", "bio") for k, v in case[f"sanity_{side}"]["checks"].items() if not v)
        except Exception as e:
            case["error"] = f"{type(e).__name__}: {e}"
            case["pass"] = False
            case["verdict"] = f"error: {case['error']}"
        R.put(key, case)
        ok = ok and case["pass"]
        print(key, case["verdict"], flush=True)
    return ok


def g1_plan(ref_traces: Path | None):
    """(names in run order, {name: stored trace path}, all local SR/HS/other records)."""
    covered = sorted(p.name[len("trace-orig_"):-4] for p in ref_traces.glob("trace-orig_*.txt")) if ref_traces else []
    covered = [n for n in covered if (RECORDS / f"{n}.tas").is_file() and re.match(r"(SR|HS)\d\d_.+", n)]
    names = [n for n in G1_FIRST if n in covered or not ref_traces] + [n for n in covered if n not in G1_FIRST]
    all_records = sorted(p.stem for p in RECORDS.glob("*.tas") if not p.stem.startswith("m0_"))
    # Stored traces named by the short level only (trace-orig_SR09.txt): used when exactly one local record
    # has that prefix and no full-name trace; the comparison itself shows whether it belongs to that record.
    stored = {n: ref_traces / f"trace-orig_{n}.txt" for n in names} if ref_traces else {}
    for short in sorted({n[:4] for n in all_records if re.match(r"(SR|HS)\d\d_", n)}):
        cands = [n for n in all_records if n.startswith(short + "_")]
        f = ref_traces / f"trace-orig_{short}.txt" if ref_traces else None
        if f and f.is_file() and len(cands) == 1 and cands[0] not in names:
            names.append(cands[0])
            stored[cands[0]] = f
    names = sorted(names, key=lambda n: (level_of(n) == 1, names.index(n)))  # level 1 last
    return names, stored, all_records


def export_supplement(level, b, d) -> dict:
    """The frames BallrainIO actually played (its frame log), exported as a legacy .tas, replayed by the
    real TASSupport in the reference copy and compared byte-exactly with the BallrainIO trace."""
    sup = {}
    try:
        fl = tas.load_raw(b["framelog"])
        rec = synth_record(fl)
        sup["exported_tas"] = str(rec)
        sup["exported_frames"] = len(fl)
        sup["exported_roundtrip"] = tas.frames_to_bytes(tas.load(rec)) == tas.frames_to_bytes(fl)
        sup["ref"] = run_ref(rec, level, d / "ref_framelog_trace.txt")
        sup["cmp"] = compare(d / "ref_framelog_trace.txt", b["trace"])
        sup["pass"] = bool(sup["cmp"]["pass"] and sup["exported_roundtrip"])
    except Exception as e:  # never masks the main verdict
        sup["error"] = f"{type(e).__name__}: {e}"
        sup["pass"] = False
    return sup


def g1(R: Results, ref_traces: Path | None, only):
    names, stored, all_records = g1_plan(ref_traces)
    R.data["g1_not_covered"] = [n for n in all_records if n not in names]
    R.data["g1_excluded"] = dict(gates.G1_EXCLUDED)
    R.save()
    for name in names:
        key = f"G1/{name}"
        if (only and name not in only) or (not only and R.done(key)):
            continue
        level = level_of(name)
        d = OUT / "G1" / name
        d.mkdir(parents=True, exist_ok=True)
        tas_path = RECORDS / f"{name}.tas"
        case = {"gate": "G1", "record": name, "level": level}
        try:
            case["ref"] = run_ref(tas_path, level, d / "ref_trace.txt")
            case["bio"] = run_bio(tas_path, level, d / "bio")
            b = case["bio"]["loads"][0]
            orig = stored.get(name)
            case["stored_trace"] = str(orig) if orig else None
            case.update(gates.judge_g1(name, d / "ref_trace.txt", b["trace"], b["events"], b["framelog"],
                                       record_frames(name), orig))
            if orig and orig.is_file():
                case["cmp_bio_vs_stored"] = compare(orig, b["trace"])  # information
            if not case["cmp_bio_vs_ref"]["pass"]:
                case["ref_repeat"] = ref_repeats(tas_path, level, d, case["ref"]["sha256"])
        except Exception as e:  # recorded, the batch goes on
            case["error"] = f"{type(e).__name__}: {e}"
            case["pass"] = False
            case["verdict"] = f"error: {case['error']}"
        if "bio" in case and case["bio"]["loads"]:
            case["export_supplement"] = export_supplement(level, case["bio"]["loads"][0], d)
        R.put(key, case)
        sup = case.get("export_supplement", {})
        print(key, case["verdict"], "| export:", "PASS" if sup.get("pass") else
              f"FAIL {sup.get('error') or sup.get('cmp', {}).get('status')}", flush=True)


def g2_cases():
    n = 40000
    sr06 = record_frames("SR06_1_36_163_132")
    sr11 = record_frames("SR11_2_00_382_132")
    sr03 = record_frames("SR03_1_36_884_264")
    # Transformer frames from the stored/fresh reference traces: SR06_..._132 turns paper at frame 1915,
    # SR11_..._132 at 2841, SR03_..._264 at 4619 (first ball-name change).
    return {
        "a1_L1_random": (1, lambda: synth.idle_then_random(n, 11, idle=300, hold_min=30, hold_max=240, q_frames=(140,))),
        "a2_L2_random": (2, lambda: synth.idle_then_random(n, 12, idle=300, hold_min=30, hold_max=240)),
        "a3_L6_random": (6, lambda: synth.idle_then_random(n, 16, idle=300, hold_min=30, hold_max=240)),
        "b1_SR06_k2100": (6, lambda: synth.off_route(sr06, 2100, n, 21, hold_min=30, hold_max=240)),
        "b2_SR11_k2700": (11, lambda: synth.off_route(sr11, 2700, n, 22, hold_min=30, hold_max=240)),
        "b3_SR03_k4500": (3, lambda: synth.off_route(sr03, 4500, n, 23, hold_min=30, hold_max=240)),
        # Short continuations: the record runs out mid-level with the ball alive (RecordEnd stop).
        "c1_SR07_k5000_n300": (7, lambda: synth.off_route(record_frames("SR07_2_05_840_264"), 5000, 300, 31)),
        "c2_SR11_k1500_n200": (11, lambda: synth.off_route(sr11, 1500, 200, 32)),
        # Round 2. d1: SR06_..._132 reaches LevelFinish at frame 13205; the prefix is cut D1_CUT frames
        # before it and a short random tail follows, which must still end in LevelFinish after the cut.
        "d1_SR06_finish_tail": (6, lambda: synth.off_route(sr06, D1_K, 400, 41, hold_min=5, hold_max=30)),
        # d2: long level-1 random run with the tutorial-ending Q press (short holds keep the ball alive longer).
        "d2_L1_long_random": (1, lambda: synth.idle_then_random(D2_N, D2_SEED, idle=300, hold_min=D2_HOLD[0],
                                                                hold_max=D2_HOLD[1], q_frames=(140,))),
        # d3: as d2 with 1..4-frame holds (near zero-mean random walk, chosen once, not seed-searched).
        "d3_L1_long_jitter": (1, lambda: synth.idle_then_random(D2_N, 52, idle=300, hold_min=1, hold_max=4,
                                                                q_frames=(140,))),
    }


def g2(R: Results, only):
    cases = g2_cases()
    order = sorted(cases, key=lambda c: (cases[c][0] == 1, c))
    for name in order:
        key = f"G2/{name}"
        if (only and name not in only) or (not only and R.done(key)):
            continue
        level, make = cases[name]
        d = OUT / "G2" / name
        d.mkdir(parents=True, exist_ok=True)
        case = {"gate": "G2", "case": name, "level": level}
        try:
            frames = make()
            inp = tas.save_raw(d / "input.bin", frames)
            case["input_frames"] = len(frames)
            case["bio"] = run_bio(inp, level, d / "bio")
            b = case["bio"]["loads"][0]
            ev = read_events(b["events"])
            case["framelog"] = check_framelog(b["framelog"], frames, ev)
            names = [n for _, n in ev]
            case["deaths"] = names.count("BallOff")
            case["game_over"] = "GameOver" in names
            case["ball_nav_active"] = "BallNavActive" in names
            case["ball_names"] = ball_names(b["trace"]) if Path(b["trace"]).is_file() else []
            fl = tas.load_raw(b["framelog"])  # the FRAME LOG, not the input, is exported
            rec = synth_record(fl)
            case["exported_tas"] = str(rec)
            case["exported_roundtrip"] = tas.frames_to_bytes(tas.load(rec)) == tas.frames_to_bytes(fl)
            case["ref"] = run_ref(rec, level, d / "ref_trace.txt")
            case["cmp_ref_vs_bio"] = compare(d / "ref_trace.txt", b["trace"])
            if not case["cmp_ref_vs_bio"]["pass"]:
                case["ref_repeat"] = ref_repeats(rec, level, d, case["ref"]["sha256"])
            case["sanity_bio"] = gates.sanity(b["trace"], ev)  # information (G0 checks on this trace)
            case["pass"] = (case["cmp_ref_vs_bio"]["pass"] and case["framelog"]["pass"]
                            and case["exported_roundtrip"] and case["ball_nav_active"])
            if name in G2_EXPECT_LEVELFINISH:  # the case is only meaningful if the tail ran into the finish
                played = len(fl)
                case["construction"] = {"stop": b["stop"], "played_frames": played,
                                        "prefix_frames": G2_EXPECT_LEVELFINISH[name],
                                        "ok": b["stop"] == "LevelFinish" and played > G2_EXPECT_LEVELFINISH[name]}
                case["pass"] = case["pass"] and case["construction"]["ok"]
            case["verdict"] = "PASS" if case["pass"] else "FAIL"
        except Exception as e:
            case["error"] = f"{type(e).__name__}: {e}"
            case["pass"] = False
            case["verdict"] = f"error: {case['error']}"
        R.put(key, case)
        print(key, case["verdict"], flush=True)


G3_LOADS = (2, 3)


def g3(R: Results, only):
    cases = {
        "SR06_1_36_163_132": (6, RECORDS / "SR06_1_36_163_132.tas", OUT / "G1" / "SR06_1_36_163_132" / "ref_trace.txt"),
        "b1_SR06_k2100": (6, OUT / "G2" / "b1_SR06_k2100" / "input.bin", OUT / "G2" / "b1_SR06_k2100" / "ref_trace.txt"),
        "SR01_1_14_457_264": (1, RECORDS / "SR01_1_14_457_264.tas", OUT / "G1" / "SR01_1_14_457_264" / "ref_trace.txt"),
    }
    for name, (level, src, fresh_ref) in cases.items():
        for loads in G3_LOADS:
            key = f"G3/{name}/loads{loads}"
            if (only and name not in only) or (not only and R.done(key)):
                continue
            d = OUT / "G3" / name / f"loads{loads}"
            d.mkdir(parents=True, exist_ok=True)
            case = {"gate": "G3", "case": name, "loads": loads, "level": level, "fresh_process_ref": str(fresh_ref)}
            try:
                if not Path(src).is_file() or not Path(fresh_ref).is_file():
                    raise FileNotFoundError(f"run G1/G2 first: {src} / {fresh_ref}")
                case["bio"] = run_bio(src, level, d / "bio", loads=loads)
                ls = case["bio"]["loads"]
                l1 = ls[0]
                case["cmp_load1_vs_fresh_ref"] = compare(fresh_ref, l1["trace"])
                vs1, vsf = [], []
                for k, lk in enumerate(ls[1:], 2):
                    case[f"cmp_load{k}_vs_load1"] = compare(l1["trace"], lk["trace"])
                    case[f"cmp_load{k}_vs_fresh_ref"] = compare(fresh_ref, lk["trace"])
                    vs1.append(case[f"cmp_load{k}_vs_load1"])
                    vsf.append(case[f"cmp_load{k}_vs_fresh_ref"])
                if loads >= 3:
                    case["cmp_load3_vs_load2"] = compare(ls[1]["trace"], ls[2]["trace"])  # information
                # An inexact reload is a verdict (fall back to one process per episode), not an error;
                # a driver error or a missing prerequisite stays an error.
                case["pass"], case["verdict"] = gates.g3_verdict_loads(case["cmp_load1_vs_fresh_ref"], vs1, vsf)
            except Exception as e:
                case["error"] = f"{type(e).__name__}: {e}"
                case["pass"] = False
                case["verdict"] = f"error: {case['error']}"
            R.put(key, case)
            print(key, case["verdict"], flush=True)


_TS = re.compile(r"^\[(\d\d)/(\d\d)/(\d{4}) (\d\d):(\d\d):(\d\d)\.(\d{3})\]")


def log_time(line: str):
    m = _TS.match(line)
    if not m:
        return None
    mo, da, yr, h, mi, s, ms = map(int, m.groups())
    return dt.datetime(yr, mo, da, h, mi, s, ms * 1000)


def ref_log_span(log_path) -> float | None:
    """ms from 'On Message PreLoadLevel' to 'TAS playing stopped.' in a reference ModLoader.log."""
    t0 = t1 = None
    for line in Path(log_path).read_text(encoding="latin-1").splitlines():
        if t0 is None and "On Message PreLoadLevel" in line:
            t0 = log_time(line)
        elif t0 is not None and "TAS playing stopped" in line:
            t1 = log_time(line)
    return (t1 - t0).total_seconds() * 1000 if t0 and t1 else None


def g4(R: Results, only):
    for name in ("SR07_2_05_840_264", "SR12_2_12_823_240"):
        key = f"G4/{name}"
        if (only and name not in only) or (not only and R.done(key)):
            continue
        level = level_of(name)
        d = OUT / "G4" / name
        d.mkdir(parents=True, exist_ok=True)
        case = {"gate": "G4", "record": name, "level": level}
        try:
            bio = run_bio(RECORDS / f"{name}.tas", level, d / "bio")
            s = bio["loads"][0]["summary"]
            case["bio"] = {"frames": s["frames"], "wall_ms": s["wall_ms"],
                           "fps": round(s["frames"] / (s["wall_ms"] / 1000), 1) if s["wall_ms"] else None,
                           "process_wall_s": bio["wall_s"], "stop": bio["loads"][0]["stop"]}
            ref = run_ref(RECORDS / f"{name}.tas", level, d / "ref_trace.txt")
            span = ref_log_span(ref["modloader_log"]) if ref["modloader_log"] else None
            # The reference writes no Summary: its frame count is the frame index at the stop (the last
            # trace frame + 1 at LevelFinish, as BallrainIO counts) and its span comes from the
            # millisecond log timestamps PreLoadLevel -> 'TAS playing stopped.'.
            last = trace.compare(d / "ref_trace.txt", d / "ref_trace.txt").last_frame_a
            frames = last + 1 if last is not None else None
            case["ref"] = {"frames": frames, "log_span_ms": span,
                           "fps": round(frames / (span / 1000), 1) if frames and span else None,
                           "process_wall_s": ref["wall_s"]}
            case["cmp_bio_vs_ref"] = compare(d / "ref_trace.txt", bio["loads"][0]["trace"])
            case["pass"] = True  # throughput is measured, not gated
            case["verdict"] = "measured"
        except Exception as e:
            case["error"] = f"{type(e).__name__}: {e}"
            case["pass"] = False
            case["verdict"] = f"error: {case['error']}"
        R.put(key, case)
        print(key, json.dumps({k: case.get(k) for k in ("bio", "ref")}), flush=True)


def g5(R: Results, only, runs=10):
    name = "SR06_1_36_163_132"
    key = f"G5/{name}"
    if (only and name not in only) or (not only and R.done(key)):
        return
    d = OUT / "G5" / name
    d.mkdir(parents=True, exist_ok=True)
    case = {"gate": "G5", "record": name, "level": 6, "runs": runs}
    try:
        tas_path = RECORDS / f"{name}.tas"
        bio, ref = [], []
        for k in range(1, runs + 1):  # interleaved, one process per copy at a time
            r = run_bio(tas_path, 6, d / f"bio{k}")
            bio.append(r["loads"][0]["trace_sha256"])
            ref.append(run_ref(tas_path, 6, d / f"ref{k}.txt")["sha256"])
        for side, shas in (("bio", bio), ("ref", ref)):
            c = Counter(shas)
            major, cnt = c.most_common(1)[0]
            case[side] = {"sha256": shas, "majority": major, "majority_count": cnt,
                          "deviating": runs - cnt, "distinct": len(c)}
        case["same_majority"] = case["bio"]["majority"] == case["ref"]["majority"]
        i = bio.index(case["bio"]["majority"])
        j = ref.index(case["ref"]["majority"])
        case["cmp_majority"] = compare(d / f"ref{j + 1}.txt", d / f"bio{i + 1}" / "trace.txt")
        case["pass"] = case["same_majority"]
        case["verdict"] = "PASS" if case["pass"] else "FAIL"
    except Exception as e:
        case["error"] = f"{type(e).__name__}: {e}"
        case["pass"] = False
        case["verdict"] = f"error: {case['error']}"
    R.put(key, case)
    print(key, json.dumps({k: case.get(k) for k in ("bio", "ref", "same_majority")}), flush=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gates", default="G0,G1,G2,G3,G4,G5")
    p.add_argument("--ref-traces", default=os.environ.get("BALLRAIN_REF_TRACES"))
    p.add_argument("--fresh", action="store_true", help="ignore earlier results")
    p.add_argument("--only", nargs="*", help="restrict to these record / case names")
    p.add_argument("--ignore-g0", action="store_true", help="run later gates even if G0 sanity fails")
    p.add_argument("--out", default="m0-results", help="results directory name under the runtime directory")
    args = p.parse_args(argv)
    set_out(args.out)
    if sha256(INSTALLED_MOD) != sha256(BUILD_MOD):
        print(f"installed mod {INSTALLED_MOD} differs from the build {BUILD_MOD}; install the build first")
        return 1
    ref_traces = Path(args.ref_traces) if args.ref_traces else None
    removed = cleanup_stale_installs()
    if removed:
        print(f"removed {len(removed)} stale installed records from {RECORDS}: {removed}")
    R = Results(args.fresh)
    R.save()
    selected = [g.strip().upper() for g in args.gates.split(",")]
    only = set(args.only or [])
    if "G0" in selected and not g0(R, only) and not args.ignore_g0:
        print("G0 sanity failed: later gates not run (use --ignore-g0 to run them anyway)")
        return 2
    if "G1" in selected:
        g1(R, ref_traces, only)
    if "G2" in selected:
        g2(R, only)
    if "G3" in selected:
        g3(R, only)
    if "G4" in selected:
        g4(R, only)
    if "G5" in selected:
        g5(R, only)
    print(json.dumps({g: {k: v for k, v in s.items() if k not in ("failed", "excluded_cases")}
                      for g, s in R.data["summary"].items() if isinstance(s, dict)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

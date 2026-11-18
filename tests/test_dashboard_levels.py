"""Dashboard per level: a run's level / TAS record from its config, the level's TAS line, checkpoints, SR clock and
thresholds, the trace loader for any ball, and a fake level-2 run next to a level-1 run (temp dirs; no game, no
real dashboard output). Needs the tas-v1 dataset records of levels 1 and 2 (skipped without them)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from ballrain.instances import REPO_ROOT

DASH = REPO_ROOT / "tools" / "dashboard"
sys.path.insert(0, str(DASH))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import data as D  # noqa: E402
import geometry as G  # noqa: E402

L1, L2 = "SR01_1_14_457_264", "SR02_1_28_992_264"
need = pytest.mark.skipif(not (D.record_ok(L1, 1) and D.record_ok(L2, 2)), reason="tas-v1 records of levels 1, 2 missing")


def _run(base: Path, name: str, cfg: dict | None) -> D.Run:
    p = base / name
    p.mkdir(parents=True, exist_ok=True)
    if cfg is not None:
        (p / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return D.Run(name, p, True)


def test_level_records_mirror_training():
    import train_m2
    assert D.LEVEL_RECORDS == G.LEVEL_RECORDS == train_m2.RECORDS


@need
def test_run_level(tmp_path):
    assert D.run_level(_run(tmp_path, "r3", None)) == {"level": 1, "record": L1}
    assert D.run_level(_run(tmp_path, "r7", {"path_dir": "runtime/refpaths-alt"})) == {"level": 1, "record": L1}
    assert D.run_level(_run(tmp_path, "r8", {"level": 2, "record": L2})) == {"level": 2, "record": L2}
    assert D.run_level(_run(tmp_path, "r9", {"level": 2, "record": None})) == {"level": 2, "record": L2}
    # mirrored config: anything that is not a dataset record of that level falls back to the level's default
    for bad in ("../../etc", "SR01_1_14_457_264", "SR02_9_99_999_999", "SR02_1_28_992_264/../x"):
        assert D.run_level(_run(tmp_path, "r10", {"level": 2, "record": bad}))["record"] == L2
    assert D.run_level(_run(tmp_path, "r11", {"level": 7}))["record"] is None  # no default record for level 7
    assert D.run_level(_run(tmp_path, "r12", {"level": "x"}))["level"] == 1


def test_benchmarks_per_level():
    assert D.benchmarks(1) == {"tas": 74446, "human_wr": 75853, "featured": 83499, "threshold": 92999}
    assert D.benchmarks(2) == {"tas": 88977, "human_wr": 96382, "featured": 143499, "threshold": 158499}


@need
def test_tas_clock():
    c1, c2 = D.tas_clock(L1), D.tas_clock(L2)
    assert (c1.total_ms, c1.end_frame, c2.total_ms, c2.end_frame) == (74446, 20587, 88977, 24394)
    z = np.load(D.DATASET / L1 / "starts.npz")
    i = int(np.searchsorted(z["frames_done"], 5000))
    assert abs(float(c1.ms(5000)) - float(z["sr_double"][i])) < 1e-6  # the record's own timer, not a linear fit
    assert float(c1.ms(100)) == 0.0
    assert c1.bins[-2:] == [18000, 20588] and c1.bins[0] == 0
    assert c2.bins[-2:] == [22000, 24395]


def test_trace_loader_any_ball(tmp_path):
    p = tmp_path / "trace.txt"
    m = "1 0 0 0 1 0 0 0 1"
    p.write_text(f"5 Ball_Wood {m} 1 2 3\n5 Box_01 {m} 9 9 9\n6 Ball_Paper {m} 4 5 6\n7 Ball_Stone {m} 7 8 9\n",
                 encoding="utf-8")
    fr, P = D.Traces._load(p)
    assert fr.tolist() == [5, 6, 7] and P[:, 0].tolist() == [1.0, 4.0, 7.0]


@need
def test_level2_geometry_and_line(tmp_path):
    L = G.Level(tmp_path, log=lambda *a: None, record=L2)
    assert (L.level, L.n, L.first) == (2, 24395, 907)
    assert L.checkpoints == [5915, 10291, 14521, 17561]
    assert L.default_frontier == 24393 - 1500
    assert all(len(L.outlines[k]) for k in ("walk", "steep", "rail"))
    assert not np.isnan(L.pos[L.ok]).any() and L.pos[L.ok][:, 0].max() > 1500  # the paper-ball part is on the line
    assert (tmp_path / f"line_{L2}_v{G.LINE_VERSION}.npz").is_file()
    L1_ = G.Level(tmp_path, log=lambda *a: None, record=L1)  # level 1: the old feature cache or the same replay
    assert (L1_.n, L1_.first, L1_.checkpoints, L1_.default_frontier) == (20588, 936, [7817, 11756, 14232], 19086)


def _csv(path: Path, rows):
    cols = list(rows[0])
    path.write_text("\n".join([",".join(cols)] + [",".join(str(r[c]) for c in cols) for r in rows]) + "\n",
                    encoding="utf-8")


@need
def test_fake_level2_run_next_to_level1(tmp_path, monkeypatch):
    import report
    monkeypatch.setattr(D, "MAC_DIR", tmp_path / "mac")
    monkeypatch.setattr(D, "PC_DIR", tmp_path / "pc")
    r7 = _run(tmp_path / "mac", "r7", {"path_dir": "runtime/refpaths-alt", "split_extra": [[4764, 6018, 2.86]]})
    r8 = _run(tmp_path / "mac", "r8", {"level": 2, "record": L2, "path_dir": "runtime/refpaths-alt"})
    for run, k0, rec in ((r7, 15000, L1), (r8, 20000, L2)):
        clk = D.tas_clock(rec)
        rows = []
        for i in range(30):
            k = k0 + 50 * i
            tk = float(clk.ms(k))
            rows.append({"update": 100 + i, "t": 1000 + i, "copy": "hl-01", "run_id": f"x{i}", "end_reason": "LevelFinish",
                         "sr_ms": int(tk + (clk.total_ms - tk) * 1.1), "start_frame": k, "deaths": 0, "invalid": ""})
        _csv(run.path / "runs.csv", rows)
        _csv(run.path / "updates.csv", [{"update": 129, "steps": 10 ** 6, "frontier": k0, "band_success": 0.5}])
        (run.path / "train.log").write_text(
            f"2026-11-17 10:00:00 curriculum: frontier -> {k0} (sector 3) after rate 0.60\n", encoding="utf-8")
    runs = D.discover_runs()
    assert [r.name for r in runs] == ["r7", "r8"]
    s8, s7 = D.sr_estimate(runs, "r8"), D.sr_estimate(runs, "r7")
    assert s8["runs_used"] == ["r8"] and s8["level"] == 2 and s8["benchmarks"]["threshold"] == 158499
    assert not s8["alt"] and s8["shortcut_extra_s"] == 0.0  # no L02 line under refpaths-alt: the TAS line
    assert s8["ratio_range"][0] == pytest.approx(1.1, abs=0.002) and s8["bins"][-1]["hi"] == 24394
    assert s7["alt"] == (D.ROOT / "runtime" / "refpaths-alt" / "L01.npz").is_file()
    assert s7["benchmarks"]["tas"] == 74446 and s7["ratio_range"][0] == pytest.approx(1.1, abs=0.002)
    L = G.Level(tmp_path / "cache", log=lambda *a: None, record=L2)
    st = D.run_status(r8, 0.0, L, {"train": "Mac mini", "eval": "PC", "sources": {}})
    assert st["level"] == 2 and st["covered_pct"] == round(100 * (24395 - 20000) / (24395 - 907), 1)
    F = D.frontier_payload(L, runs, 0.0, "r8", 20000, show_alt=True)
    assert F["level"] == 2 and [h["run"] for h in F["history"]] == ["r8"] and F["alt"] is None
    assert [c[2] for c in F["checkpoints"]] == [5915, 10291, 14521, 17561] and F["end"][2] == 24394
    C = D.curves_payload(runs, {"train": "Mac mini", "eval": "PC"})
    assert C["r7"]["level"] == 1 and C["r8"]["level"] == 2
    for payload in (s8, st, F, C, L.outlines):
        assert not report.check(json.dumps(payload, ensure_ascii=False))
        assert str(tmp_path) not in json.dumps(payload) and "SR02_" not in json.dumps(payload)


@need
def test_highlight_uses_the_runs_level(tmp_path):
    import highlight as H
    r8 = _run(tmp_path, "r8", {"level": 2, "record": L2})
    rows = [{"update": 100 + i, "t": 4000 + i, "copy": "hl-01", "run_id": f"L02-sr-hl-01-20261117-120000-1-{i}",
             "end_reason": "LevelFinish", "sr_ms": 95000 + i, "start_frame": k, "frames": 24000, "lives": 3, "deaths": 0,
             "framelog_match": True, "handover_mismatches": 0, "invalid": "", "dir": ""}
            for i, k in enumerate((24200, 22100, 23000))]
    _csv(r8.path / "runs.csv", rows)
    got, summ = H.candidates(r8)
    # one bin up to the level-2 LevelFinish (24394): all three in it, by SR
    assert summ["bin"] == [22000, 24395] and [int(r["start_frame"]) for r in got] == [24200, 22100, 23000]
    assert H.tas_file(L2).name == f"{L2}.tas"
    with pytest.raises(ValueError):
        H.tas_file("../x")
    pub = H.public_meta(r8, {"update": 7, "start_frame": 22100, "copy": "hl-01"}, {"started_at": 1794890000.0},
                        {"sr_ms": 95000}, {"matched": True}, 100.0, {})
    assert pub["level"] == 2 and pub["tas_ms"] == 88977

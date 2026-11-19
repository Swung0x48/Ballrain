"""Dashboard route bundle: reference lines from a run's own checkpoint (branch_assets), route classes of traced lives
(recorded branch member, else where the life went), and the window since the last route-guidance change. Synthetic
lines and checkpoints in temp dirs; no game, no dataset, no real dashboard output."""
from __future__ import annotations

import io
import json
import os
import sys
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ballrain.instances import REPO_ROOT  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "tools" / "dashboard"))
import data as D  # noqa: E402
import report  # noqa: E402

PIN_A, PIN_B, PIN_C = "a" * 64, "b" * 64, "c" * 64


def _line(detours=()):
    """A straight main line along x (0..200, 0.5 apart); each detour (x0, x1, dz) leaves it to z = dz between x0, x1."""
    pts = []
    x = 0.0
    while x <= 200.0:
        for x0, x1, dz in detours:
            if x0 <= x <= x1:
                if not pts or pts[-1][2] == 0.0:  # go out
                    pts += [[x, 0.0, z] for z in np.arange(0.5, dz, 0.5 if dz > 0 else -0.5)]
                pts.append([x, 0.0, dz])
                if x + 0.5 > x1:  # come back
                    pts += [[x, 0.0, z] for z in np.arange(dz, 0.0, -0.5 if dz > 0 else 0.5)]
                break
        else:
            pts.append([x, 0.0, 0.0])
        x += 0.5
    return np.asarray(pts, np.float64)


def _npz(**kw) -> bytes:
    b = io.BytesIO()
    np.savez(b, **kw)
    return b.getvalue()


def _member(level, P, drive=b"drive"):
    return {"level": level, "line_npz": _npz(points=P), "desc_json": json.dumps({"level": level}).encode(),
            "drive_npz": drive, "sha256": {}}


def _entry(mid, ordinary=False):
    e = {"id": mid, "phases": [{"when": {"ball": 0, "sector": 3}, "local_s": [0.0, 1.0]}], "merge": {"ball": 0, "sector": 3}}
    if ordinary:
        e["entry_context"] = "ordinary"
    return e


def _bundle(with_east=True, drive=b"drive"):
    entries = [_entry("south-loop")] + ([_entry("east-primary", True)] if with_east else [])
    members = [_member(2, _line([(150, 170, -30.0)]))] + ([_member(2, _line([(80, 120, 30.0)]), drive)] if with_east else [])
    return {"version": 2, "level": 2, "manifest": json.dumps({"version": 2, "branches": entries}).encode(), "members": members}


def _ckpt(path, pin, bundle, t):
    torch.save({"model": {}, "args": {}, "n_obs": 1, "n_act": 9, "branch_assets": bundle,
                "env_config": {"features": {"branch_sha256": pin}}}, path)
    os.utime(path, (t, t))


@pytest.fixture
def run(tmp_path, monkeypatch):
    main = tmp_path / "L02.npz"
    np.savez(main, points=_line())
    monkeypatch.setattr(D, "ref_line", lambda run: main)
    r = tmp_path / "mac" / "r9"
    r.mkdir(parents=True)
    (r / "config.json").write_text(json.dumps({"level": 2}), encoding="utf-8")
    _ckpt(r / "ckpt_000100.pt", PIN_A, _bundle(with_east=False), 1000)
    _ckpt(r / "ckpt_000200.pt", PIN_B, _bundle(), 2000)
    _ckpt(r / "latest.pt", PIN_C, _bundle(drive=b"drive+moving"), 3000)  # only the drive data differs from B
    for k in ("_CK_MEMO", "_ROUTE_MEMO", "_PIN_MEMO", "_LOG_MEMO"):
        monkeypatch.setattr(D, k, {})
    return D.Run("r9", r, True)


def _life(P):
    p3 = D._resample(P).astype(np.float32)
    w3 = np.r_[np.linalg.norm(np.diff(p3, axis=0), axis=1), 0.0].astype(np.float32)
    return {"p3": p3, "w3": w3}


def test_routes_from_checkpoint(run):
    R = D.run_routes(run)
    assert R["pin"] == PIN_C
    assert [(f["id"], f["name"], f["guidance"]) for f in R["families"]] == [("east-primary", "东台线", True),
                                                                             ("south-loop", "南侧绕行线", False)]
    keys = [c["key"] for c in D.route_classes(R)]
    assert keys == ["east-primary", "south-loop", "tas", "shared"]
    assert D.route_classes(R)[2]["name"] == "TAS 线（鬼躲纸变）"
    main = _line()
    L = types.SimpleNamespace(ok=np.arange(len(main)), pos=main, level=2)
    pub = D.routes_public(R, L, cur=0)
    txt = json.dumps(pub, ensure_ascii=False)
    assert not report.check(txt) and PIN_C not in txt and "a" * 16 not in txt
    assert pub["tas_name"] == "TAS 参考线（鬼躲纸变）"
    over = np.vstack([np.asarray(s) for s in pub["tas_over"]])
    assert ((over[:, 0] > 75) & (over[:, 0] < 175)).all()  # only the stretches the branches override
    east = pub["families"][0]["members"][0]
    assert east["guidance"] and all(abs(p[1] - 30.0) < 3.5 or 75 < p[0] < 125 for s in east["segs"] for p in s)
    # the covered stretch takes the guidance's detour, not the TAS stretch it overrides
    cov = np.vstack([np.asarray(s) for s in pub["covered"]])
    assert (cov[:, 1] > 29).any() and not ((cov[:, 0] > 90) & (cov[:, 0] < 110) & (np.abs(cov[:, 1]) < 1)).any()


def test_classify_lives(run):
    R = D.run_routes(run)
    east = _line([(80, 120, 30.0)])
    assert D.classify_life(_life(east[east[:, 0] > 60]), R) == ("east-primary", "spatial")
    tas = _line()
    assert D.classify_life(_life(tas[(tas[:, 0] > 60) & (tas[:, 0] < 140)]), R) == ("tas", "spatial")
    assert D.classify_life(_life(tas[tas[:, 0] < 50]), R) == ("shared", "spatial")
    south = _line([(150, 170, -30.0)])
    assert D.classify_life(_life(south[south[:, 0] > 130]), R) == ("south-loop", "spatial")
    # the life's recorded branch member (a rest start placed on it) wins over where it went
    assert D.classify_life(_life(tas[tas[:, 0] < 50]), R, {"branch_id": "south-loop"}) == ("south-loop", "recorded")
    # the guidance's fork decides first, even when the life also crossed another fork
    both = _line([(80, 120, 30.0), (150, 170, -30.0)])
    assert D.classify_life(_life(both[both[:, 0] > 60]), R)[0] == "east-primary"


def test_guidance_window_skips_moving_only_update(run):
    (run.path / "branch-updates.jsonl").write_text(
        json.dumps({"from": PIN_A, "to": PIN_B}) + "\n" + json.dumps({"from": PIN_B, "to": PIN_C}) + "\n", encoding="utf-8")
    t0 = 1_800_000_000
    idx = [{"t": t0 + 10, "pin": PIN_A}, {"t": t0 + 100, "pin": PIN_A}, {"t": t0 + 300, "pin": PIN_B},
           {"t": t0 + 900, "pin": PIN_C}]
    fmt = lambda t: D.dt.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S")  # noqa: E731
    (run.path / "train.log").write_text(f"{fmt(t0)} resumed from x at update 1\n{fmt(t0 + 200)} resumed from x at update 2\n"
                                        f"{fmt(t0 + 800)} resumed from x at update 3\n", encoding="utf-8")
    g = D.guidance_since(run, idx, 0.0)
    # B -> C only added moving tables: the window starts at the A -> B switch (the resume between the lives)
    assert g["ts"] == t0 + 200 and g["pins"] == {PIN_B, PIN_C}
    # without the jsonl the pins of the traced lives give the same sequence
    (run.path / "branch-updates.jsonl").unlink()
    assert D.guidance_since(run, idx, 0.0)["ts"] == t0 + 200


def test_run_without_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "_CK_MEMO", {})
    r = tmp_path / "r7"
    r.mkdir()
    torch.save({"model": {}, "args": {}, "env_config": {"features": {}}}, r / "latest.pt")
    run = D.Run("r7", r, True)
    assert D.run_routes(run) is None and D.run_pin(run) is None
    assert D.guidance_since(run, [], 0.0) is None
    info = {"key": "k", "id": "00000001", "run": "r7", "end": "finish", "backend": "x", "t": 0, "sr": None}
    tr = {"segs": [np.zeros((3, 2), np.float32)], "k": 1, "end_frame": 2, "start": (0.0, 0.0), "deaths": []}
    assert "route" not in D.traj_payload(info, tr)

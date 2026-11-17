"""精彩回放: selection, capture stream parsing, the dashboard's /media route (Range) and metadata check.
No game and no real dashboard data: everything runs on temp dirs."""
from __future__ import annotations

import http.client
import json
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

DASH = Path(__file__).resolve().parents[1] / "tools" / "dashboard"
sys.path.insert(0, str(DASH))

import data as D  # noqa: E402
import highlight as H  # noqa: E402
import server as SV  # noqa: E402

from ballrain import capture  # noqa: E402


def test_parse_range():
    assert SV.parse_range(None, 100) is None
    assert SV.parse_range("bytes=0-", 100) == (0, 99)
    assert SV.parse_range("bytes=0-1", 100) == (0, 1)
    assert SV.parse_range("bytes=90-200", 100) == (90, 99)
    assert SV.parse_range("bytes=-10", 100) == (90, 99)
    assert SV.parse_range("bytes=-500", 100) == (0, 99)
    assert SV.parse_range("bytes=100-", 100) == "bad"  # valid but unsatisfiable: 416
    assert SV.parse_range("bytes=-0", 100) == "bad"
    # RFC 9110: an invalid spec, an unknown unit or a multi-range is ignored (whole file)
    assert SV.parse_range("bytes=5-2", 100) is None
    assert SV.parse_range("bytes=0-1,5-6", 100) is None
    assert SV.parse_range("items=0-1", 100) is None
    assert SV.parse_range("bytes=-", 100) is None
    assert SV.parse_range("bytes=abc", 100) is None
    assert SV.parse_range("bytes=" + "9" * 5000 + "-", 100) is None  # no int() of thousands of digits
    assert SV.parse_range("bytes=-" + "9" * 5000, 100) is None


@pytest.fixture
def media_server(tmp_path, monkeypatch):
    monkeypatch.setattr(SV, "MEDIA", tmp_path)
    body = bytes(range(256)) * 40
    (tmp_path / "highlight.mp4").write_bytes(body)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), SV.make_handler(dash=None))
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield srv.server_address[1], body, tmp_path
    srv.shutdown()
    srv.server_close()


def _get(port, path, headers=None, method="GET"):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request(method, path, headers=headers or {})
    r = c.getresponse()
    return r.status, dict(r.getheaders()), r.read()


def test_media_range(media_server):
    port, body, d = media_server
    st, h, b = _get(port, "/media/highlight.mp4")
    assert st == 200 and b == body and h["Accept-Ranges"] == "bytes" and h["Content-Type"] == "video/mp4"
    st, h, b = _get(port, "/media/highlight.mp4?v=3", {"Range": "bytes=0-1"})
    assert st == 206 and b == body[:2] and h["Content-Range"] == f"bytes 0-1/{len(body)}"
    assert "max-age" in h["Cache-Control"]
    st, h, b = _get(port, "/media/highlight.mp4", {"Range": "bytes=100-"})
    assert st == 206 and b == body[100:]
    st, h, b = _get(port, "/media/highlight.mp4", {"Range": f"bytes={len(body)}-"})
    assert st == 416 and h["Content-Range"] == f"bytes */{len(body)}"
    st, h, b = _get(port, "/media/highlight.mp4", method="HEAD")
    assert st == 200 and b == b"" and int(h["Content-Length"]) == len(body)
    # whitelist only: other names, traversal and the json are not served here
    for p in ("/media/highlight.json", "/media/../highlight.mp4", "/media/other.mp4", "/media/highlight-overlay.mp4"):
        assert _get(port, p)[0] == 404
    # the file can be replaced while the server has it cached (no open handle), and the new bytes are served
    new = b"x" * 1000
    tmp = d / "new.tmp"
    tmp.write_bytes(new)
    import os
    os.replace(tmp, d / "highlight.mp4")
    st, h, b = _get(port, "/media/highlight.mp4")
    assert st == 200 and b == new


def test_highlight_public(tmp_path, monkeypatch):
    monkeypatch.setattr(SV, "MEDIA", tmp_path)
    assert SV.highlight_public() is None
    meta = {"run": "r6", "update": 25375, "episode_at": "2026-11-17 00:50", "recorded_at": "2026-11-17 01:02",
            "tz": "UTC+8", "sr_ms": 104886, "k": 4703, "tas_ms": 74446, "backend": "无头 x87-soft", "matched": True,
            "duration_s": 112.3, "v": 1, "_id": "abcd", "extra": "D:\\secret", "files": {"highlight.mp4": 1}}
    (tmp_path / "highlight.json").write_text(json.dumps(meta), encoding="utf-8")
    assert SV.highlight_public() is None  # no video yet
    (tmp_path / "highlight.mp4").write_bytes(b"0")
    p = SV.highlight_public()
    assert p["sr_ms"] == 104886 and p["k"] == 4703 and p["matched"] is True and p["overlay"] is None
    assert "_id" not in p and "extra" not in p
    (tmp_path / "highlight-overlay.mp4").write_bytes(b"0")
    assert SV.highlight_public()["overlay"] is None  # a leftover overlay file is not this recording's
    meta["files"] = {"highlight.mp4": 1, "highlight-overlay.mp4": 1}  # first-version json: fixed names
    (tmp_path / "highlight.json").write_text(json.dumps(meta), encoding="utf-8")
    p = SV.highlight_public()
    assert p["video"] == "highlight.mp4" and p["overlay"] == "highlight-overlay.mp4"
    # versioned set: names come from the json, must match the pattern and exist
    for n in ("highlight-1794893327.mp4", "highlight-overlay-1794893327.mp4"):
        (tmp_path / n).write_bytes(b"0")
    meta.update(video="highlight-1794893327.mp4", overlay="highlight-overlay-1794893327.mp4")
    (tmp_path / "highlight.json").write_text(json.dumps(meta), encoding="utf-8")
    p = SV.highlight_public()
    assert p["video"] == "highlight-1794893327.mp4" and p["overlay"] == "highlight-overlay-1794893327.mp4"
    meta.update(video="../secret.mp4")
    (tmp_path / "highlight.json").write_text(json.dumps(meta), encoding="utf-8")
    assert SV.highlight_public() is None
    meta.update(video="highlight-1794893327.mp4")
    meta["backend"] = "train-03"  # a copy name never reaches the page
    (tmp_path / "highlight.json").write_text(json.dumps(meta), encoding="utf-8")
    assert SV.highlight_public() is None


def _runs_csv(path: Path, rows):
    cols = ["update", "t", "copy", "run_id", "end_reason", "sr_ms", "start_frame", "frames", "lives", "deaths",
            "framelog_match", "handover_mismatches", "invalid", "dir"]
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join(str(r.get(c, "")) for c in cols))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _row(i, t, k, sr, end="LevelFinish", deaths=0, copy="hl-01", invalid=""):
    rid = f"L01-sr-{copy}-20261117-1200{i:02d}-1-{i}"
    return {"update": 100 + i, "t": t, "copy": copy, "run_id": rid, "end_reason": end, "sr_ms": sr if end ==
            "LevelFinish" else "", "start_frame": k, "frames": 20000, "lives": 1, "deaths": deaths,
            "framelog_match": True, "handover_mismatches": 0, "invalid": invalid,
            "dir": f"/Users/x/ballrain/repo/runtime/m2-train/r9/episodes/{copy}/{rid}"}


def test_candidates(tmp_path):
    rows = [
        _row(0, 100.0, 4500, 90000),            # outside the hour (t < 5000 - 3600)
        _row(1, 4000.0, 4800, 106000),          # first bin, slower
        _row(2, 4100.0, 4700, 104000),          # first bin, fastest
        _row(3, 4200.0, 4600, 0, end="GameOver"),  # sets the bin, did not finish
        _row(4, 4300.0, 19000, 77000),          # faster but late takeover: after the first bin
        _row(5, 4400.0, 4900, 99000, deaths=1),  # died: excluded
        _row(6, 4500.0, 4950, 98000, invalid="camera_off_quarter"),  # invalid: excluded
        _row(7, 5000.0, 8000, 95000),
    ]
    _runs_csv(tmp_path / "runs.csv", rows)
    run = D.Run("r9", tmp_path, True)
    got, summ = H.candidates(run)
    # the bin comes from the clean finishes; after it, earlier takeovers first (bin), then SR
    assert [int(r["start_frame"]) for r in got] == [4700, 4800, 8000, 19000]
    assert [int(r["update"]) for r in got] == [102, 101, 107, 104]
    assert summ["bin"] == [4000, 6000] and summ["in_bin"] == 2 and summ["finished_no_death"] == 4
    assert got[0]["in_first_bin"] and not got[2]["in_first_bin"]
    # a GameOver with an earlier start does not move the bin away from the finishes
    _runs_csv(tmp_path / "runs.csv", rows + [_row(8, 4800.0, 900, 0, end="GameOver")])
    got, summ = H.candidates(run)
    assert summ["bin"] == [4000, 6000] and [int(r["update"]) for r in got][:2] == [102, 101]


def test_remote_dir_check():
    good = "/Users/x/ballrain/repo/runtime/m2-train/r6/episodes/hl-05/L01-sr-hl-05-20261117-125007-39001-431"
    assert H.REMOTE_DIR_RE.match(good)
    for bad in (good + "; rm -rf ~", good.replace("/hl-05/", "/../"), "relative/" + good, good + "'"):
        assert not H.REMOTE_DIR_RE.match(bad) or ".." in bad


def test_capture_frame_bgr0():
    w, h, pitch = 3, 2, 16  # padded rows
    px = bytearray(h * pitch)
    for y in range(h):
        for x in range(w):
            px[y * pitch + 4 * x: y * pitch + 4 * x + 4] = bytes([x, y, 7, 0])
    f = capture.Frame(5, 2, w, h, pitch, 32, (0xFF0000, 0xFF00, 0xFF), 33.3, bytes(px))
    a = f.bgr0()
    assert a.shape == (2, 3, 4) and a[1, 2, 0] == 2 and a[1, 2, 1] == 1 and a[0, 0, 2] == 7
    with pytest.raises(ValueError):
        capture.Frame(5, 1, w, h, pitch, 16, (0xF800, 0x7E0, 0x1F), 0.0, bytes(px)).bgr0()
    assert capture.HEADER.size == 48


def test_publish_swaps_complete_sets(tmp_path):
    media, work = tmp_path / "media", tmp_path / "work"
    work.mkdir()
    sets = []
    for v in (1794890001, 1794890002, 1794890003):
        a, b = work / f"a{v}.mp4", work / f"b{v}.mp4"
        a.write_bytes(b"plain")
        b.write_bytes(b"overlay")
        sets.append(H.publish(media, {"run": "r6", "v": v}, a, b))
    j = json.loads((media / "highlight.json").read_text(encoding="utf-8"))
    assert j["video"] == "highlight-1794890003.mp4" and j["overlay"] == "highlight-overlay-1794890003.mp4"
    names = sorted(p.name for p in media.iterdir())
    # the current set and the one before (viewers still playing it) stay; older sets are removed
    assert names == ["highlight-1794890002.mp4", "highlight-1794890003.mp4", "highlight-overlay-1794890002.mp4",
                     "highlight-overlay-1794890003.mp4", "highlight.json"]


def test_media_cache_evicts(tmp_path, monkeypatch):
    monkeypatch.setattr(SV, "MEDIA", tmp_path)
    c = SV.MediaCache()
    names = [f"highlight-{1794890000 + i}.mp4" for i in range(6)]
    for n in names:
        (tmp_path / n).write_bytes(n.encode())
        assert c.get(n) == n.encode()
    assert len(c.files) == SV.MEDIA_CACHE_N and names[-1] in c.files and names[0] not in c.files
    (tmp_path / names[-1]).unlink()
    assert c.get(names[-1]) is None and names[-1] not in c.files  # a deleted file leaves the cache
    (tmp_path / names[-2]).unlink()
    (tmp_path / "highlight-1794899999.mp4").write_bytes(b"new")
    c.get("highlight-1794899999.mp4")
    assert names[-2] not in c.files  # pruned on insert


def test_legacy_names_retired(tmp_path, monkeypatch):
    monkeypatch.setattr(SV, "MEDIA", tmp_path)
    for n in ("highlight.mp4", "highlight-overlay.mp4", "highlight-1794893327.mp4"):
        (tmp_path / n).write_bytes(b"0")
    (tmp_path / "highlight.json").write_text(json.dumps({"files": {"highlight.mp4": 1}}), encoding="utf-8")
    assert SV.media_allowed("highlight.mp4")  # a first-version json still uses them
    (tmp_path / "highlight.json").write_text(json.dumps({"video": "highlight-1794893327.mp4"}), encoding="utf-8")
    assert not SV.media_allowed("highlight.mp4") and not SV.media_allowed("highlight-overlay.mp4")
    assert SV.MediaCache().get("highlight.mp4") is None
    assert SV.media_allowed("highlight-1794893327.mp4")


def test_media_versioned_route(media_server):
    port, body, d = media_server
    (d / "highlight-overlay-1794893327.mp4").write_bytes(b"ov")
    assert _get(port, "/media/highlight-overlay-1794893327.mp4")[2] == b"ov"
    assert _get(port, "/media/highlight-overlay-12.mp4")[0] == 404


def test_capture_complete(tmp_path):
    fps = H.FPS
    rows, vf = [], 0
    for g in range(1, 1201, 2):  # 7.5 ms frames
        ms = g * 7.5
        due = int(ms * fps / 1000.0) + 1 - vf
        if due > 0:
            rows.append([vf, due, g, ms])
            vf += due
    cap = {"ffmpeg_exit": 0, "server_error": None, "launches": 1, "rows": rows, "video_frames": vf, "size": [1024, 768]}
    (tmp_path / "capture.json").write_text(json.dumps(cap), encoding="utf-8")
    res = {"levelfinish_frame": 1000, "post_frames": 190}
    assert H.capture_complete(tmp_path, res) == (True, "")
    assert not H.capture_complete(tmp_path, {"levelfinish_frame": 1100, "post_frames": 190})[0]  # stops too early
    for bad in ({"launches": 2}, {"ffmpeg_exit": 1}, {"server_error": "x"}, {"video_frames": vf + 5}, {"size": [1280, 720]},
                {"rows": rows[3:]}):
        (tmp_path / "capture.json").write_text(json.dumps(dict(cap, **bad)), encoding="utf-8")
        assert not H.capture_complete(tmp_path, res)[0], bad


@pytest.mark.skipif(sys.platform != "win32", reason="named pipes")
def test_capture_server_drops_the_pipe_when_the_sink_fails():
    """A failing sink must not leave the game blocked on a full pipe: the server disconnects it."""
    import struct

    def sink(f):
        raise ValueError("sink broke")
    srv = capture.CaptureServer(sink, buf=1 << 16)
    w, h = 64, 32
    rec = struct.pack("<4s9Id", b"BRVF", 1, 1, w, h, w * 4, 32, 0xFF0000, 0xFF00, 0xFF, 7.5) + bytes(w * h * 4)
    failed = False
    with open(srv.path, "wb", buffering=0) as fh:
        fh.write(rec)
        assert srv.join(10) and "sink broke" in (srv.error or "")
        try:
            for _ in range(200):  # far more than the pipe buffer: without the disconnect this would block
                fh.write(rec)
        except OSError:
            failed = True
    srv.close()
    assert failed


def test_sr_per_frame():
    import overlay
    from ballrain import tas
    from ballrain.protocol import Ev
    fr = tas.make_frames([10.0] * 6, [0] * 6)
    # StartLevel during frame 0, CounterActive during frame 1, CounterInactive during frame 7 (a tail frame)
    ev = [(1, Ev.StartLevel), (2, Ev.CounterActive), (8, Ev.CounterInactive)]
    sr = overlay.sr_per_frame(fr, 5.0, ev, 12)
    assert sr.tolist()[:8] == [0, 10, 20, 30, 40, 50, 55, 55]


@pytest.mark.skipif(not Path(r"C:\Windows\Fonts\msyh.ttc").is_file(), reason="needs the Microsoft YaHei font")
def test_panel_draws():
    import overlay
    from ballrain import tas
    n, k = 40, 10
    keys = [0] * k + [tas.KEY_UP, tas.KEY_UP | tas.KEY_RIGHT] * ((n - k) // 2)
    fr = tas.make_frames([7.5] * n, keys)
    cam = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]
    disp = {"sr": 100.0, "sector": 1, "cps": 0, "ball": 0, "valid": True, "speed": 3.2, "fan": [8.0, 1.0] * 4,
            "heights": [0.0] * 49, "hit": [1] * 49, "yaw": 0.3, "cam": cam}
    tas_rec = [dict(disp, f=f) for f in range(0, k, 2)]
    steps = [dict(disp, f=f, a=6, probs=[1 / 9] * 9, v=1.5, r=-0.01, terms={"time": -0.015, "progress": 0.005},
                  ret=-0.1, step=i + 1) for i, f in enumerate(range(k, n, 2))]
    panel = {"k": k, "agent_start": k, "tas": tas_rec, "steps": steps, "sr": list(np.arange(n + 10) * 7.5)}
    pn = overlay.Panel(panel, fr, {"run": "r6", "update": 1})
    for g in (0, 5, k, k + 1, n, n + 5):
        im = pn.draw(g)
        assert im.size == (overlay.PANEL_W, overlay.H)


def test_public_meta_is_public_safe():
    import report
    run = D.Run("r6", Path("."), True)
    row = {"update": 25375, "start_frame": 4703, "copy": "hl-05"}
    pub = H.public_meta(run, row, {"started_at": 1794890000.0}, {"sr_ms": 104886}, {"matched": True}, 112.0,
                        {"highlight.mp4": 123})
    assert not report.check(json.dumps(pub, ensure_ascii=False))
    assert pub["backend"] == "x87-soft headless" and pub["k"] == 4703
    assert np.isfinite(pub["duration_s"])

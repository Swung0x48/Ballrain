"""Audio-session muting logic (ballrain.audio) with fake sessions and processes; one read-only Core Audio query."""
import sys

import pytest

from ballrain import audio

win = pytest.mark.skipif(sys.platform != "win32", reason="Windows only")


def S(pid, muted, state=1):
    return audio.SessionInfo(pid, muted, state)


def test_audio_enabled_switch(monkeypatch):
    monkeypatch.delenv(audio.AUDIO_VAR, raising=False)
    assert audio.audio_enabled() is False
    for v, want in (("1", True), ("yes", True), ("0", False), ("", False), (" ", False)):
        monkeypatch.setenv(audio.AUDIO_VAR, v)
        assert audio.audio_enabled() is want
    assert audio.audio_enabled(False) is False and audio.audio_enabled(True) is True


def test_sessions_to_change_matches_only_tracked_pids():
    sessions = [S(10, False), S(11, False), S(10, True, 2), S(0, False), S(12, True), S(13, False)]
    # 10 and 12 tracked muted, 13 tracked unmuted (opt-out), 11 is someone else's, 0 is the system-sounds session
    assert audio.sessions_to_change(sessions, {10: True, 12: True, 13: False}) == [0]
    assert audio.sessions_to_change(sessions, {12: False}) == [4]
    assert audio.sessions_to_change(sessions, {0: True}) == []
    assert audio.sessions_to_change(sessions, {}) == []


class FakeProc:
    def __init__(self, pid, alive=True):
        self.pid, self.alive = pid, alive

    def poll(self):
        return None if self.alive else 0


def test_watcher_round(monkeypatch):
    calls = []
    found = {"sessions": []}

    def fake_apply(tracked):
        calls.append(dict(tracked))
        return found["sessions"], []

    monkeypatch.setattr(audio, "apply_mute", fake_apply)
    monkeypatch.setattr(audio, "_image_matches", lambda pid, exe: pid != 666)  # 666: image is not the copy's exe
    w = audio._Watcher()
    a, b, other = FakeProc(100), FakeProc(101), FakeProc(666)
    for p, m in ((a, True), (b, False), (other, True)):
        w.procs[p.pid] = audio._Tracked(p, "exe", m, audio.time.monotonic())
    assert w._round() == audio.POLL_FAST_S  # no session seen yet: poll fast
    assert calls[-1] == {100: True, 101: False}  # never a PID whose image does not match
    found["sessions"] = [S(100, True), S(101, False)]
    assert w._round() == audio.POLL_SLOW_S  # every tracked process has its session: slow re-checks
    a.alive = False
    w._round()
    assert 100 not in w.procs and calls[-1] == {101: False}  # exited: dropped (and its handle released)
    b.alive = other.alive = False
    assert w._round() is None and not w.procs and w.thread is None


def test_watcher_survives_errors(monkeypatch):
    def boom(tracked):
        raise OSError("audio service gone")

    monkeypatch.setattr(audio, "apply_mute", boom)
    monkeypatch.setattr(audio, "_image_matches", lambda pid, exe: True)
    w = audio._Watcher()
    w.procs[5] = audio._Tracked(FakeProc(5), "exe", True, audio.time.monotonic())
    assert w._round() == audio.POLL_FAST_S and w.errors == 1 and "audio service gone" in w.last_error


@win
def test_list_sessions_read_only():
    sessions = audio.list_sessions(peak=True)  # read-only snapshot of this machine's sessions
    assert all(isinstance(s.pid, int) and s.state in (0, 1, 2) for s in sessions)
    assert all(s.peak is None or s.peak >= 0 for s in sessions)
    assert audio.list_sessions([0xFFFFFFFE]) == []

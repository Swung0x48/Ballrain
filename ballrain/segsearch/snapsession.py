"""AgentSession with save / load through ProcessSnapshot (ballrain.segsearch.procsnap): the engine's memory and this
side's session state (frames_done, tracker, frame log, the channel's sequence numbers) go back together.

The socket's kernel buffers are not in the snapshot, so save and load happen only when the channel is quiet: every action
sent has its observation back (the engine then waits in its recv; in normal play it sends nothing else, the agent's
STATE message goes out only on an error) and nothing from the engine is unread. After a load the next observation must
carry the saved observation sequence number + 1 (the engine's counter went back with its memory); anything else ends
the session with an error. For segment search only (not for training or frozen tests)."""
from __future__ import annotations

import copy
import select

from ..client import AgentSession, SessionError
from .procsnap import Image, ProcessSnapshot, SnapshotError

_SESSION_FIELDS = ("obs", "frames_done", "ended", "end_reason", "final_time", "sren", "sr_at_levelfinish", "trace_hash",
                   "n_logged", "_level_loaded", "_inc_pos", "n_prefix_obs", "prefix_frames")


class SnapSession(AgentSession):
    """start() as AgentSession; then save() -> Image and load(Image)."""

    raw_hook = None  # callable(bytes) for every observation's raw bytes (tests)

    def start(self):
        obs = super().start()
        self.snap = ProcessSnapshot(self.handle.proc)
        self._acked = self.chan.py_seq  # the last action whose observation came back
        self._expect_mod_seq = None
        wait = self.chan.wait_obs

        def wrapped(*a, **k):
            data = wait(*a, **k)
            if data is not None and self._expect_mod_seq is not None:
                got, want = self.chan.last_mod_seq, self._expect_mod_seq
                self._expect_mod_seq = None
                if got != want:
                    raise SessionError(f"observation sequence {got} after a load, expected {want}")
            if data is not None and self.raw_hook is not None:
                self.raw_hook(data)
            return data
        self.chan.wait_obs = wrapped
        return obs

    def step(self, action):
        obs = super().step(action)
        self._acked = self.chan.py_seq
        return obs

    def _quiet(self, what: str) -> None:
        if self.ended:
            raise SnapshotError(f"{what}: the session has ended")
        if self.chan.py_seq != self._acked:
            raise SnapshotError(f"{what}: action {self.chan.py_seq} sent without its observation back")
        r, _, _ = select.select([self.chan.conn], [], [], 0)
        if r:
            raise SnapshotError(f"{what}: unread bytes from the engine on the socket")

    def _state(self) -> dict:
        st = {k: copy.copy(getattr(self, k)) for k in _SESSION_FIELDS}
        st["log_chunks"] = list(self.log_chunks)
        st["tracker"] = copy.deepcopy(self.tracker)
        st["chan"] = {"py_seq": self.chan.py_seq, "last_mod_seq": self.chan.last_mod_seq, "_hdr": dict(self.chan._hdr)}
        return st

    def save(self) -> Image:
        self._quiet("save")
        img = self.snap.save()
        img.extra = self._state()
        return img

    def load(self, img: Image) -> dict:
        self._quiet("load")
        stats = self.snap.load(img)
        st = img.extra
        for k in _SESSION_FIELDS:
            setattr(self, k, copy.copy(st[k]))
        self.log_chunks = list(st["log_chunks"])
        self.tracker = copy.deepcopy(st["tracker"])
        self.chan.py_seq = self._acked = st["chan"]["py_seq"]
        self.chan.last_mod_seq = st["chan"]["last_mod_seq"]
        self.chan._hdr = dict(st["chan"]["_hdr"])
        self._expect_mod_seq = st["chan"]["last_mod_seq"] + 1
        return stats

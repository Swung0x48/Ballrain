"""AgentSession with save / load through ProcessSnapshot (ballrain.segsearch.procsnap): the engine's memory and this
side's session state (frames_done, tracker, frame log, the channel's sequence numbers) go back together."""
from __future__ import annotations

import copy

from ..client import AgentSession
from .procsnap import Image, ProcessSnapshot

_SESSION_FIELDS = ("obs", "frames_done", "ended", "end_reason", "final_time", "sren", "sr_at_levelfinish", "trace_hash",
                   "n_logged", "_level_loaded", "_inc_pos", "n_prefix_obs", "prefix_frames")


class SnapSession(AgentSession):
    """start() as AgentSession; then save() -> Image and load(Image). The engine must be waiting for an action (right
    after start() or a step), and must not have ended."""

    raw_hook = None  # callable(bytes) for every observation's raw bytes (tests)

    def start(self):
        obs = super().start()
        self.snap = ProcessSnapshot(self.handle.proc)
        if self.raw_hook is not None:
            wait = self.chan.wait_obs

            def wrapped(*a, **k):
                data = wait(*a, **k)
                if data is not None:
                    self.raw_hook(data)
                return data
            self.chan.wait_obs = wrapped
        return obs

    def _state(self) -> dict:
        st = {k: copy.copy(getattr(self, k)) for k in _SESSION_FIELDS}
        st["log_chunks"] = list(self.log_chunks)
        st["tracker"] = copy.deepcopy(self.tracker)
        st["chan"] = {"py_seq": self.chan.py_seq, "last_mod_seq": self.chan.last_mod_seq, "_hdr": dict(self.chan._hdr)}
        return st

    def save(self) -> Image:
        if self.ended:
            raise RuntimeError("session has ended")
        img = self.snap.save()
        img.extra = self._state()
        return img

    def load(self, img: Image) -> dict:
        stats = self.snap.load(img)
        st = img.extra
        for k in _SESSION_FIELDS:
            setattr(self, k, copy.copy(st[k]))
        self.log_chunks = list(st["log_chunks"])
        self.tracker = copy.deepcopy(st["tracker"])
        self.chan.py_seq = st["chan"]["py_seq"]
        self.chan.last_mod_seq = st["chan"]["last_mod_seq"]
        self.chan._hdr = dict(st["chan"]["_hdr"])
        return stats

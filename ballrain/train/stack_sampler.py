"""In-process sampling profiler for the learner (train_m2 --profile-sample SECONDS): a daemon thread looks at every
thread's current Python stack every `interval` seconds (sys._current_frames) and counts, per thread name, the
innermost function (self time) and every function on the stack (inclusive time). Every `dump_every` seconds and at
the end it writes the counts to `out` (JSON: samples per thread, top functions by self and inclusive samples). No root
needed (unlike attaching py-spy on macOS); the cost is one stack walk per thread per interval.

Only Python frames are seen: time inside C code (numpy, torch, pickle) is charged to the Python function that called
it, which is what we want for "which step of the loop takes the time". A thread blocked waiting (queue get, socket
recv) shows up in the function that waits."""
from __future__ import annotations

import json
import sys
import threading
import time
from collections import Counter
from pathlib import Path


class StackSampler:
    def __init__(self, out: Path, interval: float = 0.01, duration: float = 300.0, dump_every: float = 60.0,
                 depth: int = 40):
        self.out, self.interval, self.duration, self.dump_every, self.depth = Path(out), interval, duration, dump_every, depth
        self.self_c: dict = {}
        self.incl_c: dict = {}
        self.n: Counter = Counter()
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._run, name="stack-sampler", daemon=True)

    def start(self):
        self._th.start()
        return self

    def stop(self):
        self._stop.set()

    @staticmethod
    def _key(f):
        c = f.f_code
        return f"{Path(c.co_filename).name}:{c.co_name}:{c.co_firstlineno}"

    def _run(self):
        me = threading.get_ident()
        names = {}
        t0 = last = time.monotonic()
        while not self._stop.is_set() and time.monotonic() - t0 < self.duration:
            for t in threading.enumerate():
                names[t.ident] = t.name
            for tid, frame in sys._current_frames().items():
                if tid == me:
                    continue
                name = names.get(tid, str(tid))
                if name.startswith(("remote-conn", "remote-send")):
                    name = name.split("-")[0] + "-" + name.split("-")[1]  # one bucket for all connection threads
                self.n[name] += 1
                sc = self.self_c.setdefault(name, Counter())
                ic = self.incl_c.setdefault(name, Counter())
                sc[self._key(frame)] += 1
                seen, f, d = set(), frame, 0
                while f is not None and d < self.depth:
                    k = self._key(f)
                    if k not in seen:
                        ic[k] += 1
                        seen.add(k)
                    f, d = f.f_back, d + 1
            if time.monotonic() - last >= self.dump_every:
                self.dump()
                last = time.monotonic()
            time.sleep(self.interval)
        self.dump()

    def dump(self):
        res = {"interval_s": self.interval, "threads": {}}
        for name, n in self.n.most_common():
            res["threads"][name] = {"samples": n,
                                    "self": self.self_c[name].most_common(25),
                                    "inclusive": self.incl_c[name].most_common(40)}
        tmp = self.out.with_suffix(".tmp")
        tmp.write_text(json.dumps(res, indent=1), encoding="utf-8")
        tmp.replace(self.out)

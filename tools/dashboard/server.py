"""Live training-progress page for the Ballance RL runs (Chinese UI, interactive charts).

    uv run python tools/dashboard/server.py [--host 0.0.0.0] [--port 8088] [--tensorboard-url URL]

Serves static/index.html, /api/status (small JSON, polled by the page every 60 s) and
/api/data/<name>.json (map / curve payloads, gzip). A generator child process (this script with --generator, started
and watched by the server) recomputes the payloads every --interval seconds (default 10 min; the page's "立即刷新" asks
for an earlier pass, rate-limited) into runtime/dashboard/ (untracked); the server only reads its results
(summary_last.json, generator_state.json), so a pass of several minutes never holds the server's interpreter and the
page's requests answer at once. The child exits when the server goes away (its stdin closes).
Each run's level and TAS record come from its config (level 1 before r8); the maps
of a level (geometry_l<N>, frontier_l<N>, history_l<N>) are built from that level's TAS record, once, and cached. The process lowers its
own CPU priority, only reads the training outputs, and never starts or touches the game or the trainer.
See tools/dashboard/README.md for what the page publishes and what it leaves out.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import datetime as dt  # noqa: E402
import gzip  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from urllib.parse import urlparse  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import assets as A  # noqa: E402
import data as D  # noqa: E402
import report as R  # noqa: E402

ROOT = D.ROOT
if str(ROOT) not in sys.path:  # ballrain.features for the model panel (data.model_info)
    sys.path.append(str(ROOT))
OUT = ROOT / "runtime" / "dashboard"
CACHE = OUT / "cache"
ASSETS = OUT / "assets"
STATIC = HERE / "static"
STATE = "generator_state.json"  # in OUT: the generator child's state (pass running / done, map frontier, level sizes)
SCRIPT = "tools/dashboard/server.py"  # the child's command line names the script as the server's does (ops match it)
RESPAWN_S = 60  # a generator child that exited is started again after this long
STATUS_SNAPSHOT = "status_last.json"  # in OUT: the last /api/status answer, served by a restarted server at once
STATUS_TTL_S = 20
ATTEMPTS_N = 80
HISTORY_N = 150
CTYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
          ".css": "text/css; charset=utf-8", ".json": "application/json; charset=utf-8",
          ".png": "image/png", ".jpg": "image/jpeg", ".mp4": "video/mp4"}
MEDIA = OUT / "media"  # written by tools/dashboard/highlight.py
# the only names /media/ serves: one set per recording, versioned (highlight[-overlay]-<v>.mp4, named in highlight.json)
# or the fixed names of the first version
MEDIA_NAME_RE = re.compile(r"^highlight(?:-overlay)?(?:-\d{9,12})?\.mp4$")
LEGACY_NAMES = ("highlight.mp4", "highlight-overlay.mp4")
MEDIA_MAX = 64 << 20
MEDIA_CACHE_N = 4  # two sets: the current one and the one before (viewers still playing it)


def _media_json() -> dict:
    try:
        j = json.loads((MEDIA / "highlight.json").read_text(encoding="utf-8"))
        return j if isinstance(j, dict) else {}
    except (OSError, ValueError):
        return {}


def media_allowed(name: str) -> bool:
    """A versioned name, or a fixed name of the first version only while highlight.json is still a first-version json
    (no "video" key): once the job has published a versioned set, the old fixed names are never served again."""
    if not MEDIA_NAME_RE.match(name):
        return False
    if name in LEGACY_NAMES:
        return "video" not in _media_json()
    return True


class MediaCache:
    """The highlight videos, held in memory per (size, mtime): requests never keep the file open (on Windows an
    open file blocks the job's os.replace), and Range requests are cut from the cached bytes. Holds at most
    MEDIA_CACHE_N files; an entry whose file is gone or changed is dropped."""

    def __init__(self):
        self.lock = threading.Lock()
        self.files: dict = {}  # name -> (sig, bytes), oldest first

    def _prune(self):
        for n in list(self.files):
            try:
                st = (MEDIA / n).stat()
                ok = (st.st_size, st.st_mtime_ns) == self.files[n][0]
            except OSError:
                ok = False
            if not ok:
                del self.files[n]
        while len(self.files) > MEDIA_CACHE_N:
            del self.files[next(iter(self.files))]

    def get(self, name: str) -> bytes | None:
        if not media_allowed(name):
            return None
        p = MEDIA / name
        try:
            st = p.stat()
        except OSError:
            with self.lock:
                self.files.pop(name, None)
            return None
        sig = (st.st_size, st.st_mtime_ns)
        with self.lock:
            hit = self.files.get(name)
            if hit and hit[0] == sig:
                self.files[name] = self.files.pop(name)  # most recently used last
                return hit[1]
        if st.st_size > MEDIA_MAX:
            return None
        try:
            body = p.read_bytes()
        except OSError:
            return None
        with self.lock:
            self.files.pop(name, None)
            self.files[name] = (sig, body)
            self._prune()
        return body


RANGE_RE = re.compile(r"^bytes=(\d{0,15})-(\d{0,15})$")  # 15 digits: far beyond any file, far below int() limits


def parse_range(header: str | None, size: int):
    """RFC 9110 14.2 for one byte range: None = serve the whole file (no Range, an unknown unit, a multi-range or an
    invalid range spec are ignored); (a, b) inclusive; "bad" = 416, only for a valid but unsatisfiable range."""
    if not header:
        return None
    m = RANGE_RE.match(header.strip())
    if not m or (not m.group(1) and not m.group(2)):
        return None
    a, b = m.group(1), m.group(2)
    if a == "":  # suffix: the last n bytes
        n = int(b)
        if n == 0 or size == 0:
            return "bad"
        return (max(0, size - n), size - 1)
    a = int(a)
    if b and int(b) < a:
        return None  # invalid spec: ignored
    if a >= size:
        return "bad"
    b = int(b) if b else size - 1
    return (a, min(b, size - 1))


def highlight_public() -> dict | None:
    """highlight.json re-checked field by field (only these fields, plain values, no local details)."""
    j = _media_json()
    if not j:
        return None

    def media(k, legacy):
        v = j.get(k)
        if v is None and "video" not in j and isinstance(j.get("files"), dict) and legacy in j["files"]:
            v = legacy  # a json of the first version: fixed names, listed in "files"
        return v if isinstance(v, str) and media_allowed(v) and (MEDIA / v).is_file() else None

    video, overlay = media("video", "highlight.mp4"), media("overlay", "highlight-overlay.mp4")
    if video is None:
        return None

    def s(k, n=40):
        v = j.get(k)
        return str(v)[:n] if isinstance(v, str) else None

    def i(k):
        v = j.get(k)
        return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None

    out = {"run": s("run", 8), "update": i("update"), "episode_at": s("episode_at", 16), "recorded_at": s("recorded_at", 16),
           "tz": s("tz", 10), "sr_ms": i("sr_ms"), "k": i("k"), "tas_ms": i("tas_ms"), "backend": s("backend", 32),
           "matched": j.get("matched") is True,
           "duration_s": j.get("duration_s") if isinstance(j.get("duration_s"), (int, float)) else None,
           "v": i("v"), "level": i("level"),
           "recorded_ts": D.ts_from_label(j.get("recorded_at"), j.get("tz")),
           "episode_ts": D.ts_from_label(j.get("episode_at"), j.get("tz")),
           # file names of this recording's set, as its json lists them (never a leftover file)
           "video": video, "overlay": overlay}
    if out["run"] is None or not re.fullmatch(r"r\d+", out["run"]) or out["sr_ms"] is None or out["k"] is None:
        return None
    if R.check(json.dumps(out, ensure_ascii=False)):
        return None
    return out


def lower_priority():
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes
            k32 = ctypes.WinDLL("kernel32")
            k32.GetCurrentProcess.restype = wintypes.HANDLE  # the default int truncates the pseudo-handle (call failed)
            k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            k32.SetPriorityClass(k32.GetCurrentProcess(), 0x4000)  # BELOW_NORMAL_PRIORITY_CLASS
        else:
            os.nice(10)
    except Exception:
        pass


def log(*a):
    print(dt.datetime.now().strftime("%H:%M:%S"), *a, flush=True)


def write_json(name: str, obj) -> int:
    """Writes runtime/dashboard/<name>.json.gz atomically; returns its version (mtime)."""
    raw = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
    p = OUT / f"{name}.json.gz"
    tmp = OUT / f"{name}.tmp.gz"
    tmp.write_bytes(gzip.compress(raw, 6))
    os.replace(tmp, p)
    return int(p.stat().st_mtime * 1000)


def write_text_atomic(p: Path, text: str):
    """Writes p through a temporary file; retries the replace for a moment (on Windows it fails while the server
    process has the file open for reading)."""
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    for i in range(20):
        try:
            os.replace(tmp, p)
            return
        except PermissionError:
            if i == 19:
                raise
            time.sleep(0.05)


class Dashboard:
    """role "server": answers the page, keeps a generator child running and reads its results; role "generator" (the
    child): computes the payloads. argv: the server's own arguments, passed on to the child."""

    def __init__(self, a, role: str = "server", argv=None):
        self.role, self.argv = role, list(argv or [])
        self.interval, self.min_refresh = a.interval, a.min_refresh
        self.machines = {"train": a.train_machine, "eval": a.eval_machine}
        # host label -> generic machine name for the data-source tile (never sent to the page as a mapping)
        self.sources = dict(kv.split("=", 1) for kv in (a.source_labels or "").split(",") if "=" in kv)
        self.tensorboard_url = a.tensorboard_url or ""
        self.show_alt = a.show_shortcut_line
        self.game_assets = not a.no_game_assets
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.levels: dict = {}  # TAS record -> geometry.Level (built in the generator only)
        self.level_dims: dict = {}  # TAS record -> [n, first] of its Level (what the status needs of it)
        self.traces = D.Traces(CACHE / "traces.pkl") if role == "generator" else None
        self.route_memo: dict = {}  # (traced life, bundle pin) -> route class
        self.sigs: dict = {}
        self.summary: dict = {}
        # the last generation's summary (data versions, SR / HS estimates, counts): a restarted server shows the
        # previous maps and curves at once (their files are still in runtime/dashboard) while it rebuilds them
        try:
            self.summary = json.loads((OUT / "summary_last.json").read_text(encoding="utf-8"))
        except Exception:
            self.summary = {}
        self.gen = {"running": False, "last_start": 0.0, "generated_at": self.summary.get("generated_at"),
                    "generated_ts": self.summary.get("generated_ts"), "seconds": None, "errors": {}}
        self._status = (0.0, None)
        self._status_lock = threading.Lock()  # one status computation at a time; the others get the last answer
        self.assets = []
        self.serif_v = None  # version of the self-served title font (serif_font.py), None = not built
        self.child = None  # the generator child (server role)
        self._child_lock = threading.Lock()
        self._state_sig = None
        self.stop = threading.Event()
        if role == "server":
            self._read_state(initial=True)
            self._load_status_snapshot()

    # ------------------------------------------------------------------ server: the generator child

    def start(self):
        """Server role: the generator child and its watcher, the assets / title font, and a first status computation
        (so the first page request after a restart finds it ready)."""
        threading.Thread(target=self.supervise, name="generator", daemon=True).start()
        threading.Thread(target=self.side_jobs, name="assets", daemon=True).start()
        threading.Thread(target=self.status, name="status-warm", daemon=True).start()

    def side_jobs(self):
        try:
            self.assets = A.prepare(ASSETS, log=log) if self.game_assets else []
        except Exception:
            log("assets failed:\n" + traceback.format_exc())
        while not self.stop.is_set():
            self.serif_font()
            self.stop.wait(self.interval)

    def generator_cmd(self) -> list[str]:
        return [sys.executable, SCRIPT, "--generator", *[x for x in self.argv if x != "--generator"]]

    def _spawn(self):
        cmd = self.generator_cmd()
        self.child = subprocess.Popen(cmd, cwd=str(ROOT), stdin=subprocess.PIPE)
        log(f"generator started (pid {self.child.pid})")

    def supervise(self):
        """Keeps exactly one generator child: a new one only after the last one has exited, RESPAWN_S later."""
        next_start = 0.0
        while not self.stop.is_set():
            ch = self.child
            if ch is not None and ch.poll() is not None:
                log(f"generator exited with code {ch.returncode}; starting a new one in {RESPAWN_S}s")
                with self.lock:
                    self.gen.update(running=False, errors={"generator": "exited"})
                    self._expire_status()
                self.child = None
                next_start = time.time() + RESPAWN_S
            if self.child is None and time.time() >= next_start:
                try:
                    self._spawn()
                except Exception:
                    log("generator start failed:\n" + traceback.format_exc())
                    next_start = time.time() + RESPAWN_S
            self._read_state()
            self.stop.wait(1.0)
        self.close()

    def close(self):
        """Closes the child's stdin: the child exits (also what happens when the server process ends)."""
        ch = self.child
        if ch is not None and ch.stdin is not None:
            try:
                ch.stdin.close()
            except OSError:
                pass

    def _wake(self):
        """Asks for a generation pass now (or right after the running one)."""
        if self.role == "generator":
            self.wake.set()
            return
        ch = self.child
        if ch is None or ch.stdin is None:
            return
        with self._child_lock:
            try:
                ch.stdin.write(b"wake\n")
                ch.stdin.flush()
            except (OSError, ValueError):
                pass

    def _expire_status(self):
        self._status = (0.0, self._status[1])

    def _read_state(self, initial: bool = False):
        """Takes the child's state file when it changed: pass running / done, map frontier, Mac clock offset, level
        sizes; after a finished pass also the new summary_last.json. At start (initial) only the lasting parts of the
        previous child's state (its pass state is that of a process that is gone)."""
        p = OUT / STATE
        sig = D._sig(p)
        if sig is None or sig == self._state_sig:
            return
        try:
            st = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return  # being replaced: next time
        self._state_sig = sig
        if not isinstance(st, dict):
            return
        with self.lock:
            if isinstance(st.get("level_dims"), dict):
                self.level_dims.update(st["level_dims"])
            if isinstance(st.get("map_frontier"), list) and len(st["map_frontier"]) == 2:
                self.map_frontier = tuple(st["map_frontier"])
            if isinstance(st.get("off"), (int, float)):
                self.off = float(st["off"])
            g = st.get("gen")
            if initial or not isinstance(g, dict):
                return
            done = g.get("generated_ts") is not None and g.get("generated_ts") != self.gen.get("generated_ts")
            self.gen.update({k: g.get(k) for k in ("running", "last_start", "generated_at", "generated_ts", "seconds")})
            self.gen["errors"] = g.get("errors") if isinstance(g.get("errors"), dict) else {}
        if done:
            try:
                summ = json.loads((OUT / "summary_last.json").read_text(encoding="utf-8"))
                with self.lock:
                    self.summary = summ
            except (OSError, ValueError):
                log("summary reload failed:\n" + traceback.format_exc())
        self._expire_status()

    # ---------------------------------------------------------------------- generator (child role)

    def run_generator(self):
        """Generator role: a pass every --interval, or at once on a "wake" line from the server; exits when its stdin
        closes (the server is gone)."""
        threading.Thread(target=self._watch_parent, name="parent", daemon=True).start()
        while True:
            try:
                self.generate()
            except Exception:
                log("generation failed:\n" + traceback.format_exc())
            self.wake.wait(self.interval)
            self.wake.clear()

    def _watch_parent(self):
        try:
            for line in sys.stdin.buffer:
                if line.strip() == b"wake":
                    self.wake.set()
        except Exception:
            pass
        log("generator: server gone, exiting")
        os._exit(0)

    def _write_state(self):
        try:
            OUT.mkdir(parents=True, exist_ok=True)
            with self.lock:
                st = {"pid": os.getpid(), "gen": dict(self.gen), "map_frontier": getattr(self, "map_frontier", None),
                      "off": getattr(self, "off", None), "level_dims": dict(self.level_dims)}
            write_text_atomic(OUT / STATE, json.dumps(st, ensure_ascii=False, default=str))
        except Exception:
            log("generator state failed:\n" + traceback.format_exc())

    def serif_font(self):
        """Keeps the title-font subset in line with the page's text (serif_font.py; built in a child process so the
        server's threads never wait for it)."""
        import serif_font as SF
        srcs = [STATIC / "index.html", STATIC / "app.js", OUT / "reports.json"]
        try:
            ver, done = SF.version(OUT / "fonts", srcs)
            if ver and not done:
                r = subprocess.run([sys.executable, str(Path(SF.__file__)), str(OUT / "fonts"), *map(str, srcs)],
                                   capture_output=True, text=True, timeout=900)
                ver = r.stdout.strip() or None
                log(f"title font subset {ver}" + ("" if r.returncode == 0 else f" failed: {r.stderr[-400:]}"))
            self.serif_v = ver
        except Exception:
            log("title font failed:\n" + traceback.format_exc())

    def request_refresh(self):
        with self.lock:
            if self.gen["running"]:
                return False, "正在重新计算，请稍候"
            wait = self.min_refresh - (time.time() - self.gen["last_start"])
            if wait > 0:
                return False, f"刚刚算过，{int(wait)} 秒后可再刷新"
            self._expire_status()
            # until the child reports the pass: no second request (the rate limit counts from now)
            self.gen["last_start"] = time.time()
        self._wake()
        return True, "已开始重新计算"

    def _section(self, name, errors, fn):
        try:
            fn()
        except Exception as e:
            errors[name] = type(e).__name__  # details stay in the server log only
            log(f"section {name} failed:\n" + traceback.format_exc())

    def level_of(self, run):
        """What the status needs of the run's Level (its TAS record): the loaded Level in the generator, its sizes
        (n, first; reported by the generator) in the server; None while not known yet."""
        rec = D.run_level(run)["record"]
        if rec is None:
            return None
        if rec in self.levels:
            return self.levels[rec]
        d = self.level_dims.get(rec)
        if isinstance(d, list) and len(d) == 2:
            return SimpleNamespace(n=int(d[0]), first=int(d[1]))
        return None

    def generate(self):
        with self.lock:
            self.gen["running"] = True
            self.gen["last_start"] = time.time()
        self._write_state()
        t0 = time.time()
        errors: dict = {}
        OUT.mkdir(parents=True, exist_ok=True)
        summary = dict(self.summary)
        vers = dict(summary.get("data") or {})

        def put(name, sig, build):
            if self.sigs.get(name) == sig and name in vers and (OUT / f"{name}.json.gz").exists():
                return
            vers[name] = write_json(name, build())
            self.sigs[name] = sig

        try:
            runs = D.discover_runs()
            mac = D.mac_runs(runs)
            off = D.mac_offset_hours(runs)
            self.off = off
            sel = D.default_run(runs)
            # one map set per level the Mac runs train; a level's maps follow its latest run (the default run when it
            # is that level's) and its TAS record
            by_level: dict = {}
            for r in mac:
                lv = D.run_level(r)
                if lv["record"] is not None:
                    by_level.setdefault(lv["level"], []).append(r)
            if not by_level:
                by_level = {1: []}
            levels = {}
            for lv, rs in sorted(by_level.items()):
                act = [r for r in rs if D.is_active(r)]
                cur_run = next((r for r in rs if r.name == sel), None) or (act or rs or [None])[-1]
                rec = D.run_level(cur_run)["record"] if cur_run is not None else D.LEVEL_RECORDS[1]
                try:
                    if rec not in self.levels:
                        from geometry import Level
                        self.levels[rec] = Level(CACHE, log=log, record=rec)
                    levels[lv] = (self.levels[rec], cur_run)
                except Exception as e:
                    errors[f"level{lv}"] = type(e).__name__
                    log(f"level {lv} failed:\n" + traceback.format_exc())
            summary["levels"] = sorted(levels)
            with self.lock:
                self.level_dims.update({L.record: [int(L.n), int(L.first)] for L in self.levels.values()})
            self._write_state()
            # payloads of levels no run trains any more are dropped from the listing
            # (and the level-1 names of the single-level version)
            for k in list(vers):
                m = re.match(r"^(?:geometry|frontier|history|placements)(?:_l(\d+))?$", k)
                if m and (m.group(1) is None or int(m.group(1)) not in levels):
                    del vers[k]

            for lv, (L, _) in levels.items():
                put(f"geometry_l{lv}", L.record, lambda L=L: L.outlines)
                put(f"placements_l{lv}", D._sig(D.placements_path(lv)), lambda lv=lv: D.placements(lv))

            def frontier():
                for lv, (L, cur_run) in levels.items():
                    name = cur_run.name if cur_run is not None else None
                    cur_st = D.run_status(cur_run, off, L, {**self.machines, "sources": self.sources}) if cur_run else {}
                    cur = int(cur_st.get("frontier") or L.default_frontier)
                    if name == sel:
                        self.map_frontier = (sel, cur)  # what the charts were built for (status() compares)
                    routes = D.run_routes(cur_run) if cur_run is not None else None
                    sig = (tuple(D._sig(r.path / "train.log") for r in runs), tuple(D._sig(r.path / "config.json") for r in runs),
                           cur, name, L.record, self.show_alt, routes["pin"] if routes else None)
                    put(f"frontier_l{lv}", sig, lambda L=L, name=name, cur=cur, routes=routes:
                        D.frontier_payload(L, runs, off, name, cur, self.show_alt, routes))
            self._section("frontier", errors, frontier)

            def curves():
                sig = tuple((r.name, D._sig(r.path / "config.json"), tuple(D._sig(Path(f)) for f in r.files("updates*.csv")))
                            for r in runs)
                put("curves", sig, lambda: D.curves_payload(runs, self.machines))
            self._section("curves", errors, curves)

            def traces():
                keep = set()
                allidx: dict = {}  # level -> traced episodes of the runs on that level's map record
                counts = {}
                by_run: dict = {}
                for r in mac:
                    idx = self.traces.index(r)
                    counts[r.name] = len(idx)
                    lv = D.run_level(r)
                    if lv["level"] in levels and levels[lv["level"]][0].record == lv["record"]:
                        allidx.setdefault(lv["level"], []).extend(idx)
                    by_run[r.name] = idx
                    if not idx:
                        continue
                    latest = idx[-ATTEMPTS_N:]
                    keep.update(i["key"] for i in latest)
                    R = D.run_routes(r)  # the run's own route bundle: its lines and its lives' route classes
                    L_r = self.levels.get(lv["record"])

                    def build(latest=latest, n=len(idx), run=r.name, level=lv["level"], R=R, L_r=L_r):
                        trs = [D.traj_payload(i, t, R, self.route_memo) for i in latest if (t := self.traces.get(i)) is not None]
                        out = {"run": run, "level": level, "total": n, "traj": trs}
                        if R and L_r is not None:
                            out["routes"] = D.routes_public(R, L_r)
                        return out
                    put(f"attempts_{r.name}", (len(idx), idx[-1]["key"], lv["record"], R["pin"] if R else None,
                                               L_r is not None), build)
                for lv, ai in allidx.items():
                    ai.sort(key=lambda i: i["t"])
                    sub, thin = D.nested_sample(ai, HISTORY_N)
                    keep.update(i["key"] for i in sub)
                    if ai:
                        runs_lv = {i["run"] for i in ai}
                        L, cur_run = levels[lv]
                        # the level's latest run: its bundle classifies every life on the map, and the lives since its
                        # last route-guidance change are a sample of their own (the default window of the page)
                        R = D.run_routes(cur_run) if cur_run is not None else None
                        since = None
                        if R and cur_run is not None:
                            try:
                                since = D.guidance_since(cur_run, by_run.get(cur_run.name) or [], off)
                            except Exception:
                                log("guidance window failed:\n" + traceback.format_exc())
                        s_items = [i for i in ai if since and i["run"] == cur_run.name and i.get("pin") in since["pins"]]
                        s_sub, s_thin = D.nested_sample(s_items, HISTORY_N) if s_items else ([], [])
                        keep.update(i["key"] for i in s_sub)

                        def build_h(thin=thin, n=len(ai), runs_lv=runs_lv, lv=lv, L=L, R=R, since=since, s_thin=s_thin,
                                    n_s=len(s_items), cur=cur_run.name if cur_run is not None else None):
                            # one list for both windows: "w" 1 = the full sample, 2 = the since-change sample, 3 = both
                            ws: dict = {}
                            for i in thin:
                                ws[i["key"]] = ws.get(i["key"], 0) | 1
                            for i in s_thin:
                                ws[i["key"]] = ws.get(i["key"], 0) | 2
                            items = sorted({i["key"]: i for i in [*thin, *s_thin]}.values(), key=lambda i: i["t"])
                            trs = [dict(D.traj_payload(i, t, R, self.route_memo), w=ws[i["key"]])
                                   for i in items if (t := self.traces.get(i)) is not None]
                            out = {"runs": [r.name for r in mac if counts.get(r.name) and r.name in runs_lv],
                                   "level": lv, "total": n, "traj": trs}
                            if R:
                                out["routes"] = D.routes_public(R, L)
                                out["routes_run"] = cur
                            if since and s_thin:
                                out["since"] = {"ts": since["ts"], "run": cur, "total": n_s}
                            return out
                        put(f"history_l{lv}", (tuple(sorted((k, v) for k, v in counts.items() if k in runs_lv)),
                                               L.record, R["pin"] if R else None, since["ts"] if since else None,
                                               len(s_items)), build_h)
                self.traces.save(keep)
                self.route_memo = {k: v for k, v in self.route_memo.items() if k[0] in keep}
                summary["traced_counts"] = counts
            self._section("traces", errors, traces)

            def sr():
                summary["sr"] = {r.name: (D.hs_estimate if D.run_goal(r) == "HS" else D.sr_estimate)(runs, r.name)
                                 for r in mac}
                summary["k0"] = D.k0_evals()
            self._section("sr", errors, sr)
            summary["data"] = vers
        except Exception as e:
            errors["generate"] = type(e).__name__
            log("generate failed:\n" + traceback.format_exc())
        finally:
            secs = round(time.time() - t0, 1)
            summary["generated_at"] = dt.datetime.now().strftime("%m-%d %H:%M:%S")
            summary["generated_ts"] = int(time.time())
            summary["gen_seconds"] = secs
            with self.lock:
                self.summary = summary
                self.gen.update(running=False, generated_at=summary["generated_at"], generated_ts=summary["generated_ts"],
                                seconds=secs, errors=errors)
                self._status = (0.0, None)
            try:
                write_text_atomic(OUT / "summary_last.json", json.dumps(summary, ensure_ascii=False, default=str))
            except Exception:
                log("summary snapshot failed:\n" + traceback.format_exc())
            self._write_state()  # after the summary: the server reloads it when it sees the finished pass
            log(f"generated in {secs}s" + (f", errors: {list(errors)}" if errors else ""))

    # ------------------------------------------------------------------------------- status

    def status(self) -> dict:
        """Cached for STATUS_TTL_S; an expired answer is still served at once while one background thread computes the
        next (a cold computation after a start, or a training CSV parsed anew, takes seconds). Only when there is no
        answer at all (no snapshot from the last process either) do the requests wait, for one computation."""
        ts, cached = self._status
        if cached is not None and time.time() - ts < STATUS_TTL_S:
            return cached
        if cached is None:
            with self._status_lock:
                if self._status[1] is None:
                    self._refresh_status()
            if self._status[1] is None:
                raise RuntimeError("status not available")  # logged by _refresh_status; the page gets a 500
            return self._status[1]
        if self._status_lock.acquire(blocking=False):
            def bg():
                try:
                    self._refresh_status()
                finally:
                    self._status_lock.release()
            threading.Thread(target=bg, name="status", daemon=True).start()
        return cached

    def _refresh_status(self):
        """Computes the status (caller holds _status_lock) and keeps a snapshot for the next server process."""
        try:
            res = self._compute_status()
        except Exception:
            log("status failed:\n" + traceback.format_exc())
            return
        self._status = (time.time(), res)
        try:
            write_text_atomic(OUT / STATUS_SNAPSHOT, json.dumps(res, ensure_ascii=False, default=str))
        except Exception:
            log("status snapshot failed:\n" + traceback.format_exc())

    def _load_status_snapshot(self):
        try:
            res = json.loads((OUT / STATUS_SNAPSHOT).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if isinstance(res, dict) and isinstance(res.get("status"), dict):
            self._status = (0.0, res)  # expired: served until the first computation replaces it

    def _compute_status(self) -> dict:
        runs = D.discover_runs()
        mac = D.mac_runs(runs)
        off = getattr(self, "off", None)
        if off is None:
            off = D.mac_offset_hours(runs)
        st = {}
        for r in mac:
            try:
                st[r.name] = D.run_status(r, off, self.level_of(r), {**self.machines, "sources": self.sources})
            except Exception as e:
                st[r.name] = {"run": r.name, "error": type(e).__name__}
                log(f"status {r.name} failed:\n" + traceback.format_exc())
        with self.lock:
            gen = {k: v for k, v in self.gen.items() if k != "last_start"}
            summ = self.summary
        # the charts are rebuilt every --interval; when the live frontier moved since the last build, rebuild now so
        # the tiles and the maps never disagree for long (no user rate limit: at most once per frontier change)
        sel_run = D.default_run(runs)
        live = (st.get(sel_run) or {}).get("frontier")
        built = getattr(self, "map_frontier", None)
        if live is not None and built is not None and built[0] == sel_run and int(live) != built[1]:
            wake = False
            with self.lock:
                if not self.gen["running"] and getattr(self, "_woken_for", None) != int(live):
                    self._woken_for = int(live)
                    wake = True
            if wake:
                self._wake()
        try:
            compute = D.discarded_steps(runs, D.default_run(runs))
        except Exception:
            compute = None
        res = {"now": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "now_ts": int(time.time()), "tz": D.tz_label(), "milestone": "M3",
               "runs": [r.name for r in mac], "default_run": D.default_run(runs), "status": st, "gen": gen,
               "goals": {r.name: ("HS 高分" if str(D.run_level(r).get("record") or "").startswith("HS") else "SR 竞速")
                         for r in mac},
               "interval_s": self.interval, "min_refresh_s": self.min_refresh, "machines": self.machines,
               "tensorboard_url": self.tensorboard_url, "data": summ.get("data") or {},
               "sr": summ.get("sr") or {}, "k0": summ.get("k0") or [],
               "traced_counts": summ.get("traced_counts") or {}, "assets": self.assets, "compute": compute,
               "levels": summ.get("levels") or [],
               "reports": R.public(R.load()), "highlight": highlight_public()}
        return res


DATA_RE = re.compile(r"^/api/data/([a-z0-9_]+)\.json$")
GZ_MEMO: dict = {}  # static .js / .css path -> ((size, mtime), gzip bytes)
FONT_RE = re.compile(r"^/font/serif-(700|900)\.woff$")
FILE_RE = re.compile(r"^/(static|assets)/([a-z0-9_]+\.(?:js|css|png|jpg|json))$")
MEDIA_RE = re.compile(r"^/media/(highlight(?:-overlay)?(?:-\d{9,12})?\.mp4)$")


def make_handler(dash: Dashboard, media: MediaCache | None = None):
    media = media or MediaCache()

    class H(BaseHTTPRequestHandler):
        server_version = "Dashboard"
        sys_version = ""

        def log_message(self, *a):
            pass

        def _send(self, code, body: bytes, ctype, cache="no-store", gz=False):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            if gz:
                self.send_header("Content-Encoding", "gzip")
                self.send_header("Vary", "Accept-Encoding")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8"), CTYPES[".json"])

        def _media(self, name: str, head: bool = False):
            body = media.get(name)
            if body is None:
                return self._send(404, "未找到".encode("utf-8"), "text/plain; charset=utf-8")
            size = len(body)
            rng = parse_range(self.headers.get("Range"), size)
            cache = "public, max-age=86400" if "v=" in (urlparse(self.path).query or "") else "no-cache"
            if rng == "bad":
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            a, b = rng if rng else (0, size - 1)
            self.send_response(206 if rng else 200)
            self.send_header("Content-Type", CTYPES[".mp4"])
            self.send_header("Accept-Ranges", "bytes")
            if rng:
                self.send_header("Content-Range", f"bytes {a}-{b}/{size}")
            self.send_header("Content-Length", str(b - a + 1))
            self.send_header("Cache-Control", cache)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            if not head:
                try:
                    mv = memoryview(body)
                    for o in range(a, b + 1, 1 << 18):
                        self.wfile.write(mv[o:min(b + 1, o + (1 << 18))])
                except (ConnectionError, OSError):
                    pass  # the browser dropped the request (seeking)

        def do_GET(self):
            u = urlparse(self.path)
            m = MEDIA_RE.match(u.path)
            if m:
                return self._media(m.group(1))
            if u.path in ("/", "/index.html"):
                html = (STATIC / "index.html").read_text(encoding="utf-8")
                html = html.replace("__SERIF_V__", dash.serif_v or "0")
                for f in ("app.js", "style.css"):  # cache-busting: bump on every change of the file
                    html = html.replace(f"/static/{f}\"", f"/static/{f}?v={int((STATIC / f).stat().st_mtime)}\"")
                return self._send(200, html.encode("utf-8"), CTYPES[".html"], "no-cache")
            if u.path == "/api/status":
                try:
                    return self._json(dash.status())
                except Exception as e:
                    log("status failed:\n" + traceback.format_exc())
                    return self._json({"error": type(e).__name__}, 500)
            m = DATA_RE.match(u.path)
            if m and m.group(1) in (dash.summary.get("data") or {}):
                p = OUT / f"{m.group(1)}.json.gz"
                if p.exists():
                    body = p.read_bytes()
                    cache = "public, max-age=86400" if "v=" in (u.query or "") else "no-cache"
                    if "gzip" in (self.headers.get("Accept-Encoding") or ""):
                        return self._send(200, body, CTYPES[".json"], cache, gz=True)
                    return self._send(200, gzip.decompress(body), CTYPES[".json"], cache)
            m = FONT_RE.match(u.path)
            if m and dash.serif_v:
                p = OUT / "fonts" / f"serif-{m.group(1)}.woff"
                if p.is_file():
                    cache = "public, max-age=604800" if "v=" in (u.query or "") else "no-cache"
                    return self._send(200, p.read_bytes(), "font/woff", cache)
            m = FILE_RE.match(u.path)
            if m:
                base = STATIC if m.group(1) == "static" else ASSETS
                allowed = (m.group(2) in dash.assets) if m.group(1) == "assets" else True
                p = base / m.group(2)
                if allowed and p.is_file():
                    cache = "no-cache"  # assets are regenerated when their version changes: always revalidate
                    if m.group(1) == "static" and "v=" in (u.query or ""):
                        cache = "public, max-age=604800"  # versioned by the page (?v=mtime)
                    if p.suffix in (".js", ".css") and "gzip" in (self.headers.get("Accept-Encoding") or ""):
                        st = p.stat()
                        hit = GZ_MEMO.get(p)
                        if not hit or hit[0] != (st.st_size, st.st_mtime_ns):
                            hit = ((st.st_size, st.st_mtime_ns), gzip.compress(p.read_bytes(), 6))
                            GZ_MEMO[p] = hit
                        return self._send(200, hit[1], CTYPES[p.suffix], cache, gz=True)
                    return self._send(200, p.read_bytes(), CTYPES[p.suffix], cache)
            self._send(404, "未找到".encode("utf-8"), "text/plain; charset=utf-8")

        def do_HEAD(self):
            m = MEDIA_RE.match(urlparse(self.path).path)
            if m:
                return self._media(m.group(1), head=True)
            self._send(405, b"", "text/plain")

        def do_POST(self):
            u = urlparse(self.path)
            if u.path == "/api/refresh" and not u.query and not u.params:
                ok, msg = dash.request_refresh()
                return self._json({"ok": ok, "msg": msg})
            self._send(404, b"", "text/plain")

    return H


def main(argv=None):
    ap = argparse.ArgumentParser(description="Ballance RL training-progress page")
    ap.add_argument("--host", default=os.environ.get("BALLRAIN_DASH_HOST", "0.0.0.0"), help="bind address")
    ap.add_argument("--port", type=int, default=int(os.environ.get("BALLRAIN_DASH_PORT", "8088")))
    ap.add_argument("--interval", type=float, default=600, help="seconds between recomputations")
    ap.add_argument("--min-refresh", type=float, default=300,
                    help="global rate limit for '立即刷新' (s): at most one recomputation per this many seconds")
    ap.add_argument("--tensorboard-url", default=os.environ.get("BALLRAIN_DASH_TENSORBOARD_URL", ""),
                    help="link target of the TensorBoard entry; empty = shown as not available")
    ap.add_argument("--train-machine", default=os.environ.get("BALLRAIN_DASH_TRAIN_MACHINE", "Mac mini"),
                    help="generic description of the training machine shown on the page")
    ap.add_argument("--eval-machine", default=os.environ.get("BALLRAIN_DASH_EVAL_MACHINE", "Windows PC"),
                    help="generic description of the evaluation machine shown on the page")
    ap.add_argument("--source-labels", default=os.environ.get("BALLRAIN_DASH_SOURCE_LABELS", "learner=Mac mini,retail=Windows PC"),
                    help="data-source tile: comma list HOSTLABEL=machine, plus learner= (the learner's own workers) and "
                         "retail= (retail episodes); use generic machine names only")
    ap.add_argument("--no-game-assets", action="store_true",
                    help="do not serve the game textures / menu font (the page falls back to plain colours)")
    ap.add_argument("--show-shortcut-line", action="store_true",
                    help="draw the shortcut-1 reference line (derived from a human recording; off by default)")
    ap.add_argument("--generator", action="store_true", help=argparse.SUPPRESS)  # the child the server starts
    argv = list(sys.argv[1:] if argv is None else argv)
    a = ap.parse_args(argv)
    lower_priority()
    if a.generator:
        Dashboard(a, role="generator").run_generator()
        return
    dash = Dashboard(a, argv=argv)
    try:  # loading torch's libraries holds the interpreter for seconds: before serving, not in the first status
        import torch  # noqa: F401
    except ImportError:
        pass
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(dash))
    srv.daemon_threads = True
    dash.start()
    log(f"dashboard listening on port {a.port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        dash.stop.set()
        dash.close()


if __name__ == "__main__":
    main()

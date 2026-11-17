"""Live training-progress page for the Ballance RL runs (Chinese UI, interactive charts).

    uv run python tools/dashboard/server.py [--host 0.0.0.0] [--port 8088] [--tensorboard-url URL]

Serves static/index.html, /api/status (small JSON, polled by the page every 60 s) and
/api/data/<name>.json (map / curve payloads, gzip). A background thread recomputes the payloads every
--interval seconds (default 10 min; the page's "立即刷新" asks for an earlier pass, rate-limited) into
runtime/dashboard/ (untracked). The level outlines are built once and cached. The process lowers its
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
import re  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # noqa: E402
from pathlib import Path  # noqa: E402
from urllib.parse import urlparse  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import assets as A  # noqa: E402
import data as D  # noqa: E402

ROOT = D.ROOT
OUT = ROOT / "runtime" / "dashboard"
CACHE = OUT / "cache"
ASSETS = OUT / "assets"
STATIC = HERE / "static"
STATUS_TTL_S = 20
ATTEMPTS_N = 80
HISTORY_N = 150
CTYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
          ".css": "text/css; charset=utf-8", ".json": "application/json; charset=utf-8",
          ".png": "image/png", ".jpg": "image/jpeg"}


def lower_priority():
    try:
        if sys.platform == "win32":
            import ctypes
            k32 = ctypes.windll.kernel32
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


class Dashboard:
    def __init__(self, a):
        self.interval, self.min_refresh = a.interval, a.min_refresh
        self.machines = {"train": a.train_machine, "eval": a.eval_machine}
        self.tensorboard_url = a.tensorboard_url or ""
        self.show_alt = a.show_shortcut_line
        self.game_assets = not a.no_game_assets
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.level = None
        self.traces = D.Traces(CACHE / "traces.pkl")
        self.sigs: dict = {}
        self.summary: dict = {}
        self.gen = {"running": False, "last_start": 0.0, "generated_at": None, "seconds": None, "errors": {}}
        self._status = (0.0, None)
        self.assets = []

    # --------------------------------------------------------------------------- generation

    def loop(self):
        try:
            self.assets = A.prepare(ASSETS, log=log) if self.game_assets else []
        except Exception:
            log("assets failed:\n" + traceback.format_exc())
        while True:
            try:
                self.generate()
            except Exception:
                log("generation failed:\n" + traceback.format_exc())
            self.wake.wait(self.interval)
            self.wake.clear()

    def request_refresh(self):
        with self.lock:
            if self.gen["running"]:
                return False, "正在重新计算，请稍候"
            wait = self.min_refresh - (time.time() - self.gen["last_start"])
            if wait > 0:
                return False, f"刚刚算过，{int(wait)} 秒后可再刷新"
            self._status = (0.0, None)
            self.wake.set()
            return True, "已开始重新计算"

    def _section(self, name, errors, fn):
        try:
            fn()
        except Exception as e:
            errors[name] = type(e).__name__  # details stay in the server log only
            log(f"section {name} failed:\n" + traceback.format_exc())

    def generate(self):
        from geometry import Level
        with self.lock:
            self.gen["running"] = True
            self.gen["last_start"] = time.time()
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
            if self.level is None:
                self.level = Level(CACHE, log=log)
            L = self.level
            runs = D.discover_runs()
            mac = D.mac_runs(runs)
            off = D.mac_offset_hours(runs)
            self.off = off
            sel = D.default_run(runs)

            put("geometry", 1, lambda: L.outlines)

            def frontier():
                cur_st = D.run_status(next(r for r in mac if r.name == sel), off, L.n, self.machines) if sel else {}
                cur = int(cur_st.get("frontier") or 19086)
                sig = (tuple(D._sig(r.path / "train.log") for r in runs), cur, sel, self.show_alt)
                put("frontier", sig, lambda: D.frontier_payload(L, runs, off, sel, cur, self.show_alt))
            self._section("frontier", errors, frontier)

            def curves():
                sig = tuple((r.name, tuple(D._sig(Path(f)) for f in r.files("updates*.csv"))) for r in runs)
                put("curves", sig, lambda: D.curves_payload(runs, self.machines))
            self._section("curves", errors, curves)

            def traces():
                keep = set()
                allidx = []
                counts = {}
                for r in mac:
                    idx = self.traces.index(r)
                    counts[r.name] = len(idx)
                    allidx += idx
                    if not idx:
                        continue
                    latest = idx[-ATTEMPTS_N:]
                    keep.update(i["key"] for i in latest)

                    def build(latest=latest, n=len(idx), run=r.name):
                        trs = [D.traj_payload(i, t) for i in latest if (t := self.traces.get(i)) is not None]
                        return {"run": run, "total": n, "traj": trs}
                    put(f"attempts_{r.name}", (len(idx), idx[-1]["key"]), build)
                allidx.sort(key=lambda i: i["t"])
                sub, thin = D.nested_sample(allidx, HISTORY_N)
                keep.update(i["key"] for i in sub)
                if allidx:
                    def build_h():
                        trs = [D.traj_payload(i, t) for i in thin if (t := self.traces.get(i)) is not None]
                        return {"runs": [r.name for r in mac if counts.get(r.name)], "total": len(allidx),
                                "traj": trs}
                    put("history", tuple(sorted(counts.items())), build_h)
                self.traces.save(keep)
                summary["traced_counts"] = counts
            self._section("traces", errors, traces)

            def sr():
                summary["sr"] = {r.name: D.sr_estimate(runs, r.name) for r in mac}
                summary["k0"] = D.k0_evals()
            self._section("sr", errors, sr)
            summary["data"] = vers
        except Exception as e:
            errors["generate"] = type(e).__name__
            log("generate failed:\n" + traceback.format_exc())
        finally:
            secs = round(time.time() - t0, 1)
            summary["generated_at"] = dt.datetime.now().strftime("%m-%d %H:%M:%S")
            summary["gen_seconds"] = secs
            with self.lock:
                self.summary = summary
                self.gen.update(running=False, generated_at=summary["generated_at"], seconds=secs, errors=errors)
                self._status = (0.0, None)
            log(f"generated in {secs}s" + (f", errors: {list(errors)}" if errors else ""))

    # ------------------------------------------------------------------------------- status

    def status(self) -> dict:
        ts, cached = self._status
        if cached is not None and time.time() - ts < STATUS_TTL_S:
            return cached
        runs = D.discover_runs()
        mac = D.mac_runs(runs)
        off = getattr(self, "off", None)
        if off is None:
            off = D.mac_offset_hours(runs)
        n = self.level.n if self.level is not None else 20588
        st = {}
        for r in mac:
            try:
                st[r.name] = D.run_status(r, off, n, self.machines)
            except Exception as e:
                st[r.name] = {"run": r.name, "error": type(e).__name__}
                log(f"status {r.name} failed:\n" + traceback.format_exc())
        with self.lock:
            gen = {k: v for k, v in self.gen.items() if k != "last_start"}
            summ = self.summary
        res = {"now": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "tz": D.tz_label(), "milestone": "M2",
               "runs": [r.name for r in mac], "default_run": D.default_run(runs), "status": st, "gen": gen,
               "interval_s": self.interval, "min_refresh_s": self.min_refresh, "machines": self.machines,
               "tensorboard_url": self.tensorboard_url, "data": summ.get("data") or {},
               "sr": summ.get("sr") or {}, "k0": summ.get("k0") or [],
               "traced_counts": summ.get("traced_counts") or {}, "assets": self.assets}
        self._status = (time.time(), res)
        return res


DATA_RE = re.compile(r"^/api/data/([a-z0-9_]+)\.json$")
FILE_RE = re.compile(r"^/(static|assets)/([a-z0-9_]+\.(?:js|css|png|jpg|json))$")


def make_handler(dash: Dashboard):
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
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8"), CTYPES[".json"])

        def do_GET(self):
            u = urlparse(self.path)
            if u.path in ("/", "/index.html"):
                html = (STATIC / "index.html").read_text(encoding="utf-8")
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
            m = FILE_RE.match(u.path)
            if m:
                base = STATIC if m.group(1) == "static" else ASSETS
                allowed = (m.group(2) in dash.assets) if m.group(1) == "assets" else True
                p = base / m.group(2)
                if allowed and p.is_file():
                    cache = "no-cache" if m.group(1) == "static" else "public, max-age=86400"
                    return self._send(200, p.read_bytes(), CTYPES[p.suffix], cache)
            self._send(404, "未找到".encode("utf-8"), "text/plain; charset=utf-8")

        def do_HEAD(self):
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
    ap.add_argument("--no-game-assets", action="store_true",
                    help="do not serve the game textures / menu font (the page falls back to plain colours)")
    ap.add_argument("--show-shortcut-line", action="store_true",
                    help="draw the shortcut-1 reference line (derived from a human recording; off by default)")
    a = ap.parse_args(argv)
    lower_priority()
    dash = Dashboard(a)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(dash))
    srv.daemon_threads = True
    threading.Thread(target=dash.loop, name="regen", daemon=True).start()
    log(f"dashboard listening on port {a.port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

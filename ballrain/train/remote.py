"""Remote rollout workers: worker processes on another machine feeding the learner over the LAN.

Learner side (scripts/train_m2.py --remote-listen HOST:PORT): a RemoteHub runs a listener thread; every connection
(one per remote machine, scripts/remote_worker_m2.py) gets a receive thread and a send thread:
- hello -> the hub answers with the worker template (EnvConfig with the remote backend, WorkerSpec fields, run name)
  and a block of worker ids;
- the receive thread puts the remote chunks / episode records into the learner's data queue exactly like a local
  worker's (tagged with backend, host and the offset worker id), so the learner's staleness check (max_stale / the
  remote limit) and statistics apply unchanged;
- the send thread sends the newest parameter message (weights, observation statistics, curriculum state) whenever
  publish() was called with a newer version; older unsent versions are skipped. publish() only swaps a reference,
  so the learner never waits on the network.
A disconnect ends both threads; the remote side reconnects and gets the latest parameters again.

Remote side (RemoteBridge, scripts/remote_worker_m2.py): local worker processes (ballrain.train.worker, unchanged)
on the local game copies; the bridge forwards their chunks to the hub and fans the parameters out to them. While
disconnected it drops their chunks (they would be stale) and keeps reconnecting; the workers keep their last
weights. Transport: multiprocessing.connection (pickle + HMAC authkey; LAN use only).
"""
from __future__ import annotations

import copy as _copy
from pathlib import Path
import os
import queue
import socket
import threading
import time
import traceback
from multiprocessing.connection import Client, Listener

PROTOCOL = 1


def parse_addr(s: str) -> tuple[str, int]:
    host, _, port = s.rpartition(":")
    return host or "0.0.0.0", int(port)


class RemoteHub:
    def __init__(self, addr, authkey: bytes, data_q, template: dict, log=print, wid_base: int = 100, wid_block: int = 100):
        self.addr, self.authkey, self.data_q, self.template, self.log = addr, authkey, data_q, template, log
        # identity of the template for the remote side (a bridge running older code may compare dataclasses by its own,
        # older field list and miss a change)
        import hashlib
        import pickle
        self.template_id = hashlib.sha256(pickle.dumps(template)).hexdigest()
        self._latest = None
        self._cond = threading.Condition()
        self._next_base = wid_base
        self._block = wid_block
        self.conns: dict = {}  # host -> info
        self.stats = {"connects": 0, "disconnects": 0, "chunks": 0, "bytes_params": 0}
        self._stop = False
        self._listener = Listener(addr, authkey=authkey)
        self._th = threading.Thread(target=self._accept, name="remote-hub", daemon=True)
        self._th.start()

    def publish(self, msg: dict) -> None:
        with self._cond:
            self._latest = msg
            self._cond.notify_all()

    def close(self):
        self._stop = True
        with self._cond:
            self._cond.notify_all()
        try:
            self._listener.close()
        except Exception:  # noqa: BLE001
            pass
        for info in list(self.conns.values()):
            try:
                info["conn"].close()
            except Exception:  # noqa: BLE001
                pass

    def _accept(self):
        while not self._stop:
            try:
                conn = self._listener.accept()
            except Exception as e:  # noqa: BLE001
                if not self._stop:
                    self.log(f"remote hub: accept failed: {type(e).__name__}: {e}")
                    time.sleep(1.0)
                continue
            threading.Thread(target=self._serve, args=(conn,), daemon=True, name="remote-conn").start()

    def _serve(self, conn):
        host = "?"
        try:
            if not conn.poll(30):
                raise ConnectionError("no hello within 30 s")
            hello = conn.recv()
            if hello.get("type") != "hello" or hello.get("protocol") != PROTOCOL:
                conn.send({"type": "error", "error": f"bad hello {hello!r}"})
                return
            host = f"{hello.get('host', '?')}#{hello.get('session', 0)}"
            known = self.conns.get(hello.get("host"))
            base = known["wid_base"] if known else self._next_base
            if not known:
                self._next_base += self._block
            conn.send({"type": "config", "template": self.template, "template_id": self.template_id, "wid_base": base})
            info = {"conn": conn, "wid_base": base, "host": hello.get("host"), "backend": hello.get("backend"),
                    "connected": time.time(), "chunks": 0}
            self.conns[hello.get("host")] = info
            self.stats["connects"] += 1
            self.log(f"remote hub: {host} connected ({hello.get('n_workers')} workers, backend {hello.get('backend')}, "
                     f"wid {base}..)")
            alive = {"ok": True}
            snd = threading.Thread(target=self._sender, args=(conn, alive), daemon=True, name="remote-send")
            snd.start()
            while alive["ok"] and not self._stop:
                m = conn.recv()
                if m.get("type") == "data":
                    d = m["msg"]
                    d["wid"] = base + int(d.get("wid", 0))
                    for e in d.get("episodes", []) or []:
                        e["host"] = hello.get("host")
                    d["host"] = hello.get("host")
                    d["remote"] = True
                    self.data_q.put(d)
                    if "obs" in d:
                        info["chunks"] += 1
                        self.stats["chunks"] += 1
                elif m.get("type") == "bye":
                    break
        except (EOFError, OSError, ConnectionError):
            pass
        except Exception as e:  # noqa: BLE001
            self.log(f"remote hub: {host}: {type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}")
        finally:
            self.stats["disconnects"] += 1
            self.log(f"remote hub: {host} disconnected")
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass
            alive_flag = locals().get("alive")
            if alive_flag:
                alive_flag["ok"] = False
            with self._cond:
                self._cond.notify_all()

    def _sender(self, conn, alive):
        sent = None
        try:
            while alive["ok"] and not self._stop:
                with self._cond:
                    while alive["ok"] and not self._stop and (self._latest is None or self._latest is sent):
                        self._cond.wait(timeout=5.0)
                    msg = self._latest
                if not alive["ok"] or self._stop or msg is None or msg is sent:
                    continue
                conn.send({"type": "params", "msg": msg})
                sent = msg
        except (EOFError, OSError, ConnectionError):
            pass
        finally:
            alive["ok"] = False


def kill_tree(pid: int) -> None:
    """Kill a process and its children (a worker and the game / engine process it started)."""
    import signal
    import subprocess
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
        return
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass


def lower_priority() -> None:
    """Below-normal priority for this process; its children (workers, engines) inherit it."""
    if os.name == "nt":
        import ctypes
        k = ctypes.windll.kernel32
        k.SetPriorityClass(k.GetCurrentProcess(), 0x00004000)  # BELOW_NORMAL_PRIORITY_CLASS
    else:
        os.nice(10)


class RemoteBridge:
    """Remote side: connects to the hub, runs local workers from its template, forwards chunks and parameters."""

    def __init__(self, addr, authkey: bytes, copies: list, backend: str, out_root_fn, log=print, worker_overrides=None,
                 host: str | None = None, max_queue: int = 64, label: str | None = None, headless: dict | None = None,
                 quiet_file=None, quiet_check_s: float = 5.0, quiet_stop_s: float = 40.0):
        self.addr, self.authkey, self.copies, self.backend = addr, authkey, copies, backend
        # label: the data source name the learner logs for this bridge's episodes (episodes.csv "backend", TB), e.g.
        # "headless2002" for the Windows host with the original 2002 physics_RT.dll; default = the env backend
        self.label = label or backend
        # headless: this machine's HeadlessConfig fields (the learner's template carries none: its paths are its own)
        self.headless = headless
        # quiet_file: while it exists every worker is stopped (their game / engine processes exit, the CPU goes idle);
        # the connection to the learner stays up (it just sees no chunks from here); deleting it starts them again.
        # Checked every quiet_check_s; a worker gets quiet_stop_s to stop on its own, then its process tree is killed,
        # so a stop completes within quiet_check_s + quiet_stop_s + a few seconds (< 60 s with the defaults)
        self.quiet_file = Path(quiet_file) if quiet_file else None
        self.quiet_check_s, self.quiet_stop_s = quiet_check_s, quiet_stop_s
        self.quiet = False
        self._t_quiet_check = 0.0
        self.out_root_fn, self.log = out_root_fn, log
        self.overrides = worker_overrides or {}
        self.host = host or socket.gethostname()
        self.session = 0
        self.conn = None
        self.template = None
        self.template_id = None
        self.wid_base = None
        self.procs, self.ctrl_qs, self.specs = [], [], []
        self.stop = False
        self.counts = {"forwarded": 0, "dropped": 0, "params": 0, "reconnects": 0, "fatal": 0}
        self.heartbeat = None  # path: a JSON health record written every 30 s (scripts/remote_supervisor_m2.py)
        self.max_queue = max_queue

    # --------------------------------------------------------------------------------- connection
    def connect(self, timeout_s: float | None = None) -> bool:
        t0 = time.time()
        while not self.stop:
            try:
                # the whole handshake in a thread with a deadline: Client() (auth challenge) and the first recv block
                # forever when the hub's process is going down mid-handshake (seen at a learner restart)
                box: dict = {}

                def attempt():
                    try:
                        cc = Client(self.addr, authkey=self.authkey)
                        cc.send({"type": "hello", "protocol": PROTOCOL, "host": self.host, "session": self.session + 1,
                                 "backend": self.label, "n_workers": len(self.copies)})
                        if not cc.poll(30):
                            raise ConnectionError("no config within 30 s")
                        box["conn"], box["msg"] = cc, cc.recv()
                    except BaseException as e:  # noqa: BLE001
                        box["error"] = e
                th = threading.Thread(target=attempt, daemon=True, name="bridge-connect")
                th.start()
                th.join(45)
                if th.is_alive():
                    raise ConnectionError("handshake timed out")
                if "error" in box:
                    raise box["error"]
                c, m = box["conn"], box["msg"]
                self.session += 1
                if m.get("type") != "config":
                    raise ConnectionError(f"hub refused: {m}")
                if self.template is not None and m.get("template_id") != self.template_id:
                    # the learner was restarted with another configuration: restart the workers on the new template
                    self.log("bridge: the learner's worker template changed; restarting the workers")
                    self.template, self.wid_base, self.template_id = m["template"], m["wid_base"], m.get("template_id")
                    self._restart_all = True
                if self.template is None:
                    self.template, self.wid_base, self.template_id = m["template"], m["wid_base"], m.get("template_id")
                self.conn = c
                self.log(f"bridge: connected to {self.addr} (session {self.session}, wid base {m['wid_base']})")
                return True
            except Exception as e:  # noqa: BLE001  (refused, timed out, auth)
                if timeout_s is not None and time.time() - t0 > timeout_s:
                    return False
                self.log(f"bridge: connect failed ({type(e).__name__}: {e}); retrying")
                time.sleep(3.0)
        return False

    # --------------------------------------------------------------------------------- workers
    def start_workers(self, ctx, data_q):
        from .worker import WorkerSpec, worker_main
        t = self.template
        self.data_q = data_q
        self.ctx = ctx
        for i, name in enumerate(self.copies):
            self._start_worker(i, name, 0)

    def _spec(self, i, name, restart):
        from .worker import WorkerSpec
        t = self.template
        cfg = _copy.deepcopy(t["env"])
        cfg.copy = name
        cfg.backend = self.backend
        if self.backend == "headless":
            cfg.headless = dict(self.headless or {})
        cfg.out_root = str(self.out_root_fn(name))
        seed = t["seed"] * 1000 + self.wid_base + i + 100 * restart + 7919 * self.session
        cfg.seed = seed
        if cfg.curriculum is not None:
            cfg.curriculum = dict(cfg.curriculum, seed=seed)
        kw = dict(t["worker"])
        kw.update(self.overrides)
        import dataclasses
        known = {f.name for f in dataclasses.fields(WorkerSpec)}
        unknown = sorted(set(kw) - known)
        if unknown:  # the learner runs newer worker code than this bridge: restart the bridge to pick it up
            self.log(f"bridge: WARNING the learner's worker template has fields this bridge does not know: {unknown}; "
                     f"they are ignored until the bridge is restarted with the current code")
            kw = {k: v for k, v in kw.items() if k in known}
        return WorkerSpec(wid=i, copy=name, env=cfg, seed=seed, **kw)

    def _start_worker(self, i, name, restart):
        from .worker import worker_main
        q = self.ctx.Queue()
        s = self._spec(i, name, restart)
        p = self.ctx.Process(target=worker_main, args=(s, q, self.data_q), daemon=True, name=f"ballrain-rw{i}")
        p.start()
        if i < len(self.procs):
            self.procs[i], self.ctrl_qs[i], self.specs[i] = p, q, s
        else:
            self.procs.append(p), self.ctrl_qs.append(q), self.specs.append(s)
        if getattr(self, "_last_params", None) is not None:
            q.put(self._last_params)

    # --------------------------------------------------------------------------------- loop
    def _recv_loop(self):
        while not self.stop:
            c = self.conn
            if c is None:
                time.sleep(0.5)
                continue
            try:
                m = c.recv()
            except (EOFError, OSError, ConnectionError):
                self._lost(c)
                continue
            if m.get("type") == "params":
                self._last_params = m["msg"]
                self.counts["params"] += 1
                if self.quiet:  # no live workers: the newest parameters go to them when they start again
                    continue
                for q in self.ctrl_qs:
                    q.put(m["msg"])

    def _lost(self, c):
        if self.conn is c:
            self.conn = None
            try:
                c.close()
            except Exception:  # noqa: BLE001
                pass
            self.log("bridge: disconnected; reconnecting")
            self.counts["reconnects"] += 1
            threading.Thread(target=self.connect, daemon=True).start()

    # --------------------------------------------------------------------------------- quiet switch
    def quiet_requested(self) -> bool:
        return self.quiet_file is not None and self.quiet_file.exists()

    def _stop_workers(self, data_q, wait_s: float):
        """Stop every worker: the stop command first (the worker closes its env, i.e. its game / engine process),
        then, for any still alive after wait_s, its whole process tree."""
        for q in self.ctrl_qs:
            try:
                q.put({"cmd": "stop"})
            except Exception:  # noqa: BLE001
                pass
        t_end = time.time() + wait_s
        while time.time() < t_end and any(p.is_alive() for p in self.procs):
            try:  # keep reading: a worker cannot exit while its last messages sit in a full pipe
                m = data_q.get(timeout=0.5)
                self._forward(m)
            except queue.Empty:
                pass
        for p in self.procs:
            if p.is_alive():
                kill_tree(p.pid)
                p.join(timeout=5)
                if p.is_alive():
                    p.terminate()

    def check_quiet(self, data_q, restarts: list, now: float | None = None) -> None:
        """Enter / leave the quiet state when the quiet file appeared / went away (at most every quiet_check_s)."""
        now = time.time() if now is None else now
        if self.quiet_file is None or now - self._t_quiet_check < self.quiet_check_s:
            return
        self._t_quiet_check = now
        want = self.quiet_requested()
        if want and not self.quiet:
            self.log(f"bridge: quiet file {self.quiet_file} present: stopping all {len(self.procs)} workers")
            self.quiet = True
            self._stop_workers(data_q, self.quiet_stop_s)
            self.log("bridge: quiet: all workers stopped (still connected)")
        elif not want and self.quiet:
            self.log(f"bridge: quiet file gone: starting {len(self.copies)} workers")
            self.quiet = False
            for i, name in enumerate(self.copies):
                restarts[i] += 1
                self._start_worker(i, name, restarts[i])

    def _forward(self, m):
        if "fatal" in m:  # a worker that died while being stopped (quiet switch): not restarted, reported as an error
            m = {"wid": m.get("wid"), "errors": [f"fatal while stopping: {str(m['fatal'])[-800:]}"]}
        m["backend"] = self.label
        c = self.conn
        if c is None:
            self.counts["dropped"] += 1
            return
        try:
            c.send({"type": "data", "msg": m})
            self.counts["forwarded"] += 1
        except (OSError, EOFError, ConnectionError, ValueError):
            self.counts["dropped"] += 1
            self._lost(c)

    def run(self, ctx, data_q, should_stop=lambda: False, on_fatal=None):
        if not self.connect():
            return
        self.ctx, self.data_q = ctx, data_q
        self.quiet = self.quiet_requested()
        if self.quiet:
            self.log(f"bridge: quiet file {self.quiet_file} present: workers wait until it is deleted")
        else:
            self.start_workers(ctx, data_q)
        threading.Thread(target=self._recv_loop, daemon=True, name="bridge-recv").start()
        restarts = [0] * len(self.copies)
        t_beat = 0.0
        while not self.stop and not should_stop():
            if self.heartbeat is not None and time.time() - t_beat > 30:
                t_beat = time.time()
                try:
                    import json as _json
                    tmp = Path(str(self.heartbeat) + ".tmp")
                    tmp.write_text(_json.dumps(dict(self.counts, t=t_beat, connected=self.conn is not None,
                                                    pid=os.getpid(), quiet=self.quiet)), encoding="utf-8")
                    os.replace(tmp, self.heartbeat)
                except OSError:
                    pass
            self.check_quiet(data_q, restarts)
            if self.quiet:
                self._restart_all = False  # the workers start on the newest template when the quiet file goes
                try:
                    self._forward(data_q.get(timeout=2.0))  # last chunks of stopping workers
                except queue.Empty:
                    pass
                continue
            if getattr(self, "_restart_all", False):
                self._restart_all = False
                for i, p in enumerate(self.procs):
                    self.ctrl_qs[i].put({"cmd": "stop"})
                t_end = time.time() + 120
                while time.time() < t_end and any(p.is_alive() for p in self.procs):
                    try:  # keep reading: a worker cannot exit while its last messages sit in a full pipe
                        data_q.get(timeout=0.5)
                    except queue.Empty:
                        pass
                for i, p in enumerate(self.procs):
                    if p.is_alive():
                        p.terminate()
                    if on_fatal:
                        on_fatal(self.copies[i])
                    restarts[i] += 1
                    self._start_worker(i, self.copies[i], restarts[i])
            try:
                m = data_q.get(timeout=2.0)
            except queue.Empty:
                continue
            if "fatal" in m:
                i = m["wid"]
                restarts[i] += 1
                self.counts["fatal"] += 1
                self.log(f"bridge: worker {i} fatal (restart {restarts[i]}): {m['fatal'][-800:]}")
                self.procs[i].join(timeout=30)
                if on_fatal:
                    on_fatal(self.copies[i])
                self._start_worker(i, self.copies[i], restarts[i])
                m = {"wid": i, "errors": [f"fatal (restarted): {m['fatal'][-800:]}"]}
            self._forward(m)

    def shutdown(self, wait_s: float = 120.0):
        self.stop = True
        for q in self.ctrl_qs:
            try:
                q.put({"cmd": "stop"})
            except Exception:  # noqa: BLE001
                pass
        t_end = time.time() + wait_s
        for p in self.procs:
            p.join(timeout=max(1.0, t_end - time.time()))
            if p.is_alive():
                p.terminate()
        c = self.conn
        if c is not None:
            try:
                c.send({"type": "bye"})
                c.close()
            except Exception:  # noqa: BLE001
                pass
        for q in self.ctrl_qs:
            q.cancel_join_thread()

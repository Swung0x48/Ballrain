"""M2 training: PPO on level 1 (SR) with a reverse curriculum of TAS start states (launches the game).

    uv run python scripts/train_m2.py --run <name> [--workers 12] [--total-steps 50e6] ...
    uv run python scripts/train_m2.py --run <name> --resume            # continue from <run>/latest.pt
    uv run python scripts/train_m2.py --run smoke --fake --workers 2 --total-steps 3000   # no game (FakeLauncher)

Learner in this process, one rollout worker process per game copy (ballrain.train.worker; copies
train-<copy_offset> .. ). Outputs in <runtime>/m2-train/<run>/: config.json, updates.csv (one row per update),
episodes.csv (one row per RL episode), runs.csv (one row per game process), tb/ (tensorboard), ckpt_*.pt and
latest.pt (model, optimiser, observation statistics, curriculum state, counters), episodes/<copy>/<run_id>/ (kept
runs: finishing processes, .json + .tas), pids.json, train.log.

Stop: create the file <run>/STOP (the learner stops the workers, which close their game processes; any game
process still running in the copies used is then stopped by path, never by image name).

Curriculum (docs/observations.md 9): one ReverseCurriculum over SR01_1_14_457_264 in the learner; a TAS-start
episode succeeds when, within that life, it reaches the next checkpoint (the sector number grows) or LevelFinish
(--success cp, default) or only at LevelFinish (--success finish). The state is pushed to the workers after every update.
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import queue
import sys
import time
from collections import Counter, defaultdict, deque
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

RECORD = "SR01_1_14_457_264"
TERMS = ("progress", "time", "finish", "death", "death_refund", "shaping", "checkpoint", "stall_penalty", "gate_bonus", "split_bonus",
         "camera_off_quarter", "zone_penalty", "wall_push")
EP_COLS = ["update", "t", "wid", "backend", "host", "copy", "kind", "probes", "death_probe", "splits", "start_frame", "start_sector", "end", "end_sector", "max_sector",
           "success_cp", "finish", "ret", "steps", "frames", "sr_ms_end", "start_sr_ms", "agent_s", "s_start", "s_end",
           "speed", "jumps", "cap_frames", "version_start", "new_process", "run_id", "ball_order", "respawn_skipped", "zone_entries", "zones"] + [f"r_{k}" for k in TERMS] + ["actions"]
UPDATE_COLS = (["update", "steps", "frames", "wall_s", "steps_per_s", "frames_per_s", "frontier", "frontier_sector",
                "band_success", "band_n", "band_finish", "gate_blocked", "episodes", "ret_mean", "ret_tas", "ret_respawn", "success_cp_tas", "finish_tas",
                "deaths", "finishes", "stalls", "caps", "camera_off_quarter", "invalid_runs", "speed_mean", "runs", "finish_runs", "sr_ms_min", "sr_ms_k0_min",
                "k_p10", "k_p50", "k0_eps", "update_s", "collect_s", "reset_frac", "dropped_chunks", "worker_restarts",
                "best_sr_k0", "best_sr_any"] + [f"deaths_s{s}" for s in range(1, 5)]
               + [f"{p}_bin{b}" for b in range(12) for p in ("cp", "fin")]
               + [f"{p}_sec{s}" for s in range(1, 5) for p in ("cp", "fin")]
               + [f"{p}_sec{s}" for s in range(1, 5) for p in ("rsp_cp", "rsp_n")]
               + [f"{p}_sec{s}" for s in range(1, 5) for p in ("rest_cp", "rest_n")]
               + [f"u_{k}" for k in ("pg_loss", "v_loss", "entropy", "approx_kl", "clip_frac", "early_stop",
                                     "explained_var", "adv_std", "ret_mean")])
RUN_COLS = ["update", "t", "copy", "run_id", "end_reason", "sr_ms", "start_frame", "frames", "lives", "deaths",
            "framelog_match", "handover_mismatches", "invalid", "ball_order", "dir"]


def gated_curriculum(starts, cfg, finish_gate: float):
    """ReverseCurriculum whose frontier also needs the LevelFinish rate of the band episodes to reach finish_gate:
    with --success cp a frontier in an early sector only has to reach the next checkpoint, and the gate keeps it from
    running ahead of whole-run competence (the band episodes continue to the end of the level after the checkpoint)."""
    from ballrain.curriculum import ReverseCurriculum

    class Gated(ReverseCurriculum):
        def __init__(self):
            super().__init__(starts, cfg)
            self.finishes = deque(maxlen=max(cfg.min_episodes, 1))
            self.finish_gate = finish_gate
            self.gate_blocked = 0

        def update_gated(self, k: int, success: bool, finish: bool) -> bool:
            if not self.in_band(k):
                return False
            self.outcomes.append(bool(success))
            self.finishes.append(bool(finish))
            if len(self.outcomes) >= self.cfg.min_episodes and not self.at_start:
                rate, frate = float(np.mean(self.outcomes)), float(np.mean(self.finishes))
                if rate >= self.cfg.advance_at:
                    if frate < self.finish_gate:
                        self.gate_blocked += 1
                        return False
                    self.frontier = self._snap(self.frontier - self.cfg.step)
                    self.history.append((self.frontier, rate))
                    self.outcomes.clear()
                    self.finishes.clear()
                    return True
            return False

        def state_dict(self):
            return dict(super().state_dict(), finishes=list(self.finishes), finish_gate=self.finish_gate)

        def load_state_dict(self, st):
            super().load_state_dict(st)
            self.finishes = deque(st.get("finishes", []), maxlen=max(self.cfg.min_episodes, 1))

    return Gated()


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--init-from", help="checkpoint to initialise model / obs stats from (new run)")
    p.add_argument("--init-full", action="store_true",
                   help="with --init-from: also take the optimiser, curriculum state and counters (continue a run elsewhere)")
    p.add_argument("--fake", action="store_true", help="FakeLauncher (no game), for smoke tests")
    p.add_argument("--backend", choices=("retail", "headless"), default="retail",
                   help="retail: Player.exe copies train-NN; headless: the approximate engine (ballrain.headless), instances hl-NN")
    p.add_argument("--hl-bin", default=None, help="headless: SimTool binary (default $BALLRAIN_HL_BIN)")
    p.add_argument("--hl-root", default=None, help="headless: Ballance game folder (default $BALLRAIN_HL_ROOT)")
    p.add_argument("--hl-socket", default=None, help="headless: unix | tcp (default $BALLRAIN_HL_SOCKET)")
    p.add_argument("--hl-ball-order", default=None,
                   help="headless: ball-ball contact order (ckid | reverse | perm:<seed> | table:... | retail[:<seed>])")
    p.add_argument("--split-k", type=float, default=0.0, help="split bonus scale (0 = off; WorkerSpec.split_k)")
    p.add_argument("--split-min-ref-s", type=float, default=5.0)
    p.add_argument("--split-extra", default="", help="JSON [[lo_frame, hi_frame, seconds], ...] (WorkerSpec.split_extra)")
    p.add_argument("--exclude-starts", default="", help="record frame ranges lo-hi never used as starts")
    p.add_argument("--path-dir", default=None, help="directory of the reference line assets (FeatureConfig.path_dir)")
    p.add_argument("--gates", default="", help="JSON sub-goal gates (ballrain.train.worker.WorkerSpec.gates)")
    p.add_argument("--zones", default="",
                   help='JSON penalty zones {name: {"box": [xmin, xmax, zmin, zmax], "y_below": y, "penalty": p}} '
                        '(WorkerSpec.zones: one-off penalty per entry, the life goes on)')
    p.add_argument("--wall-push-rate", type=float, default=0.0,
                   help="penalty per game second while pushing into an unclimbable wall (WorkerSpec.wall_push_rate; 0 = off)")
    p.add_argument("--wall-min-rise", type=float, default=2.0, help="wall height above the ball's lowest point that counts as unclimbable")
    p.add_argument("--wall-max-progress", type=float, default=2.0, help="wall push: only below this route progress (units / s)")
    p.add_argument("--route-probes", default="", help='JSON {name: [xmin, xmax, zmin, zmax]}: per-route visit / death stats')
    p.add_argument("--remote-listen", default=None,
                   help="HOST:PORT: accept remote rollout workers (scripts/remote_worker_m2.py, ballrain.train.remote)")
    p.add_argument("--remote-key", default=None, help="shared secret (default $BALLRAIN_REMOTE_KEY)")
    p.add_argument("--remote-backend", default="retail", help="backend of the remote workers' env")
    p.add_argument("--max-stale-remote", type=int, default=2, help="max_stale for remote chunks")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--copy-offset", type=int, default=0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--total-steps", type=float, default=200e6)
    p.add_argument("--batch-steps", type=int, default=4096)
    p.add_argument("--chunk", type=int, default=128)
    p.add_argument("--max-stale", type=int, default=1, help="drop chunks collected more than this many updates ago")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.997)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=None)
    p.add_argument("--reward-scale", type=float, default=0.1)
    p.add_argument("--hidden", default="256,256")
    p.add_argument("--activation", default="tanh")
    p.add_argument("--torch-threads", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--action-repeat", type=int, default=4)
    p.add_argument("--hs-block", action="store_true", help="keep the HS feature block (off for SR)")
    p.add_argument("--far-waypoints", default="", help="f6: line points this far ahead in arc units, e.g. 20,30,45")
    p.add_argument("--ref-speed-at", default="", help="f6: reference speed at these arc offsets, e.g. 0,15,30")
    p.add_argument("--world-actions", action="store_true",
                   help="world-frame directional features (FeatureConfig camera=world) and world-direction actions")
    p.add_argument("--reward", default="{}", help="RewardConfig overrides as JSON")
    p.add_argument("--no-respawn", action="store_true",
                   help="end the game process at BallOff (EnvConfig.per_life=False): no respawn lives at the checkpoint")
    p.add_argument("--cap-mult", type=float, default=2.5)
    p.add_argument("--stall-s", type=float, default=10.0)
    p.add_argument("--stall-terminal", action="store_true",
                   help="a stall is a terminal failure (death penalty + sector refund) instead of a bootstrapped truncation")
    p.add_argument("--stall-speed", type=float, default=None,
                   help="the stall clock pauses while the ball is faster than this (units/s): run-ups are not cut")
    p.add_argument("--stall-max-s", type=float, default=15.0, help="with --stall-speed: max time without new arc progress")
    p.add_argument("--rest-starts", default="", help="at-rest starts: lo-hi record frame ranges, e.g. 11760-11900,13450-13650")
    p.add_argument("--p-rest", type=float, default=0.0, help="probability of an at-rest start for a new process")
    p.add_argument("--rehearse-starts", default="", help="rehearsal start ranges lo-hi (record frames), e.g. 4900-5075")
    p.add_argument("--p-rehearse", type=float, default=0.0, help="probability of a rehearsal start for a new process")
    p.add_argument("--stall-extra", type=float, default=0.0, help="with --stall-terminal: extra penalty on top of a death's")
    p.add_argument("--stall-compensate", action="store_true",
                   help="with --stall-terminal: divide the stall penalty by gamma ** window steps (never cheaper than dying now)")
    p.add_argument("--respawn-stall-s", type=float, default=None, help="stall limit of respawn lives (default --stall-s)")
    p.add_argument("--p-early-respawn", type=float, default=1.0,
                   help="probability of playing a respawn life whose sector the record enters before the frontier; "
                        "otherwise the process ends and a new curriculum start follows (WorkerSpec.p_early_respawn; "
                        "1.0 = always, the old behaviour)")
    p.add_argument("--success", choices=("cp", "finish"), default="cp")
    p.add_argument("--finish-gate", type=float, default=0.0,
                   help="the frontier also needs this LevelFinish rate over the band episodes (with --success cp)")
    p.add_argument("--set-frontier", type=int, default=None, help="move the curriculum frontier to this frame (resume)")
    p.add_argument("--cu-weighted", type=float, default=0.0,
                   help="fraction of the starts drawn by failure-weighted bins (ReverseCurriculum weighted)")
    p.add_argument("--cu-bin-floor", type=float, default=0.05)
    p.add_argument("--cu-bin-frames", type=int, default=2000)
    p.add_argument("--start-back", type=int, default=1500)
    p.add_argument("--cu-step", type=int, default=1000)
    p.add_argument("--cu-band", type=int, default=1500)
    p.add_argument("--cu-recent", type=float, default=0.7)
    p.add_argument("--advance-at", type=float, default=0.6)
    p.add_argument("--min-episodes", type=int, default=40)
    p.add_argument("--p-level-start", type=float, default=0.3, help="k = 0 probability once the frontier is at the start")
    p.add_argument("--ckpt-every", type=int, default=25)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--prune", default="finish")
    p.add_argument("--trace-sample", type=float, default=0.0,
                   help="fraction of game processes traced and kept for export spot-checks (scripts/sim2real_m2.py)")
    return p.parse_args(argv)


class Logger:
    def __init__(self, run_dir: Path, resume: bool):
        from torch.utils.tensorboard import SummaryWriter
        self.dir = run_dir
        self.tb = SummaryWriter(str(run_dir / "tb"))
        self.files = {}
        self.writers = {}
        self.cols = {"episodes": EP_COLS, "runs": RUN_COLS}
        self.resume = resume
        self.log = open(run_dir / "train.log", "a", encoding="utf-8")

    def row(self, name: str, d: dict, cols=None):
        if name not in self.writers:
            f = self.dir / f"{name}.csv"
            cols = cols or self.cols.get(name) or list(d.keys())
            i = 1
            while f.is_file():  # an existing file with fewer columns (older trainer): continue in <name>.<i>.csv
                with open(f, encoding="utf-8") as fh:
                    old_cols = next(csv.reader(fh), [])
                if set(cols) <= set(old_cols):
                    cols = old_cols
                    break
                f = self.dir / f"{name}.{i}.csv"
                i += 1
            new = not f.is_file()
            fh = open(f, "a", newline="", encoding="utf-8")
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            if new:
                w.writeheader()
            self.files[name], self.writers[name] = fh, w
        self.writers[name].writerow(d)

    def flush(self):
        for fh in self.files.values():
            fh.flush()
        self.tb.flush()
        self.log.flush()

    def msg(self, s: str):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {s}"
        print(line, flush=True)
        self.log.write(line + "\n")
        self.log.flush()


def make_env_config(a, out_root: Path):
    from ballrain.env import EnvConfig
    from ballrain.features import FeatureConfig
    from ballrain.rewards import RewardConfig
    rc = RewardConfig(mode="sr", **json.loads(a.reward))
    fl = lambda x: tuple(float(v) for v in x.split(",") if v.strip()) if x else ()  # noqa: E731
    from ballrain.path import asset_path, file_sha256, portable_path_dir, resolve_path_dir
    # repo-relative, so remote workers on another machine resolve it in their own checkout, plus the asset's sha256
    # (a worker whose copy of the line differs refuses to run: ballrain.path.RouteError)
    path_dir = portable_path_dir(a.path_dir)
    asset = asset_path(1, resolve_path_dir(path_dir))
    if path_dir is not None and not asset.is_file():
        raise SystemExit(f"--path-dir {a.path_dir}: no reference line {asset}")
    fc = FeatureConfig(hs_block=a.hs_block, camera="world" if a.world_actions else "orient_ref",
                       far_waypoints=fl(a.far_waypoints), ref_speed_at=fl(a.ref_speed_at), path_dir=path_dir,
                       path_sha256=file_sha256(asset) if asset.is_file() else None)
    cu = {"start_back": a.start_back, "step": a.cu_step, "band": a.cu_band, "recent": a.cu_recent,
          "advance_at": a.advance_at, "min_episodes": a.min_episodes, "p_level_start": a.p_level_start}
    hl = None
    if a.backend == "headless":
        hl = {k: v for k, v in (("binary", a.hl_bin), ("root", a.hl_root), ("socket", a.hl_socket),
                                ("ball_order", a.hl_ball_order)) if v}
    return EnvConfig(level=1, mode="sr", action_repeat=a.action_repeat, out_root=str(out_root), start_record=RECORD,
                     backend=a.backend, headless=hl,
                     curriculum=cu, features=fc, reward=rc, max_process_frames=200000, world_actions=a.world_actions,
                     per_life=not a.no_respawn,
                     rest_starts=[[int(v) for v in r.split("-")] for r in a.rest_starts.split(",") if r.strip()] or None,
                     p_rest=a.p_rest,
                     rehearse_starts=[[int(v) for v in r.split("-")] for r in a.rehearse_starts.split(",") if r.strip()] or None,
                     p_rehearse=a.p_rehearse,
                     exclude_starts=[[int(v) for v in r.split("-")] for r in a.exclude_starts.split(",") if r.strip()] or None)


def stop_copies(names, log=None):
    """Stop every Player.exe whose image is runtime/<copy>/Bin/Player.exe of the copies this run used."""
    from ballrain import instances as I
    stopped = []
    for n in names:
        c = I.get_copy(n)
        for pid in I.running_in_copy(c):
            try:
                with I.ProcessHandle(pid, c.exe) as h:
                    h.verify()
                    h.terminate()
                    h.wait(10)
                stopped.append((n, pid))
            except Exception as e:  # noqa: BLE001
                if log:
                    log(f"could not stop {n} pid {pid}: {e}")
    return stopped


def main(argv=None) -> int:
    a = parse(argv)
    if not 0.0 <= a.p_early_respawn <= 1.0:
        raise SystemExit(f"--p-early-respawn {a.p_early_respawn}: must be in [0, 1]")
    import torch
    from ballrain import instances
    from ballrain.curriculum import CurriculumConfig, ReverseCurriculum, TasStarts
    from ballrain.train.ppo import ActorCritic, RunningMeanStd, explained_variance, gae, normalize, ppo_update
    from ballrain.train.worker import WorkerSpec, worker_main

    run_dir = instances.runtime_dir() / "m2-train" / a.run
    ckpt_path = run_dir / "latest.pt"
    if run_dir.exists() and not a.resume and (run_dir / "config.json").is_file():
        raise SystemExit(f"{run_dir} exists; use --resume or another --run")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "STOP").unlink(missing_ok=True)
    if a.resume:
        if not ckpt_path.is_file():
            raise SystemExit(f"no {ckpt_path}")
        ck = torch.load(ckpt_path, weights_only=False)
        saved = ck["args"]
        for k in ("hidden", "activation", "action_repeat", "hs_block", "world_actions"):  # architecture / observation: keep
            setattr(a, k, saved.get(k, getattr(a, k)))
    else:
        ck = None
    (run_dir / ("config.json" if not a.resume else f"config.resume-{time.strftime('%Y%m%d-%H%M%S')}.json")).write_text(
        json.dumps(vars(a), indent=1), encoding="utf-8")
    log = Logger(run_dir, a.resume)
    log.tb.add_text("config", "    " + json.dumps(vars(a), indent=1).replace("\n", "\n    "),
                    global_step=int(ck["counters"]["steps"]) if ck is not None else 0)
    torch.set_num_threads(a.torch_threads)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    gen = torch.Generator().manual_seed(a.seed)

    env_cfg = make_env_config(a, run_dir / "episodes")
    if a.fake:  # the toy level has no TAS start states: plain starts from level load
        env_cfg.start_record = None
        cu_cfg = env_cfg.curriculum
        env_cfg.curriculum = None
    starts = TasStarts(RECORD)
    if a.exclude_starts:
        keep = np.ones(len(starts.valid), bool)
        for r in a.exclude_starts.split(","):
            lo, hi = (int(v) for v in r.split("-"))
            keep &= ~((starts.valid >= lo) & (starts.valid < hi))
        starts.restrict(keep)
    cu = gated_curriculum(starts, CurriculumConfig(**(env_cfg.curriculum or cu_cfg), seed=a.seed), a.finish_gate)
    hidden = tuple(int(x) for x in a.hidden.split(","))
    from ballrain.env import BallanceEnv
    from ballrain.fakegame import FakeLauncher
    probe = BallanceEnv(env_cfg, launcher=FakeLauncher()) if a.fake else None
    n_obs = probe.observation_space.shape[0] if probe else None
    if probe is None:
        from ballrain.features import FeatureExtractor
        n_obs = FeatureExtractor(env_cfg.features).size
    n_act = 9
    model = ActorCritic(n_obs, n_act, hidden, a.activation).to(a.device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, eps=1e-5)
    rms = RunningMeanStd(n_obs)
    counters = {"update": 0, "steps": 0, "frames": 0, "episodes": 0, "runs": 0, "wall_s": 0.0, "finish_runs": 0,
                "dropped_chunks": 0, "worker_restarts": 0}
    best = {"sr_ms_k0": None, "sr_ms_any": None}
    def load_policy(src: dict) -> bool:
        """Model + observation statistics from a checkpoint; a checkpoint with fewer observation entries (the f6
        look-ahead block appended at the end) is extended with zero input weights and neutral statistics (mean 0,
        var 1), so the policy and value are unchanged at the switch. Returns True when it was extended."""
        sd, n_old = src["model"], int(src["n_obs"])
        st = src["rms"]
        if n_old == n_obs:
            model.load_state_dict(sd)
            rms.load_state_dict(st)
            return False
        if n_old > n_obs:
            raise SystemExit(f"checkpoint has {n_old} observation entries, this config {n_obs}")
        sd = dict(sd)
        for k in ("pi.0.weight", "v.0.weight"):
            w = sd[k]
            sd[k] = torch.cat([w, torch.zeros(w.shape[0], n_obs - n_old, dtype=w.dtype)], 1)
        model.load_state_dict(sd)
        rms.load_state_dict({"mean": np.r_[np.asarray(st["mean"]), np.zeros(n_obs - n_old)],
                             "var": np.r_[np.asarray(st["var"]), np.ones(n_obs - n_old)], "count": st["count"]})
        log.msg(f"policy extended from {n_old} to {n_obs} observation entries (zero input weights)")
        return True

    if ck is not None:
        extended = load_policy(ck)
        if not extended:
            opt.load_state_dict(ck["opt"])
        for g in opt.param_groups:
            g["lr"] = a.lr
        cu.load_state_dict(ck["curriculum"])
        counters.update(ck["counters"])
        best.update(ck.get("best", {}))
        log.msg(f"resumed from {ckpt_path} at update {counters['update']}, frontier {cu.frontier}")
    elif a.init_from:
        ck0 = torch.load(a.init_from, weights_only=False)
        extended = load_policy(ck0)
        if a.init_full:
            if not extended:
                opt.load_state_dict(ck0["opt"])
            for g in opt.param_groups:
                g["lr"] = a.lr
            cu.load_state_dict(ck0["curriculum"])
            counters.update(ck0["counters"])
            best.update(ck0.get("best", {}))
        log.msg(f"initialised from {a.init_from}" + (f" (full: update {counters['update']}, frontier {cu.frontier})"
                                                     if a.init_full else ""))
    # the command line is authoritative for the curriculum parameters (a resumed state carries the old ones)
    for k, v in (("step", a.cu_step), ("band", a.cu_band), ("recent", a.cu_recent), ("advance_at", a.advance_at),
                 ("min_episodes", a.min_episodes), ("weighted", a.cu_weighted), ("bin_floor", a.cu_bin_floor),
                 ("bin_frames", a.cu_bin_frames)):
        setattr(cu.cfg, k, v)
    cu.outcomes = deque(cu.outcomes, maxlen=max(cu.cfg.min_episodes, 1))
    cu.finishes = deque(cu.finishes, maxlen=max(cu.cfg.min_episodes, 1))
    if a.set_frontier is not None:
        old = cu.frontier
        cu.frontier = cu._snap(a.set_frontier)
        cu.outcomes.clear()
        cu.finishes.clear()
        log.msg(f"curriculum: frontier set {old} -> {cu.frontier} (--set-frontier)")
    if a.fake:
        names = [f"fake-{i:02d}" for i in range(a.workers)]
    elif a.backend == "headless":
        names = [f"hl-{a.copy_offset + i:02d}" for i in range(a.workers)]
    else:
        names = instances.training_copy_names(a.copy_offset + a.workers)[a.copy_offset:]
        for n in names:
            if instances.running_in_copy(instances.get_copy(n)):
                raise SystemExit(f"Player.exe already running in {n}")

    ctx = mp.get_context("spawn")
    data_q = ctx.Queue()
    ctrl_qs, procs, specs = [], [], []

    def spec_for(i, restart=0):
        import copy as _c
        cfg = _c.deepcopy(env_cfg)
        cfg.copy = names[i] if not a.fake else instances.TRAINING_COPY  # headless: hl-NN names the instance
        seed = a.seed * 1000 + i + 100 * restart + 7919 * counters["update"]  # a resume draws new seeds
        cfg.seed = seed
        if cfg.curriculum is not None:
            cfg.curriculum = dict(cfg.curriculum, seed=seed)
        if a.fake:
            cfg.out_root = str(run_dir / "episodes" / names[i])
        return WorkerSpec(wid=i, copy=names[i], env=cfg, seed=seed, chunk=a.chunk, hidden=hidden,
                          activation=a.activation, cap_mult=a.cap_mult, stall_s=a.stall_s,
                          respawn_stall_s=a.respawn_stall_s, stall_terminal=a.stall_terminal, prune=a.prune,
                          stall_extra=a.stall_extra, stall_compensate=a.stall_compensate, gamma=a.gamma,
                          stall_speed=a.stall_speed, stall_max_s=a.stall_max_s,
                          trace_sample=a.trace_sample, route_probes=json.loads(a.route_probes) if a.route_probes else {},
                          gates=json.loads(a.gates) if a.gates else {}, split_k=a.split_k,
                          split_min_ref_s=a.split_min_ref_s, split_extra=json.loads(a.split_extra) if a.split_extra else [],
                          p_early_respawn=a.p_early_respawn, zones=json.loads(a.zones) if a.zones else {},
                          wall_push_rate=a.wall_push_rate, wall_min_rise=a.wall_min_rise,
                          wall_max_progress=a.wall_max_progress, launcher_factory=FakeLauncher if a.fake else None)

    hub = None
    cu_bins = defaultdict(lambda: deque(maxlen=100))
    by_rest = defaultdict(lambda: deque(maxlen=200))  # at-rest starts by start sector  # curriculum bin -> recent LevelFinish outcomes (failure-weighted starts)

    def broadcast(qs=None, with_model=True):
        if cu.cfg.weighted:
            cu.set_bin_fail({b: 1.0 - float(np.mean(v)) for b, v in cu_bins.items() if len(v) >= 10})
        msg = {"version": counters["update"], "rms": rms.state_dict(), "curriculum": cu.state_dict()}
        if with_model:
            msg["model"] = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
        for q in (qs or ctrl_qs):
            q.put(msg)
        if hub is not None and qs is None and with_model:
            hub.publish(msg)

    def start_worker(i, restart=0):
        s = spec_for(i, restart)
        q = ctx.Queue()
        p = ctx.Process(target=worker_main, args=(s, q, data_q), daemon=True, name=f"ballrain-w{i}")
        p.start()
        return s, q, p

    if a.remote_listen:
        import copy as _c
        from ballrain.train.remote import RemoteHub, parse_addr
        key = (a.remote_key or os.environ.get("BALLRAIN_REMOTE_KEY") or "").encode()
        if not key:
            raise SystemExit("--remote-listen needs --remote-key or $BALLRAIN_REMOTE_KEY")
        tcfg = _c.deepcopy(env_cfg)
        tcfg.backend, tcfg.headless, tcfg.out_root = a.remote_backend, None, None
        template = {"env": tcfg, "seed": a.seed, "run": a.run,
                    "worker": {"chunk": a.chunk, "hidden": hidden, "activation": a.activation, "cap_mult": a.cap_mult,
                               "stall_s": a.stall_s, "respawn_stall_s": a.respawn_stall_s,
                               "stall_terminal": a.stall_terminal, "stall_extra": a.stall_extra,
                               "stall_compensate": a.stall_compensate, "gamma": a.gamma, "prune": a.prune,
                               "stall_speed": a.stall_speed, "stall_max_s": a.stall_max_s,
                               "trace_sample": a.trace_sample,
                               "route_probes": json.loads(a.route_probes) if a.route_probes else {},
                               "gates": json.loads(a.gates) if a.gates else {}, "split_k": a.split_k,
                               "split_min_ref_s": a.split_min_ref_s,
                               "split_extra": json.loads(a.split_extra) if a.split_extra else [],
                               "p_early_respawn": a.p_early_respawn,
                               "zones": json.loads(a.zones) if a.zones else {}, "wall_push_rate": a.wall_push_rate,
                               "wall_min_rise": a.wall_min_rise, "wall_max_progress": a.wall_max_progress}}
        hub = RemoteHub(parse_addr(a.remote_listen), key, data_q, template, log=log.msg)
        log.msg(f"remote hub listening on {a.remote_listen} (remote backend {a.remote_backend})")
    for i in range(a.workers):
        s, q, p = start_worker(i)
        specs.append(s), ctrl_qs.append(q), procs.append(p)
    (run_dir / "pids.json").write_text(json.dumps({"learner": os.getpid(), "workers": [p.pid for p in procs],
                                                   "copies": names}), encoding="utf-8")
    log.msg(f"run {a.run}: {a.workers} workers on {names[0] if names else '-'}..{names[-1] if names else '-'}, obs {n_obs}, seed {a.seed}, "
            f"frontier {cu.frontier}, device {a.device}")
    broadcast()

    # rolling statistics
    win_eps: list = []
    win_runs: list = []
    by_bin = defaultdict(lambda: deque(maxlen=200))  # start-frame bin -> (success_cp, finish)
    by_sector = defaultdict(lambda: deque(maxlen=200))
    by_rsp_sector = defaultdict(lambda: deque(maxlen=200))  # respawn lives (at rest at the checkpoint): reached the next one
    restarts = Counter()
    t_start = time.time()
    t_last_log = time.time()
    steps_last_log = counters["steps"]
    frames_last_log = counters["frames"]
    stop = False
    update_times = deque(maxlen=50)
    remote_last = [time.time()]  # last remote chunk (watchdog)
    remote_chunks = [0]
    remote_warned = [False]
    last_be_steps: dict = {}
    collect_times = deque(maxlen=50)
    reset_frac = deque(maxlen=200)

    def handle_episode(e):
        e["update"], e["t"] = counters["update"], round(time.time() - t_start + counters["wall_s"], 1)
        e["agent_s"] = (e["sr_ms_end"] - e["start_sr_ms"]) / 1000.0
        if e["s_start"] is not None and e["s_end"] is not None and e["agent_s"] > 0:
            e["speed"] = (e["s_end"] - e["s_start"]) / e["agent_s"]
        for k in TERMS:
            e[f"r_{k}"] = e["terms"].get(k, 0.0)
        counters["episodes"] += 1
        win_eps.append(e)
        log.row("episodes", e)
        if e["kind"] == "respawn":
            by_rsp_sector[e["start_sector"]].append((e["success_cp"], e["finish"]))
        if e["kind"] == "tas_rest":
            by_rest[e["start_sector"]].append((e["success_cp"], e["finish"]))
        if e["kind"] == "tas":
            k = e["start_frame"]
            succ = e["finish"] if a.success == "finish" else e["success_cp"]
            moved = cu.update_gated(k, bool(succ), bool(e["finish"]))
            if moved:
                log.msg(f"curriculum: frontier -> {cu.frontier} (sector {int(starts.sector[starts.index_of(cu.frontier)])})"
                        f" after rate {cu.history[-1][1]:.2f}")
            if cu.at_start and cu.cfg.p_level_start != a.p_level_start:
                cu.cfg.p_level_start = a.p_level_start
            b = 0 if k == 0 else 1 + (k - 936) // 2000
            by_bin[b].append((e["success_cp"], e["finish"]))
            if k:
                cu_bins[cu.bin_of(k)].append(bool(e["finish"]))
            by_sector[e["start_sector"]].append((e["success_cp"], e["finish"]))

    def handle_run(r):
        r["update"], r["t"] = counters["update"], round(time.time() - t_start + counters["wall_s"], 1)
        counters["runs"] += 1
        log.row("runs", r)
        if r.get("invalid"):  # e.g. camera_off_quarter: never a result
            counters["invalid_runs"] = counters.get("invalid_runs", 0) + 1
            log.msg(f"invalid run ({r['invalid']}): {r['dir']}")
            return
        win_runs.append(r)
        if r["end_reason"] == "LevelFinish" and r["sr_ms"] is not None:
            counters["finish_runs"] += 1
            key = "sr_ms_k0" if not r["start_frame"] else "sr_ms_any"
            if best[key] is None or r["sr_ms"] < best[key]:
                best[key] = r["sr_ms"]
                if key == "sr_ms_k0":
                    log.msg(f"new best k=0 finish: SR {r['sr_ms']} ms, deaths {r['deaths']}, {r['dir']}")
            if not r["start_frame"] and counters.get("first_k0_finish") is None:
                counters["first_k0_finish"] = counters["update"]
                log.msg(f"MILESTONE first closed-loop finish from level load: SR {r['sr_ms']} ms ({r['dir']})")
        if counters.get("first_finish") is None and r["end_reason"] == "LevelFinish":
            counters["first_finish"] = counters["update"]
            log.msg(f"MILESTONE first finish: start frame {r['start_frame']}, SR {r['sr_ms']}")

    def save_ckpt():
        state = {"model": model.state_dict(), "opt": opt.state_dict(), "rms": rms.state_dict(),
                 "curriculum": cu.state_dict(), "counters": dict(counters, wall_s=counters["wall_s"] + time.time() - t_start),
                 "best": best, "args": vars(a), "n_obs": n_obs, "n_act": n_act, "env_config": asdict(env_cfg)}
        f = run_dir / f"ckpt_{counters['update']:06d}.pt"
        torch.save(state, f)
        tmp = run_dir / "latest.pt.tmp"
        torch.save(state, tmp)
        os.replace(tmp, ckpt_path)
        old = sorted(run_dir.glob("ckpt_*.pt"))
        for o in old[:-5]:
            u = int(o.stem.split("_")[1])
            if u % 250 != 0:
                o.unlink(missing_ok=True)

    try:
        first = ck is None and a.init_from is None
        while counters["steps"] < a.total_steps and not stop:
            chunks = []
            n = 0
            tc = time.perf_counter()
            while n < a.batch_steps:
                if (run_dir / "STOP").is_file():
                    stop = True
                    break
                try:
                    m = data_q.get(timeout=30)
                except queue.Empty:
                    dead = [i for i, p in enumerate(procs) if not p.is_alive()]
                    if dead:
                        log.msg(f"workers not alive: {dead}")
                    continue
                if "fatal" in m and m.get("remote"):
                    log.msg(f"remote worker {m['wid']} fatal: {m['fatal'][-800:]}")
                    continue
                if "fatal" in m:
                    i = m["wid"]
                    restarts[i] += 1
                    counters["worker_restarts"] += 1
                    log.msg(f"worker {i} fatal (restart {restarts[i]}): {m['fatal'][-1500:]}")
                    procs[i].join(timeout=30)
                    if not a.fake and a.backend == "retail":
                        stop_copies([names[i]], log.msg)
                    if restarts[i] > 20:
                        raise RuntimeError(f"worker {i} keeps failing")
                    s, q, p = start_worker(i, restarts[i])
                    specs[i], ctrl_qs[i], procs[i] = s, q, p
                    broadcast([q])
                    continue
                for e in m.get("errors", []):
                    log.msg(f"worker {m['wid']} error: {e[-800:]}")
                for r in m.get("runs", []):
                    handle_run(r)
                if "obs" not in m:
                    continue
                be = m.get("backend", a.backend)
                for e in m["episodes"]:
                    e["wid"] = m["wid"]
                    e["backend"] = be
                    handle_episode(e)
                counters[f"steps_{be}"] = counters.get(f"steps_{be}", 0) + len(m["rewards"])
                if m.get("remote"):
                    remote_last[0] = time.time()
                    remote_chunks[0] += 1
                counters["frames"] += m["frames"]
                reset_frac.append(m["t_reset"] / max(m["t_chunk"], 1e-9))
                if m["raw"].shape[1] != n_obs:  # a worker still on an older observation layout (remote, before its restart)
                    counters["dropped_chunks"] += 1
                    continue
                if counters["update"] - m["version"] > (a.max_stale_remote if m.get("remote") else a.max_stale):
                    counters["dropped_chunks"] += 1
                    counters[f"dropped_{be}"] = counters.get(f"dropped_{be}", 0) + 1
                    counters[f"dropped_steps_{be}"] = counters.get(f"dropped_steps_{be}", 0) + len(m["rewards"])
                    continue
                chunks.append(m)
                n += len(m["rewards"])
            if stop or not chunks:
                break
            collect_times.append(time.perf_counter() - tc)
            raw = np.concatenate([c["raw"] for c in chunks])
            rms.update(raw)
            counters["steps"] += len(raw)
            tu = time.perf_counter()
            if first:  # the first batch only fits the observation statistics
                first = False
                ustats = {}
            else:
                advs, rets = zip(*(gae(c, a.gamma, a.lam, a.reward_scale) for c in chunks))
                batch = {"obs": normalize(raw, rms.state_dict()), "actions": np.concatenate([c["actions"] for c in chunks]),
                         "logp": np.concatenate([c["logp"] for c in chunks]), "adv": np.concatenate(advs),
                         "ret": np.concatenate(rets)}
                model.train()
                ustats = ppo_update(model, opt, batch, a.epochs, a.minibatch, a.clip, a.vf_coef, a.ent_coef,
                                    a.max_grad_norm, a.device, gen, a.target_kl)
                model.eval()
                ustats["explained_var"] = explained_variance(np.concatenate([c["values"] for c in chunks]), batch["ret"])
                ustats["adv_std"] = float(batch["adv"].std())
                ustats["ret_mean"] = float(batch["ret"].mean())
            update_times.append(time.perf_counter() - tu)
            counters["update"] += 1
            broadcast()
            u = counters["update"]
            for k, v in ustats.items():
                log.tb.add_scalar(f"loss/{k}", v, counters["steps"])
            if u % a.log_every == 0:
                now = time.time()
                dt = now - t_last_log
                sps = (counters["steps"] - steps_last_log) / dt
                fps = (counters["frames"] - frames_last_log) / dt
                t_last_log, steps_last_log, frames_last_log = now, counters["steps"], counters["frames"]
                eps = win_eps
                tas_eps = [e for e in eps if e["kind"] == "tas"]
                row = {"update": u, "steps": counters["steps"], "frames": counters["frames"],
                       "wall_s": round(counters["wall_s"] + now - t_start, 1), "steps_per_s": round(sps, 1),
                       "frames_per_s": round(fps, 1), "frontier": cu.frontier,
                       "frontier_sector": int(starts.sector[starts.index_of(cu.frontier)]),
                       "band_success": float(np.mean(cu.outcomes)) if cu.outcomes else float("nan"),
                       "band_n": len(cu.outcomes), "episodes": len(eps),
                       "band_finish": float(np.mean(cu.finishes)) if cu.finishes else float("nan"),
                       "gate_blocked": cu.gate_blocked,
                       "ret_mean": float(np.mean([e["ret"] for e in eps])) if eps else float("nan"),
                       "ret_tas": float(np.mean([e["ret"] for e in tas_eps])) if tas_eps else float("nan"),
                       "ret_respawn": float(np.mean([e["ret"] for e in eps if e["kind"] == "respawn"])) if any(e["kind"] == "respawn" for e in eps) else float("nan"),
                       "success_cp_tas": float(np.mean([e["success_cp"] for e in tas_eps])) if tas_eps else float("nan"),
                       "finish_tas": float(np.mean([e["finish"] for e in tas_eps])) if tas_eps else float("nan"),
                       "deaths": sum(e["end"] == "BallOff" for e in eps), "finishes": sum(e["finish"] for e in eps),
                       "stalls": sum(e["end"] == "stall" for e in eps), "caps": sum(e["end"] == "cap" for e in eps),
                       "camera_off_quarter": sum(e["end"] == "camera_off_quarter" for e in eps),
                       "invalid_runs": counters.get("invalid_runs", 0),
                       "speed_mean": float(np.nanmean([e.get("speed", np.nan) for e in eps])) if eps else float("nan"),
                       "runs": len(win_runs), "finish_runs": sum(r["end_reason"] == "LevelFinish" for r in win_runs),
                       "sr_ms_min": min([r["sr_ms"] for r in win_runs if r["end_reason"] == "LevelFinish" and r["sr_ms"]], default=None),
                       "sr_ms_k0_min": min([r["sr_ms"] for r in win_runs if r["end_reason"] == "LevelFinish" and r["sr_ms"] and not r["start_frame"]], default=None),
                       "k_p10": float(np.percentile([e["start_frame"] for e in tas_eps], 10)) if tas_eps else None,
                       "k_p50": float(np.percentile([e["start_frame"] for e in tas_eps], 50)) if tas_eps else None,
                       "k0_eps": sum(e["start_frame"] == 0 for e in tas_eps),
                       "update_s": round(float(np.mean(update_times)), 3), "collect_s": round(float(np.mean(collect_times)), 3),
                       "reset_frac": round(float(np.mean(reset_frac)), 3) if reset_frac else None,
                       "dropped_chunks": counters["dropped_chunks"], "worker_restarts": counters["worker_restarts"],
                       "best_sr_k0": best["sr_ms_k0"], "best_sr_any": best["sr_ms_any"]}
                for s in range(1, 5):
                    d = [e for e in eps if e["end"] == "BallOff" and e["end_sector"] == s]
                    row[f"deaths_s{s}"] = len(d)
                for b in range(0, 12):
                    if by_bin.get(b):
                        row[f"cp_bin{b}"] = round(float(np.mean([x[0] for x in by_bin[b]])), 3)
                        row[f"fin_bin{b}"] = round(float(np.mean([x[1] for x in by_bin[b]])), 3)
                for s in range(1, 5):
                    if by_sector.get(s):
                        row[f"cp_sec{s}"] = round(float(np.mean([x[0] for x in by_sector[s]])), 3)
                        row[f"fin_sec{s}"] = round(float(np.mean([x[1] for x in by_sector[s]])), 3)
                for s in range(1, 5):
                    if by_rest.get(s):
                        row[f"rest_cp_sec{s}"] = round(float(np.mean([x[0] for x in by_rest[s]])), 3)
                        row[f"rest_n_sec{s}"] = len(by_rest[s])
                    if by_rsp_sector.get(s):
                        row[f"rsp_cp_sec{s}"] = round(float(np.mean([x[0] for x in by_rsp_sector[s]])), 3)
                        row[f"rsp_n_sec{s}"] = len(by_rsp_sector[s])
                row.update({f"u_{k}": round(v, 5) for k, v in ustats.items()})
                log.row("updates", row, cols=UPDATE_COLS + [c for c in row if c not in UPDATE_COLS])
                for k, v in row.items():
                    if isinstance(v, (int, float)) and v is not None and np.isfinite(v):
                        log.tb.add_scalar(f"train/{k}", v, counters["steps"])
                n_skip = sum(bool(e.get("respawn_skipped")) for e in eps)
                log.tb.add_scalar("train/respawn_skipped", n_skip, counters["steps"])
                for be in sorted({e.get("backend", a.backend) for e in eps}):
                    be_eps = [e for e in eps if e.get("backend", a.backend) == be]
                    be_tas = [e for e in be_eps if e["kind"] == "tas"]
                    sk = f"steps_{be}"
                    be_sps = (counters.get(sk, 0) - last_be_steps.get(be, 0)) / dt
                    last_be_steps[be] = counters.get(sk, 0)
                    log.tb.add_scalar(f"train_{be}/steps_per_s", be_sps, counters["steps"])
                    log.tb.add_scalar(f"train_{be}/episodes", len(be_eps), counters["steps"])
                    log.tb.add_scalar(f"train_{be}/ret_mean", float(np.mean([e["ret"] for e in be_eps])), counters["steps"])
                    if be_tas:
                        log.tb.add_scalar(f"train_{be}/success_cp_tas", float(np.mean([e["success_cp"] for e in be_tas])),
                                          counters["steps"])
                        log.tb.add_scalar(f"train_{be}/finish_tas", float(np.mean([e["finish"] for e in be_tas])),
                                          counters["steps"])
                    log.tb.add_scalar(f"train_{be}/deaths", sum(e["end"] == "BallOff" for e in be_eps), counters["steps"])
                    log.tb.add_scalar(f"train_{be}/dropped_chunks", counters.get(f"dropped_{be}", 0), counters["steps"])
                    log.tb.add_scalar(f"train_{be}/dropped_step_frac",
                                      counters.get(f"dropped_steps_{be}", 0) / max(counters.get(sk, 0), 1), counters["steps"])
                if a.split_k:
                    sp_rows = [x for e in eps for x in (e.get("splits") or [])]
                    if sp_rows:
                        log.tb.add_scalar("split/ratio_mean", float(np.mean([x[1] / max(x[2], 1e-3) for x in sp_rows])),
                                          counters["steps"])
                        log.tb.add_scalar("split/bonus_mean", float(np.mean([x[3] for x in sp_rows])), counters["steps"])
                if a.gates:
                    for name in json.loads(a.gates):
                        hits = sum(f"gate:{name}" in (e.get("probes") or []) for e in eps)
                        log.tb.add_scalar(f"gate/{name}_hits", hits, counters["steps"])
                        log.tb.add_scalar(f"gate/{name}_hit_rate", hits / max(len(eps), 1), counters["steps"])
                if a.zones or a.wall_push_rate:
                    # TensorBoard only (a new updates.csv column would start updates.<i>.csv); per episode in episodes*.csv
                    n_zone = sum(int(e.get("zone_entries") or 0) for e in eps)
                    log.tb.add_scalar("zone/entries", n_zone, counters["steps"])
                    log.tb.add_scalar("zone/entries_per_episode", n_zone / max(len(eps), 1), counters["steps"])
                    log.tb.add_scalar("zone/wall_push", float(sum((e.get("terms") or {}).get("wall_push", 0.0) for e in eps)),
                                      counters["steps"])
                if a.route_probes:
                    for name in json.loads(a.route_probes):
                        vis = [e for e in eps if name in (e.get("probes") or [])]
                        log.tb.add_scalar(f"route/{name}_visits", len(vis), counters["steps"])
                        log.tb.add_scalar(f"route/{name}_deaths", sum(e.get("death_probe") == name for e in eps),
                                          counters["steps"])
                        row[f"route_{name}_visits"] = len(vis)
                        row[f"route_{name}_deaths"] = sum(e.get("death_probe") == name for e in eps)
                if hub is not None:
                    idle = time.time() - remote_last[0]
                    log.tb.add_scalar("remote/alive", float(idle < 600), counters["steps"])
                    log.tb.add_scalar("remote/chunks_per_min", remote_chunks[0] * 60.0 / max(dt, 1e-9), counters["steps"])
                    remote_chunks[0] = 0
                    if idle >= 600 and not remote_warned[0]:
                        log.msg(f"WARNING remote: no remote chunks for {idle / 60:.0f} min (bridge down?)")
                        remote_warned[0] = True
                    elif idle < 600 and remote_warned[0]:
                        log.msg("remote: chunks arriving again")
                        remote_warned[0] = False
                    log.tb.add_scalar("remote/connections", len([c for c in hub.conns.values()]), counters["steps"])
                    log.tb.add_scalar("remote/chunks", hub.stats["chunks"], counters["steps"])
                    log.tb.add_scalar("remote/disconnects", hub.stats["disconnects"], counters["steps"])
                if tas_eps:
                    log.tb.add_histogram("curriculum/start_frame", np.array([e["start_frame"] for e in tas_eps]),
                                         counters["steps"])
                log.msg(f"u{u} steps {counters['steps']} sps {sps:.0f} frontier {cu.frontier} band {row['band_success']:.2f}"
                        f"/{row['band_n']} eps {len(eps)} cp {row['success_cp_tas']:.2f} fin {row['finish_tas']:.2f} "
                        f"ret {row['ret_mean']:.2f} deaths {row['deaths']} stalls {row['stalls']} fin_runs {row['finish_runs']} "
                        f"sr_min {row['sr_ms_min']} ent {ustats.get('entropy', float('nan')):.3f} "
                        f"kl {ustats.get('approx_kl', float('nan')):.4f} ev {ustats.get('explained_var', float('nan')):.2f} "
                        f"upd {row['update_s']}s reset {row['reset_frac']} rsp_skip {n_skip}")
                log.flush()
                win_eps, win_runs = [], []
            if u % a.ckpt_every == 0:
                save_ckpt()
    finally:
        log.msg("stopping workers")
        if hub is not None:
            hub.close()
        for q in ctrl_qs:
            try:
                q.put({"cmd": "stop"})
            except Exception:  # noqa: BLE001
                pass
        deadline = time.time() + 120
        closed = set()
        while time.time() < deadline and len(closed) < len(procs):
            try:
                m = data_q.get(timeout=5)
            except queue.Empty:
                if all(not p.is_alive() for p in procs):
                    break
                continue
            for r in m.get("runs", []):
                try:
                    handle_run(r)
                except Exception:  # noqa: BLE001
                    pass
            if m.get("closed") or "fatal" in m:
                closed.add(m["wid"])
        for p in procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        if not a.fake and a.backend == "retail":
            st = stop_copies(names, log.msg)
            if st:
                log.msg(f"stopped leftover game processes: {st}")
        try:
            save_ckpt()
        except Exception as e:  # noqa: BLE001
            log.msg(f"final checkpoint failed: {e}")
        log.msg(f"stopped at update {counters['update']}, steps {counters['steps']}")
        log.flush()
        for q in ctrl_qs + [data_q]:  # unread messages (e.g. to dead workers) must not block the exit
            q.cancel_join_thread()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

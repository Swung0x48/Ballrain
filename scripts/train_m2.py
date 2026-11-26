"""M2 training: PPO on level 1 (SR) with a reverse curriculum of TAS start states (launches the game).

    uv run python scripts/train_m2.py --run <name> [--workers 12] [--total-steps 50e6] ...
    uv run python scripts/train_m2.py --run <name> --resume            # continue from <run>/latest.pt
    uv run python scripts/train_m2.py --run smoke --fake --workers 2 --total-steps 3000   # no game (FakeLauncher)

Learner in this process, one rollout worker process per game copy (ballrain.train.worker; copies
train-<copy_offset> .. ). Outputs in <runtime>/m2-train/<run>/: config.json, updates.csv (one row per update),
episodes.csv (one row per RL episode), runs.csv (one row per game process), tb/ (tensorboard), ckpt_*.pt and
latest.pt (model, optimiser, observation statistics, curriculum state, counters), episodes/<copy>/<run_id>/ (kept
runs: finishing processes, .json + .tas), pids.json, train.log, nonfinite-u<update>.npz (the batch of a skipped
non-finite update, ballrain.train.nonfinite; at most 5), anchor.csv (--anchor-mastered: one row per update; the
mastered-state anchor, ballrain.train.anchor; its family statistics and teacher snapshots are in the checkpoints).

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
RECORDS = {1: RECORD, 2: "SR02_1_28_992_264"}  # default start record per level (--record overrides)
TERMS = ("progress", "time", "finish", "death", "death_refund", "shaping", "checkpoint", "stall_penalty", "gate_bonus", "split_bonus",
         "camera_off_quarter", "zone_penalty", "wall_push", "pickup", "pickup_followers_finish", "pickup_death_life", "pickup_shaping")
EP_COLS = ["update", "t", "wid", "backend", "host", "copy", "kind", "probes", "death_probe", "splits", "start_frame", "start_sector", "end", "end_sector", "max_sector",
           "success_cp", "finish", "ret", "steps", "frames", "sr_ms_end", "start_sr_ms", "agent_s", "s_start", "s_end",
           "speed", "jumps", "cap_frames", "version_start", "new_process", "run_id", "ball_order", "respawn_skipped", "zone_entries", "zones"] + [f"r_{k}" for k in TERMS] + ["actions", "end_x", "end_y", "end_z", "points_end", "lives_end", "pickups_missed"]
UPDATE_COLS = (["update", "steps", "frames", "wall_s", "steps_per_s", "frames_per_s", "frontier", "frontier_sector",
                "band_success", "band_n", "band_finish", "gate_blocked", "episodes", "ret_mean", "ret_tas", "ret_respawn", "success_cp_tas", "finish_tas",
                "deaths", "finishes", "stalls", "caps", "camera_off_quarter", "invalid_runs", "speed_mean", "runs", "finish_runs", "sr_ms_min", "sr_ms_k0_min",
                "k_p10", "k_p50", "k0_eps", "update_s", "collect_s", "reset_frac", "dropped_chunks", "worker_restarts",
                "nonfinite_skipped", "best_sr_k0", "best_sr_any"] + [f"deaths_s{s}" for s in range(1, 5)]
               + [f"{p}_bin{b}" for b in range(12) for p in ("cp", "fin")]
               + [f"{p}_sec{s}" for s in range(1, 5) for p in ("cp", "fin")]
               + [f"{p}_sec{s}" for s in range(1, 5) for p in ("rsp_cp", "rsp_n")]
               + [f"{p}_sec{s}" for s in range(1, 5) for p in ("rest_cp", "rest_n")]
               + [f"u_{k}" for k in ("pg_loss", "v_loss", "entropy", "approx_kl", "clip_frac", "early_stop",
                                     "explained_var", "adv_std", "ret_mean", "grad_norm", "watch_entropy", "watch_n",
                                     "teacher_kl", "teacher_n", "teacher_coef")])
ANCHOR_COLS = ["update", "steps", "families", "mastered", "in_grace", "teachers", "batch_families", "rows", "tagged_rows",
               "kl_pre", "kl", "n", "coef", "terms_s", "ppo_s", "n_mastered", "n_released", "n_retired", "n_dropped",
               "top"]
RUN_COLS = ["update", "t", "copy", "run_id", "end_reason", "sr_ms", "start_frame", "frames", "lives", "deaths",
            "framelog_match", "handover_mismatches", "invalid", "ball_order", "dir"]


def start_record(a) -> str:
    """The TAS start record of the run: --record, else the level's default (RECORDS)."""
    if a.record:
        return a.record
    if a.level not in RECORDS:
        raise SystemExit(f"--level {a.level}: no default start record, give --record")
    return RECORDS[a.level]


def parse_rest_starts(spec: str):
    """--rest-starts: "lo-hi", "lo-hi@x:z:jitter" or "lo-hi@x:z:jitter:max_frames" entries -> [[lo, hi]] /
    [[lo, hi, x, z, jitter]] / [[lo, hi, x, z, jitter, max_frames]] (None when empty).
    Branch forms carry a target dict; k-k@branch-move:<member>:<station> requires an exact pinned sample.
    """
    out = []
    for r in spec.split(","):
        r = r.strip()
        if not r:
            continue
        rng, _, tgt = r.partition("@")
        if tgt.startswith("branch-move:"):
            import math
            import re
            name, sep, station = tgt.removeprefix("branch-move:").rpartition(":")
            try:
                e = [int(v) for v in rng.split("-")]
                value = float(station)
                if (len(e) != 2 or not 0 < e[0] == e[1] or not sep or
                        not re.fullmatch(r"[a-z][a-z0-9-]{0,31}", name) or not math.isfinite(value) or value < 0):
                    raise ValueError
            except ValueError:
                raise SystemExit(f"--rest-starts entry {r!r}: k-k@branch-move:<member>:<recorded station>") from None
            out.append([*e, {"branch_s": value, "id": name, "branch_move": True}])
            continue
        e = [int(v) for v in rng.split("-")]
        if tgt.startswith("branch:"):
            name, sep, station = tgt.removeprefix("branch:").rpartition(":")
            e.append({"branch_s": float(station), **({"id": name} if sep else {})})
        elif tgt:
            e += [float(v) for v in tgt.split(":")]
            if len(e) not in (5, 6):
                raise SystemExit(f"--rest-starts entry {r!r}: lo-hi@x:z:jitter[:max_frames]")
            if len(e) == 6:
                e[5] = int(e[5])
        out.append(e)
    return out or None


def gated_curriculum(starts, cfg, finish_gate: float, *, valid=None):
    """ReverseCurriculum whose frontier also needs the LevelFinish rate of the band episodes to reach finish_gate:
    with --success cp a frontier in an early sector only has to reach the next checkpoint, and the gate keeps it from
    running ahead of whole-run competence (the band episodes continue to the end of the level after the checkpoint)."""
    from ballrain.curriculum import ReverseCurriculum

    class Gated(ReverseCurriculum):
        def __init__(self):
            super().__init__(starts, cfg, valid=valid)
            self.finishes = deque(maxlen=max(cfg.min_episodes, 1))
            self.finish_gate = finish_gate
            self.gate_blocked = 0
            self.floor = 0  # --frontier-floor: the frontier never moves below this frame (0 = no floor)

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
                    new = self.next_frontier(max(self.frontier - self.cfg.step, self.floor))
                    if new >= self.frontier or new < self.floor:  # at the floor: no move
                        self.outcomes.clear()
                        self.finishes.clear()
                        return False
                    self.frontier = new
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
    return build_parser().parse_args(argv)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--level", type=int)
    p.add_argument("--record", default="", help="TAS start record (tas-v1 dataset; default: RECORDS[level])")
    p.add_argument("--rehearse-behind-frontier", action="store_true",
                   help="rehearsal starts only from the part of each range at or after the curriculum frontier "
                        "(EnvConfig.rehearse_behind_frontier): a section is rehearsed once the frontier has passed it")
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
    p.add_argument("--branch-update", action="store_true", help="explicitly replace the checkpoint branch pin")
    p.add_argument("--branch-dir", help="paired ball-type diversion assets (main line unchanged)")
    p.add_argument("--path-dir", default=None, help="directory of the reference line assets (FeatureConfig.path_dir)")
    p.add_argument("--gates", default="", help="JSON sub-goal gates (ballrain.train.worker.WorkerSpec.gates)")
    p.add_argument("--zones", default="",
                   help='JSON penalty zones {name: {"box": [xmin, xmax, zmin, zmax], "y_below": y, "penalty": p, '
                        '"wall_push": true}} (WorkerSpec.zones: one-off penalty per entry, the life goes on; '
                        '"wall_push": the wall push applies inside this zone)')
    p.add_argument("--wall-push-rate", type=float, default=0.0,
                   help="penalty per game second while pushing into an unclimbable wall (WorkerSpec.wall_push_rate; 0 = off); "
                        'only inside zones with "wall_push": true unless --wall-push-everywhere')
    p.add_argument("--wall-push-everywhere", action="store_true",
                   help="wall push on the whole route (the behaviour before zone-limited wall push)")
    p.add_argument("--zone-stall-exempt", action="store_true",
                   help="a life that entered a penalty zone pays no stall penalty when it ends by stall (WorkerSpec.zone_stall_exempt)")
    p.add_argument("--wall-min-rise", type=float, default=2.0, help="wall height above the ball's lowest point that counts as unclimbable")
    p.add_argument("--wall-max-progress", type=float, default=2.0, help="wall push: only below this route progress (units / s)")
    p.add_argument("--route-probes", default="", help='JSON {name: [xmin, xmax, zmin, zmax]}: per-route visit / death stats')
    p.add_argument("--remote-listen", default=None,
                   help="HOST:PORT: accept remote rollout workers (scripts/remote_worker_m2.py, ballrain.train.remote)")
    p.add_argument("--remote-key", default=None, help="shared secret (default $BALLRAIN_REMOTE_KEY)")
    p.add_argument("--remote-backend", default="retail", help="backend of the remote workers' env")
    p.add_argument("--max-stale-remote", type=int, default=2, help="max_stale for remote chunks")
    p.add_argument("--pool-ingest", action="store_true",
                   help="train on volunteer-pool chunks (host 'pool' on the remote hub: ballrain.pool.learner_link); "
                        "off by default: pool chunks are dropped and counted (pool_refused)")
    p.add_argument("--pool-max-lag", type=int, default=300,
                   help="drop pool chunks whose policy is more than this many updates old (pool_dropped)")
    p.add_argument("--pool-correction", choices=("vtrace", "none"), default="vtrace",
                   help="vtrace: V-trace targets / advantages with the current critic and PPO clipping around the "
                        "current policy (ppo.vtrace); none: like a remote chunk (GAE on the behaviour values, ratio to "
                        "the behaviour policy; for comparison runs)")
    p.add_argument("--pool-rho-bar", type=float, default=1.0, help="V-trace importance-weight truncation (rho bar)")
    p.add_argument("--pool-c-bar", type=float, default=1.0, help="V-trace trace truncation (c bar)")
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
    p.add_argument("--actor-heads", choices=("single", "ball_type"), default=None,
                   help="actor routing (default: checkpoint layout on resume, otherwise single)")
    p.add_argument("--activation", default="tanh")
    p.add_argument("--torch-threads", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--action-repeat", type=int, default=4)
    p.add_argument("--fps", type=float, default=None,
                   help="game frames per second the env injects (delta_ms = f32(1000 / fps)); default: the resumed or "
                        "--init-from checkpoint's, else 132. 264 = the HS01 TAS's own rate (its keys reproduce the record "
                        "at action_repeat 2 there, not at 132 fps)")
    p.add_argument("--hs-block", action="store_true", help="keep the HS feature block (off for SR)")
    p.add_argument("--value-warmup", type=int, default=0,
                   help="the first N updates of this process train the value head only (the policy is frozen): after a reward change, e.g. --init-from an SR policy for HS")
    p.add_argument("--pickups", action="store_true",
                   help="the pickup block (FeatureConfig.pickups: nearest score / life balls still present in the sector, "
                        "after every other entry; a checkpoint without it is grown with zero input weights)")
    p.add_argument("--pickups-success", action="store_true",
                   help="for the failure-weighted starts a finish that left a score / life ball behind is a failure")
    p.add_argument("--auto-collect", action="store_true",
                   help="the harness collects small score balls (Esc, wait, Esc) after any step that leaves some following (EnvConfig.auto_collect); with --reward pickup_weight for HS training without an Esc action")
    p.add_argument("--far-waypoints", default="", help="f6: line points this far ahead in arc units, e.g. 20,30,45")
    p.add_argument("--ref-speed-at", default="", help="f6: reference speed at these arc offsets, e.g. 0,15,30")
    p.add_argument("--reach-mechanisms", action="store_true",
                   help="opt into retail mechanism layers for f7; Python workers only, requires --reach")
    p.add_argument("--reach", action="store_true",
                   help="f7: append the fastreach reach block (FeatureConfig.reach, ballrain.reach_features; 59 entries) "
                        "after every other observation entry; a checkpoint without it is grown with zero input weights "
                        "(--resume / --init-from). Sticky: a run resumed from a reach checkpoint keeps it. Needs numba")
    p.add_argument("--contacts", action="store_true",
                   help="f8: append the contact / rail blocks (FeatureConfig(version='f8'): top-4 merged contacts, "
                        "aggregates, left / right rails; 49 entries) after the reach block; an f7 checkpoint is grown "
                        "with zero input weights (--resume / --init-from). Sticky like --reach. Workers on older code "
                        "cannot unpickle the f8 template and refuse it")
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
    p.add_argument("--rest-starts", default="", help="at-rest or moving branch starts: lo-hi record frame ranges, e.g. 11760-11900,13450-13650; "
                   "lo-hi@branch:<id>:s uses a measured branch drive (lo-hi@branch:s for legacy); lo-hi@x:z:jitter drives the ball to a target in [x +- jitter] x [z +- jitter] before braking "
                   "(lo-hi@x:z:jitter:max_frames: that entry's drive limit; the hand-over may be an excluded frame)")
    p.add_argument("--counter-rand", default="",
                   help='counter randomisation on TAS / at-rest starts (WorkerSpec.counter_rand), JSON, e.g. '
                   '{"lives": {"values": [0.0, 0.1, 0.2, 0.3], "p": [0.2, 0.2, 0.2, 0.4]}, "sr_s": [0, 0.3], '
                   '"points": [-0.05, 0.45]}')
    p.add_argument("--respawn-level-starts", action="store_true",
                   help="play every respawn of a process started from level load (WorkerSpec.respawn_level_starts)")
    p.add_argument("--p-rest", type=float, default=0.0, help="probability of an at-rest start for a new process")
    p.add_argument("--rehearse-starts", default="", help="rehearsal start ranges lo-hi (record frames), e.g. 4900-5075")
    p.add_argument("--p-rehearse", type=float, default=0.0, help="probability of a rehearsal start for a new process")
    p.add_argument("--rehearse-always", default="",
                   help="rehearsal ranges lo-hi that --rehearse-behind-frontier never clips (EnvConfig.rehearse_always): "
                        "drawn like --rehearse-starts entries, also behind the frontier")
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
    p.add_argument("--ckpt-keep-every", type=int, default=250,
                   help="besides the newest 5, keep the checkpoints of updates divisible by this (rollback targets)")
    p.add_argument("--auto-rollback", default="",
                   help="JSON ballrain.train.guard.ReachGuard fields, e.g. {\"lo\": 4100, \"hi\": 4299, \"bin\": 100, "
                        "\"min_n\": 20, \"low\": 0.5, \"good\": 0.7}: reload the newest good bin's checkpoint when the "
                        "TAS starts in [lo, hi] reach the next checkpoint at < low in consecutive bins of policy versions")
    p.add_argument("--frontier-floor", type=int, default=0,
                   help="the curriculum frontier never moves below this TAS frame (0 = no floor)")
    p.add_argument("--rail-gate", default="",
                   help='JSON {"s": route arc length, "below": TAS frame}: truncate (bootstrapped, end gate_truncate) a 灏忚妭-1 '
                        'life not started at or after "below" once it reaches route s (WorkerSpec.rail_gate)')
    p.add_argument("--demo", default="",
                   help="demonstration set (scripts/build_demo.py: states of a TAS record with its actions): the PPO loss "
                        "gets + coef x cross-entropy of the policy against the demonstrated actions (imitation beside RL)")
    p.add_argument("--demo-coef", type=float, default=0.0, help="weight of the imitation term at the start of the process")
    p.add_argument("--demo-coef-floor", type=float, default=0.0, help="weight after --demo-decay-updates")
    p.add_argument("--demo-decay-updates", type=int, default=0,
                   help="linear decay of the imitation weight to the floor over this many updates of the process (0 = constant)")
    p.add_argument("--profile-sample", type=float, default=0.0,
                   help="sample every thread's Python stack every 10 ms for this many seconds after the start and write "
                        "the counts to <run>/profile.json (ballrain.train.stack_sampler; 0 = off)")
    p.add_argument("--sil-coef", type=float, default=0.0,
                   help="self-imitation (ballrain.train.sil): the PPO loss gets + coef x the policy's own better-than-"
                        "expected actions (batch rows with advantage above --sil-adv-min std, kept while their return "
                        "still beats the value head), weighted by clamp((R - V) / adv_std, 0, --sil-wmax); 0 = off")
    p.add_argument("--sil-capacity", type=int, default=65536)
    p.add_argument("--sil-adv-min", type=float, default=2.0, help="advantage (batch std) a row needs to enter the buffer")
    p.add_argument("--sil-keep-min", type=float, default=0.1, help="(R - V) / adv_std a buffer row needs to stay")
    p.add_argument("--sil-confirm", choices=("checkpoint", "horizon"), default="checkpoint",
                   help="when a candidate row enters the buffer: its life reaches the next checkpoint (or finishes the "
                        "level) / its life goes on --sil-horizon-s without ending; a life that ends first drops it")
    p.add_argument("--sil-horizon-s", type=float, default=8.0,
                   help="a candidate row enters the buffer only after its life went on this many game seconds without "
                        "ending (or finished the level within them); a death, stall or cut-off first drops it")
    p.add_argument("--sil-batch", type=int, default=256)
    p.add_argument("--sil-vcoef", type=float, default=0.01)
    p.add_argument("--sil-wmax", type=float, default=5.0)
    p.add_argument("--teacher", default="", help="checkpoint of a frozen teacher policy: the PPO loss gets + coef x "
                   "KL(teacher || policy) on batch states whose route s (path_progress x line length) is >= --teacher-s-min")
    p.add_argument("--teacher-allow-context-append", action="store_true",
                   help="allow byte-preserving moving metadata additions or an unchanged teacher beyond an appended ordinary-only branch span")
    p.add_argument("--teacher-coef", type=float, default=0.1)
    p.add_argument("--teacher-coef-floor", type=float, default=None,
                   help="decay the coefficient linearly to this floor over --teacher-decay-updates (default: constant)")
    p.add_argument("--teacher-decay-updates", type=int, default=0)
    p.add_argument("--teacher-s-min", type=float, default=331.4,
                   help="route arc length from which the teacher term applies (L1 refpaths-alt: 331.4 = TAS frame 4460)")
    p.add_argument("--anchor-mastered", action="store_true",
                   help="mastered-state anchor (ballrain.train.anchor): + coef x KL(teacher || policy) on the batch rows of "
                        "lives that started in a start family (--rest-starts entry, rehearsal range, curriculum band, respawn "
                        "sector) whose rolling success (the learner's: next checkpoint, or finish with --success finish) "
                        "is >= --anchor-threshold; a mean over families of the mean KL within each. Off: unchanged")
    p.add_argument("--anchor-teacher", default="",
                   help="with --anchor-mastered: one explicit teacher checkpoint for every family (the --teacher layout "
                        "rules); default: per family the snapshot of the policy that mastered it")
    p.add_argument("--anchor-coef", type=float, default=0.1,
                   help="coefficient of the anchor term (a mean over families: each family's pull is coef / families "
                        "present, not comparable to --teacher-coef, a mean over rows)")
    p.add_argument("--anchor-window", type=int, default=40, help="lives per family in the rolling success")
    p.add_argument("--anchor-min-lives", type=int, default=40, help="lives in the window before a family can be mastered")
    p.add_argument("--anchor-threshold", type=float, default=0.9, help="rolling success at which a family is mastered")
    p.add_argument("--anchor-grace", type=int, default=80,
                   help="lives in a row below the threshold before a mastered family is released (hysteresis)")
    p.add_argument("--anchor-drop-scale", type=float, default=0.0,
                   help="family weight 1 + this x (peak rate - current rate)+: a slipping family is pulled harder (0 = equal)")
    p.add_argument("--anchor-families", default="rest,rehearse,band,respawn,level",
                   help="start family kinds that can be anchored (ballrain.train.anchor.KINDS)")
    p.add_argument("--anchor-band-frames", type=int, default=0,
                   help="record frames per curriculum band family (0 = --cu-bin-frames)")
    p.add_argument("--anchor-min-rows", type=int, default=16,
                   help="a family with fewer rows in the batch gets rows / this of a full share")
    p.add_argument("--anchor-max-teachers", type=int, default=8, help="policy snapshots kept (automatic teachers)")
    p.add_argument("--anchor-rows", choices=("life", "first-cp"), default="life",
                   help="rows of a life in a mastered family that are anchored: life = all of them (also after it reached "
                        "later checkpoints); first-cp = only those before it reached its next checkpoint (the stretch "
                        "its success measures; with --success finish both are the whole life)")
    p.add_argument("--anchor-only", default="",
                   help="comma-separated family keys (e.g. rehearse:4000-4450): only these families are anchored when "
                        "mastered; the others are still counted (anchor.csv) but never pulled. Empty: every family")
    p.add_argument("--anchor-ratchet", action="store_true",
                   help="automatic teachers: a family whose last --anchor-window lives were all played by policies newer "
                        "than its snapshot, at >= the rate it had then, moves to a newer snapshot (follows improvements)")
    p.add_argument("--watch-s", default="",
                   help="LO,HI: also log the behaviour policy's entropy on batch states whose route position (feature "
                        "path_progress x line length) is in [LO, HI] (u_watch_entropy, u_watch_n; L1 r6: 300,400, the exit)")
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--prune", default="finish")
    p.add_argument("--trace-sample", type=float, default=0.0,
                   help="fraction of game processes traced and kept for export spot-checks (scripts/sim2real_m2.py)")
    return p


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


def saved_args(a) -> dict:
    """vars(a) as config.json / the checkpoint store them: without the --anchor-* options while --anchor-mastered is off
    (a run that does not use the anchor writes exactly what it wrote before the option existed)."""
    d = dict(vars(a))
    if not d.get("anchor_mastered"):
        d = {k: v for k, v in d.items() if not k.startswith("anchor_")}
    return d


def checkpoint_config(env):
    ec = asdict(env)
    return {"branch_assets": ec.pop("branch_assets"), "env_config": ec}


def obs_layout(fc):
    """(n_obs, {block: slice}) of a FeatureConfig from the layout alone (no FeatureExtractor, no numba)."""
    from ballrain.features import layout_size, layout_slices
    return layout_size(fc), layout_slices(fc)


def make_env_config(a, out_root: Path, ck=None, init=None):
    """ck: the checkpoint resumed from (--resume); init: the one a new run starts from (--init-from), whose route
    branch bundle the new run keeps unless --branch-dir names one (it must match that checkpoint's pin)."""
    from ballrain.branch_assets import read_bundle, uses_entry_context, validate
    from ballrain.path import RouteError
    from ballrain.env import EnvConfig
    from ballrain.features import FeatureConfig
    from ballrain.rewards import RewardConfig
    rc = RewardConfig(mode="sr", **json.loads(a.reward))
    fl = lambda x: tuple(float(v) for v in x.split(",") if v.strip()) if x else ()  # noqa: E731
    from ballrain.path import asset_path, file_sha256, portable_path_dir, resolve_path_dir
    # repo-relative, so remote workers on another machine resolve it in their own checkout, plus the asset's sha256
    # (a worker whose copy of the line differs refuses to run: ballrain.path.RouteError)
    path_dir = portable_path_dir(a.path_dir)
    asset = asset_path(a.level, resolve_path_dir(path_dir))
    if path_dir is not None and not asset.is_file():
        raise SystemExit(f"--path-dir {a.path_dir}: no reference line {asset}")
    main_sha = file_sha256(asset) if asset.is_file() else None
    saved = (ck or {}).get("branch_assets")
    old = (ck or {}).get("env_config", {}).get("features", {})
    if saved is None and (old.get("branch_sha256") or (old.get("branch_dir") and not a.branch_dir)):
        raise RouteError("checkpoint lacks branch assets (legacy runs require explicit --branch-dir)")
    bundle = read_bundle(a.level, a.branch_dir) if a.branch_dir else saved
    if ck is None and init is not None and init.get("branch_assets") is not None and not a.branch_dir:
        # --init-from a branch run: the new run follows the same bundle (before, it silently dropped the branch)
        bundle = init["branch_assets"]
        want = ((init.get("env_config") or {}).get("features") or {}).get("branch_sha256")
        if validate(bundle, a.level, main_sha) != want:
            raise RouteError("--init-from checkpoint: its branch assets do not match its own pin")
    pin = validate(bundle, a.level, main_sha) if bundle is not None else None
    if getattr(a, "pool_ingest", False) and uses_entry_context(bundle):
        raise RouteError("contextual branch entrances are Python-only; disable --pool-ingest")
    update = getattr(a, "branch_update", False)
    if update and (not a.branch_dir or saved is None):
        raise RouteError("--branch-update needs --branch-dir and an existing checkpoint bundle")
    if saved is not None and (old.get("branch_sha256") != pin or bundle != saved):
        if not update or validate(saved, a.level, main_sha) != old.get("branch_sha256"):
            raise RouteError("branch assets differ from checkpoint pin")
        print(f"branch-update: {old['branch_sha256']} -> {pin}", flush=True)
    from ballrain.features import CONTACT_FEATURE_VERSION, FEATURE_VERSION
    fc = FeatureConfig(hs_block=a.hs_block, camera="world" if a.world_actions else "orient_ref",
                       far_waypoints=fl(a.far_waypoints), ref_speed_at=fl(a.ref_speed_at), path_dir=path_dir,
                       branch_sha256=pin, path_sha256=main_sha, reach=bool(getattr(a, "reach", False)),
                       version=CONTACT_FEATURE_VERSION if getattr(a, "contacts", False) else FEATURE_VERSION,
                       pickups=bool(getattr(a, "pickups", False)))
    from ballrain.reach_features import is_mechanism_version, reach_version
    old_reach = old.get("reach_version")
    saved_mechanisms = is_mechanism_version(old_reach)
    if getattr(a, "reach_mechanisms", False) or saved_mechanisms:
        if not fc.reach:
            raise RouteError("--reach-mechanisms requires --reach")
        if getattr(a, "pool_ingest", False):
            raise RouteError("reach mechanisms require Python workers; disable --pool-ingest")
        if not 1 <= a.level <= 13:
            raise RouteError("no retail reach mechanism asset for this level")
        fc.reach_version = reach_version(mechanisms=True)
        if saved_mechanisms and old_reach != fc.reach_version:
            raise RouteError("reach mechanism assets/code differ from the checkpoint pin")
    cu = {"start_back": a.start_back, "step": a.cu_step, "band": a.cu_band, "recent": a.cu_recent,
          "advance_at": a.advance_at, "min_episodes": a.min_episodes, "p_level_start": a.p_level_start}
    hl = None
    if a.backend == "headless":
        hl = {k: v for k, v in (("binary", a.hl_bin), ("root", a.hl_root), ("socket", a.hl_socket),
                                ("ball_order", a.hl_ball_order)) if v}
    rest_starts = parse_rest_starts(a.rest_starts)
    if getattr(a, "pool_ingest", False) and any(len(e) == 3 and isinstance(e[2], dict) and e[2].get("branch_move")
                                               for e in rest_starts or ()):
        raise RouteError("moving branch starts are Python-only; disable --pool-ingest")
    if getattr(a, "fps", None):
        delta_ms = float(np.float32(1000.0 / a.fps))
    else:  # keep the frame rate the checkpoint was trained at
        delta_ms = float(((ck or init or {}).get("env_config") or {}).get("delta_ms") or EnvConfig.delta_ms)
    return EnvConfig(branch_assets=bundle, level=a.level, mode="sr", action_repeat=a.action_repeat, out_root=str(out_root),
                     delta_ms=delta_ms,
                     start_record=start_record(a),
                     backend=a.backend, headless=hl,
                     curriculum=cu, features=fc, reward=rc, max_process_frames=200000, world_actions=a.world_actions,
                     auto_collect=a.auto_collect,
                     per_life=not a.no_respawn,
                     rest_starts=rest_starts,
                     p_rest=a.p_rest,
                     rehearse_starts=[[int(v) for v in r.split("-")] for r in a.rehearse_starts.split(",") if r.strip()] or None,
                     p_rehearse=a.p_rehearse,
                     rehearse_always=[[int(v) for v in r.split("-")] for r in a.rehearse_always.split(",") if r.strip()] or None,
                     rehearse_behind_frontier=a.rehearse_behind_frontier,
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
    from ballrain.train.nonfinite import NonFiniteError, NonFiniteGuard, nonfinite_keys
    from ballrain.train.ppo import ActorCritic, RunningMeanStd, explained_variance, gae, normalize, vtrace
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
        a.reach = bool(a.reach or saved.get("reach"))  # on stays on (the observation never shrinks); off -> on grows it
        a.contacts = bool(a.contacts or saved.get("contacts"))  # likewise for the f8 contact blocks
        a.auto_collect = bool(a.auto_collect or saved.get("auto_collect"))  # the harness keeps collecting on resume
        a.pickups = bool(a.pickups or saved.get("pickups"))  # on stays on (the observation never shrinks)
        from ballrain.reach_features import is_mechanism_version
        a.reach_mechanisms = a.reach_mechanisms or is_mechanism_version(saved.get("reach_version"))
    else:
        ck = None
    if a.actor_heads is None:
        a.actor_heads = (ck or {}).get("args", {}).get("actor_heads", "single")
    if a.level is None:
        a.level = (ck or {}).get("env_config", {}).get("level", 1)
    if ck and ck.get("branch_assets") and a.path_dir is None:
        a.path_dir = ck["env_config"]["features"].get("path_dir")
    # --init-from: loaded once (its branch bundle carries over into the new run's config, its policy is loaded below)
    ck0 = torch.load(a.init_from, weights_only=False) if (ck is None and a.init_from) else None
    if ck0 is not None and ck0.get("branch_assets") and a.path_dir is None and not a.branch_dir:
        a.path_dir = ck0["env_config"]["features"].get("path_dir")
    (run_dir / ("config.json" if not a.resume else f"config.resume-{time.strftime('%Y%m%d-%H%M%S')}.json")).write_text(
        json.dumps(saved_args(a), indent=1), encoding="utf-8")
    log = Logger(run_dir, a.resume)
    log.tb.add_text("config", "    " + json.dumps(saved_args(a), indent=1).replace("\n", "\n    "),
                    global_step=int(ck["counters"]["steps"]) if ck is not None else 0)
    torch.set_num_threads(a.torch_threads)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    gen = torch.Generator().manual_seed(a.seed)

    env_cfg = make_env_config(a, run_dir / "episodes", ck, init=ck0)
    if getattr(a, "branch_update", False):
        change = {"from": ck["env_config"]["features"]["branch_sha256"], "to": env_cfg.features.branch_sha256}
        with (run_dir / "branch-updates.jsonl").open("a", encoding="utf8") as audit:
            audit.write(json.dumps(change) + "\n")
    if a.fake:  # the toy level has no TAS start states: plain starts from level load
        env_cfg.start_record = None
        cu_cfg = env_cfg.curriculum
        env_cfg.curriculum = None
    starts = TasStarts(start_record(a))
    if a.exclude_starts:
        keep = np.ones(len(starts.valid), bool)
        for r in a.exclude_starts.split(","):
            lo, hi = (int(v) for v in r.split("-"))
            keep &= ~((starts.valid >= lo) & (starts.valid < hi))
        starts.restrict(keep)
    from ballrain.branch_assets import ordinary_start_frames
    ordinary_valid = ordinary_start_frames(starts, env_cfg.branch_assets)
    if len(ordinary_valid) != len(starts.valid):
        log.msg(f"ordinary branch filter: removed {len(starts.valid)-len(ordinary_valid)} prefixes; drills unchanged")
    cu = gated_curriculum(starts, CurriculumConfig(**(env_cfg.curriculum or cu_cfg), seed=a.seed), a.finish_gate,
                          valid=ordinary_valid)
    cu.floor = int(a.frontier_floor or 0)
    hidden = tuple(int(x) for x in a.hidden.split(","))
    from ballrain.env import BallanceEnv
    from ballrain.fakegame import FakeLauncher
    # the observation size from the layout alone: the learner never builds a FeatureExtractor (with the f7 block that
    # would need numba and the fields); --fake runs its toy env here (and its local workers), which does
    n_obs = obs_layout(env_cfg.features)[0]
    probe = BallanceEnv(env_cfg, launcher=FakeLauncher()) if a.fake else None
    if probe is not None and probe.observation_space.shape[0] != n_obs:
        raise SystemExit(f"layout says {n_obs} observation entries, the fake env has "
                         f"{probe.observation_space.shape[0]}")
    n_act = 9
    model = ActorCritic(n_obs, n_act, hidden, a.activation, a.actor_heads).to(a.device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, eps=1e-5)
    rms = RunningMeanStd(n_obs)
    counters = {"update": 0, "steps": 0, "frames": 0, "episodes": 0, "runs": 0, "wall_s": 0.0, "finish_runs": 0,
                "dropped_chunks": 0, "worker_restarts": 0, "nonfinite_skipped": 0}
    best = {"sr_ms_k0": None, "sr_ms_any": None}
    from ballrain.train.grow import GrowError, load_grown
    new_features = asdict(env_cfg)["features"]

    def load_policy(src: dict, load_opt: bool) -> bool:
        """Model, observation statistics and (load_opt) optimiser state from a checkpoint; a checkpoint with fewer
        observation entries (a block appended at the end: f6 look-ahead, f7 reach) is grown with zero input weights,
        zero Adam moments for the new columns and neutral statistics (mean 0, var 1), so the policy and value are
        unchanged at the switch (ballrain.train.grow). Returns True when it was grown."""
        try:
            return load_grown(model, opt, rms, src, n_obs, new_features, load_opt=load_opt, log=log.msg)
        except GrowError as e:
            raise SystemExit(str(e))

    if ck is not None:
        load_policy(ck, load_opt=True)
        rv_old = ((ck.get("env_config") or {}).get("features") or {}).get("reach_version")
        if rv_old and rv_old != env_cfg.features.reach_version:
            log.msg(f"reach block version {rv_old} -> {env_cfg.features.reach_version}: the policy continues on the "
                    "current block (FeatureConfig.reach_version pins it for every worker)")
        for g in opt.param_groups:
            g["lr"] = a.lr
        cu.load_state_dict(ck["curriculum"])
        counters.update(ck["counters"])
        best.update(ck.get("best", {}))
        log.msg(f"resumed from {ckpt_path} at update {counters['update']}, frontier {cu.frontier}")
    elif a.init_from:
        load_policy(ck0, load_opt=a.init_full)
        if a.init_full:
            for g in opt.param_groups:
                g["lr"] = a.lr
            cu.load_state_dict(ck0["curriculum"])
            counters.update(ck0["counters"])
            best.update(ck0.get("best", {}))
        log.msg(f"initialised from {a.init_from}" + (f" (full: update {counters['update']}, frontier {cu.frontier})"
                                                     if a.init_full else ""))

    def learner_state() -> dict:
        return {"model": model.state_dict(), "opt": opt.state_dict(), "rms": rms.state_dict()}

    def require_finite(what: str, state: dict):
        """The last line of defence of the non-finite guard: nothing non-finite is saved or published."""
        bad = nonfinite_keys(state)
        if bad:
            raise NonFiniteError(f"refusing to {what} non-finite parameters ({', '.join(bad[:5])}"
                                 f"{', ...' if len(bad) > 5 else ''})")

    try:  # before any worker starts: a checkpoint written by a learner without the guard may hold NaN
        require_finite("start from", learner_state())
    except NonFiniteError as e:
        raise SystemExit(f"{e}: --resume / --init-from an earlier checkpoint")
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
    data_q = ctx.Queue()  # local worker processes -> learner
    # The main loop reads in_q only: the local workers' chunks are moved over from data_q by a thread, and the remote
    # hub puts its chunks straight into in_q (before 2026-11-26 the hub put them into data_q, which pickled each
    # received chunk again in the feeder thread and unpickled it once more in the main loop).
    in_q: queue.Queue = queue.Queue()

    def _forward_local():
        while True:
            try:
                in_q.put(data_q.get())
            except (EOFError, OSError, ValueError):
                return

    import threading as _threading
    _threading.Thread(target=_forward_local, daemon=True, name="local-forward").start()
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
                          activation=a.activation, actor_heads=a.actor_heads, cap_mult=a.cap_mult, stall_s=a.stall_s,
                          respawn_stall_s=a.respawn_stall_s, stall_terminal=a.stall_terminal, prune=a.prune,
                          stall_extra=a.stall_extra, stall_compensate=a.stall_compensate, gamma=a.gamma,
                          stall_speed=a.stall_speed, stall_max_s=a.stall_max_s,
                          trace_sample=a.trace_sample, route_probes=json.loads(a.route_probes) if a.route_probes else {},
                          gates=json.loads(a.gates) if a.gates else {}, split_k=a.split_k,
                          split_min_ref_s=a.split_min_ref_s, split_extra=json.loads(a.split_extra) if a.split_extra else [],
                          p_early_respawn=a.p_early_respawn, zones=json.loads(a.zones) if a.zones else {},
                          wall_push_rate=a.wall_push_rate, wall_min_rise=a.wall_min_rise,
                          wall_max_progress=a.wall_max_progress, wall_push_everywhere=a.wall_push_everywhere,
                          zone_stall_exempt=a.zone_stall_exempt, rail_gate=json.loads(a.rail_gate) if a.rail_gate else {},
                          respawn_level_starts=a.respawn_level_starts,
                          counter_rand=json.loads(a.counter_rand) if a.counter_rand else {},
                          launcher_factory=FakeLauncher if a.fake else None,
                          **({"start_tags": True} if a.anchor_mastered else {}))

    hub = None
    cu_bins = defaultdict(lambda: deque(maxlen=100))
    by_rest = defaultdict(lambda: deque(maxlen=200))  # at-rest starts by start sector  # curriculum bin -> recent LevelFinish outcomes (failure-weighted starts)

    def broadcast(qs=None, with_model=True):
        if cu.cfg.weighted:
            cu.set_bin_fail({b: 1.0 - float(np.mean(v)) for b, v in cu_bins.items() if len(v) >= 10})
        msg = {"version": counters["update"], "rms": rms.state_dict(), "curriculum": cu.state_dict()}
        if with_model:
            msg["model"] = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
        require_finite("publish", {k: msg[k] for k in ("model", "rms") if k in msg})
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
        # pool_ingest: the pool's learner link connects only to a learner that advertises pool support (older code
        # would train pool chunks as plain remote chunks) and sends chunks only when it is true
        template = {"env": tcfg, "seed": a.seed, "run": a.run, "pool_ingest": bool(a.pool_ingest),
                    "pool_max_lag": int(a.pool_max_lag),
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
                               "wall_min_rise": a.wall_min_rise, "wall_max_progress": a.wall_max_progress,
                               "wall_push_everywhere": a.wall_push_everywhere, "zone_stall_exempt": a.zone_stall_exempt,
                               "rail_gate": json.loads(a.rail_gate) if a.rail_gate else {},
                               "respawn_level_starts": a.respawn_level_starts,
                               "counter_rand": json.loads(a.counter_rand) if a.counter_rand else {}}}
        if a.actor_heads != "single":
            template["worker"]["actor_heads"] = a.actor_heads
        if a.anchor_mastered:  # (only then: the template, and its id, stay as before with the anchor off)
            template["worker"]["start_tags"] = True
        if getattr(env_cfg, "branch_assets", None):
            # the same raw bytes the checkpoint holds ({level, line_npz, desc_json, drive_npz, sha256}); the pool's
            # learner link carries them into the exported policy state
            template["branch_assets"] = env_cfg.branch_assets
        hub = RemoteHub(parse_addr(a.remote_listen), key, in_q, template, log=log.msg)
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
    updates_this_process = 0  # --value-warmup counts these
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
        if e["kind"] in ("branch_moving", "branch_moving_fallback"):
            log.row("branch-moving", e, cols=["update", "wid", "kind", "start_frame", "branch_move_member",
                    "branch_move_station", "branch_move_offset", "branch_move_phase", "branch_rest_failures",
                    "branch_rest_error", "success_cp", "finish"])
        if e["kind"] == "respawn":
            by_rsp_sector[e["start_sector"]].append((e["success_cp"], e["finish"]))
        if e["kind"] == "tas_rest":
            by_rest[e["start_sector"]].append((e["success_cp"], e["finish"]))
        if guard is not None:
            guard.add(e["kind"], e["start_frame"], e.get("version_start"), e["success_cp"])
        if anchor is not None and e.get("start_tag") is not None and e.get("end") not in AN.INVALID_ENDS:
            fam = anchor_key(e["start_tag"])  # lives of workers without start tags (older code) are not counted
            if fam is not None:
                anchor.add_life(fam, bool(e["finish"] if a.success == "finish" else e["success_cp"]),
                                e.get("version_start"), counters["update"])
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
                # a finish that left a score / life ball behind counts as a failure of its bin (HS: failure-weighted
                # starts then go where balls are missed, not only where lives end)
                cu_bins[cu.bin_of(k)].append(bool(e["finish"]) and not (a.pickups_success and e.get("pickups_missed")))
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
        require_finite("save", learner_state())  # latest.pt stays the last finite checkpoint
        state = {"model": model.state_dict(), "opt": opt.state_dict(), "rms": rms.state_dict(),
                 "curriculum": cu.state_dict(), "counters": dict(counters, wall_s=counters["wall_s"] + time.time() - t_start),
                 "best": best, "args": saved_args(a), "n_obs": n_obs, "n_act": n_act, **checkpoint_config(env_cfg)}
        if anchor is not None:  # family statistics and teacher snapshots (--resume continues them)
            state["anchor"] = anchor.state_dict()
        f = run_dir / f"ckpt_{counters['update']:06d}.pt"
        torch.save(state, f)
        tmp = run_dir / "latest.pt.tmp"
        torch.save(state, tmp)
        os.replace(tmp, ckpt_path)
        old = sorted(run_dir.glob("ckpt_*.pt"))
        for o in old[:-5]:
            u = int(o.stem.split("_")[1])
            if u % a.ckpt_keep_every != 0:
                o.unlink(missing_ok=True)

    guard = None
    if a.auto_rollback:
        from ballrain.train.guard import ReachGuard
        guard = ReachGuard.from_config(json.loads(a.auto_rollback))
        if guard.bin % a.ckpt_every or a.ckpt_keep_every % a.ckpt_every or guard.bin % a.ckpt_keep_every:
            raise SystemExit("--auto-rollback: bin must be a multiple of --ckpt-keep-every (and both of --ckpt-every)")
        guard.rolled_back(counters["update"])  # bins of a previous process are not in this one's statistics
        log.msg(f"auto-rollback guard: TAS starts {guard.lo}-{guard.hi}, bins of {guard.bin} versions (>= {guard.min_n} "
                f"episodes), roll back after {guard.consecutive} bins < {guard.low} to the newest bin >= {guard.good}")

    pool_lags: list = []
    pool_warned = [False]
    pool_rho: list = []
    nf_guard = NonFiniteGuard(run_dir, log.msg, counters)

    def pool_correction(chunks, obs_n) -> dict:
        """--pool-correction vtrace: chunk index -> (adv, ret, logp_prox, v_prox) for the pool chunks of the batch: the
        current policy / critic (before this update) on their normalised observations, V-trace targets and advantages
        (ppo.vtrace), and the current log-probs as the batch's "old" log-probs (PPO clips around the current policy;
        the behaviour policy enters through vtrace's truncated weights)."""
        out, off = {}, 0
        idx = []
        for i, c in enumerate(chunks):
            T = len(c["rewards"])
            if c.get("pool"):
                idx.append((i, off, T))
            off += T
        if not idx:
            return out
        rs = rms.state_dict()
        with torch.no_grad():
            for i, o0, T in idx:
                c = chunks[i]
                x = torch.as_tensor(obs_n[o0:o0 + T], device=a.device)
                xl = torch.as_tensor(normalize(c["raw_last"][None].astype(np.float32), rs), device=a.device)
                logits, v = model(x)
                _, vl = model(xl)
                lp = torch.log_softmax(logits, -1).gather(-1, torch.as_tensor(c["actions"], device=a.device)[:, None])
                lp = lp.squeeze(-1).cpu().numpy().astype(np.float32)
                v = v.cpu().numpy().astype(np.float32)
                adv, ret, rho = vtrace(c, lp, v, float(vl.cpu().numpy()[0]), a.gamma, a.lam, a.reward_scale,
                                       a.pool_rho_bar, a.pool_c_bar)
                pool_rho.append(rho)
                out[i] = (adv, ret, lp, v)
        return out

    def pool_stats(chunks) -> dict:
        n = sum(len(c["rewards"]) for c in chunks if c.get("pool"))
        if not n:
            return {}
        st = {"pool_steps": float(n), "pool_frac": n / max(sum(len(c["rewards"]) for c in chunks), 1)}
        if pool_lags:
            st["pool_lag"] = float(np.mean(pool_lags))
            pool_lags.clear()
        if pool_rho:
            r = np.concatenate(pool_rho)
            st["pool_rho"] = float(r.mean())
            st["pool_rho_clip"] = float(np.mean(r >= a.pool_rho_bar - 1e-6))
            pool_rho.clear()
        return st

    watch = None
    if a.watch_s:
        from ballrain.path import ReferencePath, asset_path, resolve_path_dir
        lo_s, hi_s = (float(v) for v in a.watch_s.split(","))
        sl = obs_layout(env_cfg.features)[1].get("path_progress")
        asset = asset_path(env_cfg.level, resolve_path_dir(env_cfg.features.path_dir)) if env_cfg.features.path_dir else None
        if sl is None or asset is None or not asset.is_file():
            raise SystemExit("--watch-s needs the path_progress feature and a reference line asset (--path-dir)")
        length = ReferencePath.load(asset).length
        watch = (sl.start, lo_s / length, hi_s / length)
        log.msg(f"watch entropy on route s {lo_s:g}-{hi_s:g} (line length {length:.1f})")

    def branch_append_ignore(tck, what, s_min=None):
        """frozenset({"branch_sha256"}) when a teacher checkpoint's branch pin differs from this run's only by an
        append-only, context-gated member and/or additive moving metadata, both checked by contextual_append_end;
        with s_min (the position-based --teacher) the teacher's route mask must also start above the appended span."""
        mine, theirs = asdict(env_cfg)["features"], tck["env_config"]["features"]
        if not a.teacher_allow_context_append or mine.get("branch_sha256") == theirs.get("branch_sha256"):
            return frozenset()
        from ballrain.branch_assets import contextual_append_end
        from ballrain.path import RouteError
        try:
            end = contextual_append_end(tck.get("branch_assets"), env_cfg.branch_assets, env_cfg.level,
                                        mine.get("path_sha256"), theirs.get("branch_sha256"), mine.get("branch_sha256"))
            if end is not None and s_min is not None and not s_min > end:
                raise RouteError(f"teacher s-min must be above the appended main span ({end:.6f})")
        except RouteError as exc:
            raise SystemExit(f"{what} contextual append refused: {exc}") from None
        if end is None:
            log.msg(f"{what} branch compatibility: added moving metadata only; guidance unchanged")
        else:
            log.msg(f"{what} branch append: appended span ends at {end:.6f}"
                    + (f"; teacher s >= {s_min:g}" if s_min is not None else ""))
        return frozenset({"branch_sha256"})

    demo = None
    if a.demo:
        z = np.load(a.demo)
        if z["obs"].shape[1] != n_obs:
            raise SystemExit(f"--demo {a.demo}: {z['obs'].shape[1]} observation entries, the run has {n_obs}")
        from ballrain.features import make_layout
        dfc = json.loads(str(z["features"]))
        from dataclasses import fields as dc_fields
        known = {f.name for f in dc_fields(type(env_cfg.features))}
        if make_layout(type(env_cfg.features)(**{k: v for k, v in dfc.items() if k in known})) != make_layout(env_cfg.features):
            raise SystemExit(f"--demo {a.demo}: built with another observation layout")
        # a demo built at another frame rate labels the states with actions that do something else here
        d_ms = float(z["delta_ms"]) if "delta_ms" in z.files else float(np.float32(1000 / 132))
        if abs(d_ms - env_cfg.delta_ms) > 1e-4:
            raise SystemExit(f"--demo {a.demo}: built at delta_ms {d_ms:.4f}, the run steps {env_cfg.delta_ms:.4f}")
        demo = {"raw": z["obs"].astype(np.float32), "act": z["act"].astype(np.int64)}
        log.msg(f"imitation: {len(demo['act'])} demonstrated states from {a.demo} (ranges {z['ranges']}), coef "
                f"{a.demo_coef} -> {a.demo_coef_floor} over {a.demo_decay_updates} updates")

    sil, last_sil = None, {}
    if a.sil_coef > 0:
        from ballrain.train.sil import SILBuffer, value_fn as sil_value_fn
        sil_h = int(round(a.sil_horizon_s * 1000.0 / (env_cfg.delta_ms * a.action_repeat)))
        sil_col = None
        if a.sil_confirm == "checkpoint":
            from ballrain.features import make_layout
            o = 0
            for name, size in make_layout(env_cfg.features):
                if name == "sector":
                    sil_col = o
                    break
                o += size
            if sil_col is None:
                raise SystemExit("--sil-confirm checkpoint: the observation has no sector entry")
        sil = SILBuffer(a.sil_capacity, a.sil_adv_min, a.sil_keep_min, horizon=sil_h, sector_col=sil_col)
        log.msg(f"self-imitation: coef {a.sil_coef}, buffer {a.sil_capacity} rows (advantage > {a.sil_adv_min} std in, "
                f"kept while (R - V) / std > {a.sil_keep_min}; confirmed when "
                + (f"the same life reaches the next checkpoint or finishes (sector column {sil_col})" if sil_col is not None
                   else f"the same life goes on {a.sil_horizon_s} s = {sil_h} steps without a death / stall / cut-off")
                + f"), {a.sil_batch} rows per minibatch, w <= {a.sil_wmax}")

    def demo_coef_now() -> float:
        if not a.demo_decay_updates:
            return a.demo_coef
        f = min(1.0, updates_this_process / a.demo_decay_updates)
        return a.demo_coef + (a.demo_coef_floor - a.demo_coef) * f

    teacher = None
    if a.teacher:
        from ballrain.path import ReferencePath, asset_path, resolve_path_dir
        from ballrain.train.anchor import teacher_layout_error
        tck = torch.load(a.teacher, weights_only=False)
        # the teacher reads the first n_obs entries: a teacher without the f7 reach block / the f8 contact blocks
        # (appended at the end) or on the plain reach block under --reach-mechanisms is allowed (teacher_layout_error);
        # an append-only branch repin is allowed with --teacher-allow-context-append (branch_append_ignore)
        t_ignore = branch_append_ignore(tck, "--teacher", a.teacher_s_min)
        t_obs = int(tck["n_obs"])
        err = teacher_layout_error(asdict(env_cfg)["features"], tck["env_config"]["features"], t_obs, n_obs,
                                   int(tck["n_act"]), n_act, t_ignore)
        if err:
            raise SystemExit(f"--teacher {a.teacher}: {err}")
        ta = tck["args"]
        tmodel = ActorCritic(t_obs, n_act, tuple(int(x) for x in ta["hidden"].split(",")), ta["activation"],
                             ta.get("actor_heads", "single")).to(a.device)
        tmodel.load_state_dict(tck["model"])
        tmodel.eval()
        sl = obs_layout(env_cfg.features)[1]["path_progress"]
        length = ReferencePath.load(asset_path(env_cfg.level, resolve_path_dir(env_cfg.features.path_dir))).length
        teacher = {"model": tmodel, "rms": tck["rms"], "col": sl.start, "lo": a.teacher_s_min / length,
                   "u0": counters["update"], "n_obs": t_obs}
        log.msg(f"teacher {a.teacher} (update {tck['counters']['update']}): KL coef {a.teacher_coef}"
                + (f" -> {a.teacher_coef_floor} over {a.teacher_decay_updates} updates" if a.teacher_decay_updates else "")
                + f" on route s >= {a.teacher_s_min:g} (line length {length:.1f})")

    anchor, anchor_teachers, AN = None, {}, None
    if a.anchor_mastered:
        from ballrain.train import anchor as AN
        kinds = tuple(k.strip() for k in a.anchor_families.split(",") if k.strip())
        if not kinds or set(kinds) - set(AN.KINDS):
            raise SystemExit(f"--anchor-families {a.anchor_families}: kinds from {','.join(AN.KINDS)}")
        band_frames, base_frame = int(a.anchor_band_frames or a.cu_bin_frames), int(starts.valid[0])
        mine = asdict(env_cfg)["features"]
        arch = {"hidden": a.hidden, "activation": a.activation, "actor_heads": a.actor_heads}

        def anchor_key(tag):
            return AN.family_key(tag, band_frames=band_frames, base_frame=base_frame, kinds=kinds)

        anchor_only = {k.strip() for k in a.anchor_only.split(",") if k.strip()}

        anchor = AN.MasteredAnchor(window=a.anchor_window, min_lives=a.anchor_min_lives, threshold=a.anchor_threshold,
                                   grace=a.anchor_grace, drop_scale=a.anchor_drop_scale,
                                   max_teachers=a.anchor_max_teachers, explicit=bool(a.anchor_teacher),
                                   ratchet=a.anchor_ratchet)
        snapshot_sources = {}
        if ck is not None and ck.get("branch_assets") is not None:
            snapshot_sources[ck["env_config"]["features"].get("branch_sha256")] = ck["branch_assets"]
        if a.anchor_teacher:
            tck = torch.load(a.anchor_teacher, weights_only=False)
            if tck.get("branch_assets") is not None:
                snapshot_sources[tck["env_config"]["features"].get("branch_sha256")] = tck["branch_assets"]
            err = AN.teacher_layout_error(mine, tck["env_config"]["features"], int(tck["n_obs"]), n_obs,
                                          int(tck["n_act"]), n_act, branch_append_ignore(tck, "--anchor-teacher"))
            if err:
                raise SystemExit(f"--anchor-teacher {a.anchor_teacher}: {err}")
            ta = tck["args"]
            anchor_teachers["explicit"] = AN.build_teacher(
                {"actor": tck["model"], "rms": tck["rms"], "n_obs": int(tck["n_obs"]),
                 "arch": {"hidden": ta["hidden"], "activation": ta["activation"],
                          "actor_heads": ta.get("actor_heads", "single")}}, n_act, a.device)
        if ck is not None and ck.get("anchor"):
            anchor.load_state_dict(ck["anchor"])
            for sid, pay in list(anchor.snapshots.items()):  # a later observation growth: the --teacher layout rules
                from ballrain.path import RouteError
                try:
                    ign = AN.snapshot_branch_ignore(pay, snapshot_sources, env_cfg.branch_assets, env_cfg.level,
                                                    mine, a.teacher_allow_context_append)
                except RouteError as exc:
                    anchor.drop_snapshot(sid, f"branch compatibility: {exc}")
                    log.msg(f"anchor snapshot {sid} refused: branch compatibility: {exc}")
                    continue
                if ign:
                    log.msg(f"anchor snapshot {sid}: validated branch compatibility")
                err = AN.teacher_layout_error(mine, pay["features"], int(pay["n_obs"]), n_obs, n_act, n_act, ign)
                if err:
                    anchor.drop_snapshot(sid, f"layout: {err}")
        sm = anchor.summary()
        log.msg(f"mastered-state anchor: coef {a.anchor_coef}, families {','.join(kinds)} (bands of {band_frames} "
                f"frames), mastered at >= {a.anchor_threshold:g} over {a.anchor_min_lives} of the last "
                f"{a.anchor_window} lives, grace {a.anchor_grace} lives, teacher "
                + (a.anchor_teacher if a.anchor_teacher else f"per-family snapshots (<= {a.anchor_max_teachers})")
                + (f", only {','.join(sorted(anchor_only))}" if anchor_only else "")
                + (f"; resumed: {sm['families']} families, {sm['mastered']} mastered, {sm['teachers']} snapshots"
                   if ck is not None and ck.get("anchor") else ""))
    elif ck is not None and ck.get("anchor"):
        log.msg("the checkpoint holds mastered-anchor state; --anchor-mastered is off: it is not carried on")

    def anchor_terms(chunks, raw, obs_n, rms_before) -> tuple:
        """(teacher log-probs, row weights, stats) of the mastered-state anchor on this batch; (None, None, stats) when
        no anchored family has rows in it. New masteries get their teacher snapshot here (the current policy)."""
        u = counters["update"]
        def snapshot():
            payload = AN.actor_payload(model, rms_before, n_obs, mine, arch, u)
            if env_cfg.branch_assets is not None:
                payload["branch_assets"] = env_cfg.branch_assets
            return payload
        anchor.refresh(u, snapshot)
        if not anchor.explicit:
            for sid in [t for t in anchor_teachers if t not in anchor.snapshots]:
                del anchor_teachers[sid]
            for sid, pay in anchor.snapshots.items():
                if sid not in anchor_teachers:
                    anchor_teachers[sid] = AN.build_teacher(pay, n_act, a.device)
        for what, who, detail in anchor.drain_events():
            log.msg(f"anchor: {what} {who} ({detail})")
        anchored = anchor.anchored()
        if anchor_only:
            anchored = {k: v for k, v in anchored.items() if k in anchor_only}
        keys = AN.row_families(chunks, anchor_key, past_cp_counts=a.anchor_rows == "life" or a.success == "finish")
        w, groups = AN.row_weights(keys, anchored, anchor.weight, a.anchor_min_rows)
        st = {**anchor.summary(), "batch_families": len(groups), "rows": int(sum(len(g) for g in groups.values())),
              "tagged_rows": sum(k is not None for k in keys)}
        if not groups:
            return None, None, st
        lt = AN.teacher_logp(anchor_teachers, groups, anchored, raw, n_act, a.device)
        if (u + 1) % a.log_every == 0:  # per-family KL before the update (logging only: on the logged updates)
            fk = AN.family_kl(model, obs_n, lt, groups, a.device)
            st["kl_pre"] = float(np.mean(list(fk.values())))
            st["top"] = sorted(((v, k, len(groups[k])) for k, v in fk.items()), reverse=True)[:5]
        return lt, w, st

    def teacher_terms(raw):
        """Teacher log-probabilities on the batch (the teacher's own observation statistics), the route mask, the
        coefficient of this update."""
        with torch.no_grad():
            x = torch.as_tensor(normalize(raw[:, :teacher["n_obs"]], teacher["rms"]), device=a.device)
            logits, _ = teacher["model"](x)
            lt = torch.log_softmax(logits, -1).cpu().numpy()
        mask = raw[:, teacher["col"]] >= teacher["lo"]
        coef = a.teacher_coef
        if a.teacher_decay_updates and a.teacher_coef_floor is not None:
            f = min(1.0, (counters["update"] - teacher["u0"]) / a.teacher_decay_updates)
            coef = a.teacher_coef + (a.teacher_coef_floor - a.teacher_coef) * f
        return lt, mask, coef

    def watch_entropy(obs_n, raw):
        """Entropy of the current (behaviour) policy on the batch states inside the watched route window."""
        m = (raw[:, watch[0]] >= watch[1]) & (raw[:, watch[0]] <= watch[2])
        n = int(m.sum())
        if n == 0:
            return {"watch_entropy": float("nan"), "watch_n": 0.0}
        with torch.no_grad():
            logits, _ = model(torch.as_tensor(obs_n[m], device=a.device))
            lp = torch.log_softmax(logits, -1)
            ent = float(-(lp.exp() * lp).sum(-1).mean())
        return {"watch_entropy": ent, "watch_n": float(n)}

    def auto_rollback():
        """--auto-rollback: reload the newest good bin's checkpoint (policy, optimiser, observation statistics,
        curriculum; counters / best continue) and write ALERT_auto_rollback.txt."""
        u = counters["update"]
        if not guard.triggered(u):
            return
        cands = guard.targets(u)
        f = next((run_dir / f"ckpt_{v:06d}.pt" for v in cands if (run_dir / f"ckpt_{v:06d}.pt").is_file()), None)
        msg = (f"AUTO-ROLLBACK at u{u}: TAS starts {guard.lo}-{guard.hi} reach < {guard.low} in {guard.consecutive} bins "
               f"({guard.summary(u)})")
        if f is None:
            log.msg(msg + f"; no checkpoint of a bin with reach >= {guard.good} (candidates {cands[:5]}), not rolled back")
            guard.rolled_back(u)
        else:
            src = torch.load(f, weights_only=False)
            load_policy(src, load_opt=True)  # a checkpoint from before a growth is grown again (optimiser included)
            for g in opt.param_groups:
                g["lr"] = a.lr
            cu.load_state_dict(src["curriculum"])
            guard.rolled_back(u)
            msg += f" -> reloaded {f.name} (policy, optimiser, rms, curriculum; frontier {cu.frontier})"
            # a weight swap can change behaviour far from the guarded band (L1: the section-4 ramp was lost in a rollback)
            msg += "; check the section-4 ramp share of the reloaded policy (runtime/analysis/alt3/ramp.py on an eval)"
            log.msg(msg)
            broadcast()
        with open(run_dir / "ALERT_auto_rollback.txt", "a", encoding="utf-8") as fh:
            fh.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")

    if a.profile_sample > 0:
        from ballrain.train.stack_sampler import StackSampler
        StackSampler(run_dir / "profile.json", duration=a.profile_sample).start()
        log.msg(f"stack sampler: {a.profile_sample} s, every 10 ms, -> profile.json")
    try:
        first = ck is None and a.init_from is None
        drop_hist: dict = {}  # "local|remote:lag" -> dropped chunks since the last log line
        wait_hist: list = []  # seconds blocked waiting for data per batch, since the last log line
        while counters["steps"] < a.total_steps and not stop:
            chunks = []
            n = 0
            tc = time.perf_counter()
            t_wait = 0.0  # time blocked in data_q.get this batch (data arriving slower than the learner takes it)
            while n < a.batch_steps:
                if (run_dir / "STOP").is_file():
                    stop = True
                    break
                try:
                    tw0 = time.perf_counter()
                    m = in_q.get(timeout=30)
                    t_wait += time.perf_counter() - tw0
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
                if m.get("pool"):  # volunteer-pool data (ballrain.pool.learner_link): its own flag and lag bound; nothing
                    # of a refused / dropped pool chunk reaches the statistics (frames, reset_frac, the remote watchdog)
                    lag = counters["update"] - int(m["version"])
                    if not a.pool_ingest:
                        counters["pool_refused"] = counters.get("pool_refused", 0) + 1
                        if not pool_warned[0]:
                            log.msg("pool chunks arrive but --pool-ingest is off: dropped (counted in pool_refused)")
                            pool_warned[0] = True
                        continue
                    if m["raw"].shape[1] != n_obs or lag > a.pool_max_lag or lag < 0:
                        counters["pool_dropped"] = counters.get("pool_dropped", 0) + 1
                        counters["pool_dropped_steps"] = counters.get("pool_dropped_steps", 0) + len(m["rewards"])
                        continue
                    counters["steps_pool"] = counters.get("steps_pool", 0) + len(m["rewards"])
                    counters["frames"] += m["frames"]
                    pool_lags.append(lag)
                    chunks.append(m)
                    n += len(m["rewards"])
                    continue
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
                    # by source and lag (versions behind), for the log line: are drops a backlog or late arrivals?
                    dk = f"{'remote' if m.get('remote') else 'local'}:{min(counters['update'] - m['version'], 9)}"
                    drop_hist[dk] = drop_hist.get(dk, 0) + 1
                    counters["dropped_chunks"] += 1
                    counters[f"dropped_{be}"] = counters.get(f"dropped_{be}", 0) + 1
                    counters[f"dropped_steps_{be}"] = counters.get(f"dropped_steps_{be}", 0) + len(m["rewards"])
                    continue
                chunks.append(m)
                n += len(m["rewards"])
            if stop or not chunks:
                break
            collect_times.append(time.perf_counter() - tc)
            wait_hist.append(t_wait)
            raw = np.concatenate([c["raw"] for c in chunks])
            # Non-finite guard (ballrain.train.nonfinite): a batch with a non-finite value, or an update whose loss /
            # gradient turns non-finite, is skipped: model, optimiser and observation statistics stay as before it. It
            # still counts as an update, like the first batch (version, broadcast, log / checkpoint cadence unchanged).
            u_next = counters["update"] + 1
            rms_prev = rms.state_dict()
            skipped = nf_guard.check(u_next, chunks, {"raw": raw}, model, opt)  # before it reaches the statistics
            if not skipped:
                rms.update(raw)
            counters["steps"] += len(raw)
            tu = time.perf_counter()
            ustats, astats = {}, {}
            if skipped:
                pass
            elif first:  # the first batch only fits the observation statistics
                first = False
            else:
                obs_n = normalize(raw, rms.state_dict())
                pool_terms = pool_correction(chunks, obs_n) if a.pool_correction == "vtrace" else {}
                advs, rets, logps, vals = [], [], [], []
                for i, c in enumerate(chunks):
                    if i in pool_terms:
                        adv_i, ret_i, lp_i, v_i = pool_terms[i]
                    else:
                        adv_i, ret_i = gae(c, a.gamma, a.lam, a.reward_scale)
                        lp_i, v_i = c["logp"], c["values"]
                    advs.append(adv_i), rets.append(ret_i), logps.append(lp_i), vals.append(v_i)
                batch = {"obs": obs_n, "actions": np.concatenate([c["actions"] for c in chunks]),
                         "logp": np.concatenate(logps).astype(np.float32), "adv": np.concatenate(advs),
                         "ret": np.concatenate(rets)}
                skw, sstats = {}, {}
                if sil is not None and updates_this_process >= a.value_warmup:
                    adv_std = float(batch["adv"].std())
                    sstats["sil_dropped"] = sil.refresh(sil_value_fn(model, a.device),
                                                        normalize(sil.raw, rms.state_dict()) if len(sil) else None, adv_std)
                    sstats["sil_added"] = sil.offer(chunks, raw, batch["actions"], batch["ret"], batch["adv"], adv_std)
                    sstats["sil_confirmed"], sstats["sil_discarded"] = sil.confirmed, sil.discarded
                    sstats["sil_pending"] = sum(len(v["rows"]) for v in sil.pending.values())
                    if len(sil):
                        skw = {"sil_obs": normalize(sil.raw, rms.state_dict()), "sil_act": sil.act, "sil_ret": sil.ret,
                               "sil_coef": a.sil_coef, "sil_scale": adv_std, "sil_batch": a.sil_batch,
                               "sil_vcoef": a.sil_vcoef, "sil_wmax": a.sil_wmax}
                    sstats["sil_size"] = len(sil)
                wstats = watch_entropy(batch["obs"], raw) if watch is not None else {}
                model.train()
                tl, tm, tc = teacher_terms(raw) if teacher is not None else (None, None, 0.0)
                akw = {}
                if anchor is not None:
                    ta0 = time.perf_counter()
                    al, aw, astats = anchor_terms(chunks, raw, batch["obs"], rms_prev)
                    astats["terms_s"] = time.perf_counter() - ta0
                    if al is not None:
                        akw = {"anchor_logp": al, "anchor_weight": aw, "anchor_coef": a.anchor_coef}
                tu_ppo = time.perf_counter()
                ustats = nf_guard.update(u_next, chunks, {"values": np.concatenate(vals)}, model, opt, batch, a.epochs,
                                         a.minibatch, a.clip, a.vf_coef, a.ent_coef, a.max_grad_norm, a.device, gen,
                                         a.target_kl, teacher_logp=tl, teacher_mask=tm, teacher_coef=tc,
                                         value_only=updates_this_process < a.value_warmup,
                                         **({"demo_obs": normalize(demo["raw"], rms.state_dict()), "demo_act": demo["act"],
                                             "demo_coef": demo_coef_now()} if demo is not None else {}), **akw, **skw)
                if demo is not None and ustats is not None:
                    ustats["demo_coef"] = demo_coef_now()
                if ustats is not None:  # TensorBoard only (a new updates.csv column would start updates.<i>.csv)
                    sstats.update({k: ustats.pop(k) for k in [k for k in ustats if k.startswith("sil_")]})
                    for k, v in sstats.items():
                        if isinstance(v, (int, float)) and np.isfinite(v):
                            log.tb.add_scalar(f"sil/{k.removeprefix('sil_')}", v, counters["steps"])
                    last_sil = sstats
                updates_this_process += 1
                if anchor is not None:
                    astats["ppo_s"] = time.perf_counter() - tu_ppo
                    for k in [k for k in (ustats or {}) if k.startswith("anchor_")]:  # not in updates.csv (new columns
                        astats[k.removeprefix("anchor_")] = ustats.pop(k)  # would start updates.<i>.csv): anchor.csv
                model.eval()
                if ustats is None:  # skipped (logged, dumped): model and optimiser are the pre-update ones
                    skipped, ustats = True, {}
                    rms.load_state_dict(rms_prev)
                else:
                    ustats["explained_var"] = explained_variance(np.concatenate(vals), batch["ret"])
                    for k, v in pool_stats(chunks).items():  # TensorBoard only: updates.csv keeps its columns
                        log.tb.add_scalar(f"pool/{k}", v, counters["steps"])
                    ustats["adv_std"] = float(batch["adv"].std())
                    ustats["ret_mean"] = float(batch["ret"].mean())
                    ustats.update(wstats)
            if skipped:  # counted in counters["nonfinite_skipped"] by the guard
                pool_lags.clear()  # the skipped batch's pool statistics do not enter the next update's
                pool_rho.clear()
            update_times.append(time.perf_counter() - tu)
            counters["update"] += 1
            broadcast()
            u = counters["update"]
            for k, v in ustats.items():
                log.tb.add_scalar(f"loss/{k}", v, counters["steps"])
            if anchor is not None and astats:
                for k, v in astats.items():
                    if isinstance(v, (int, float)) and np.isfinite(v):
                        log.tb.add_scalar(f"anchor/{k}", v, counters["steps"])
                top = astats.get("top") or []
                log.row("anchor", {**{k: (round(v, 6) if isinstance(v, float) else v) for k, v in astats.items()
                                      if k != "top"}, "update": u, "steps": counters["steps"], "coef": a.anchor_coef,
                                   "top": "; ".join(f"{k}={v:.4f}/{n}" for v, k, n in top)}, cols=ANCHOR_COLS)
                if u % a.log_every == 0:
                    log.msg(f"anchor u{u}: mastered {astats['mastered']}/{astats['families']} (grace "
                            f"{astats['in_grace']}), teachers {astats['teachers']}, batch families "
                            f"{astats['batch_families']} rows {astats['rows']}/{astats['tagged_rows']}, kl_pre "
                            f"{astats.get('kl_pre', float('nan')):.4f} kl {astats.get('kl', float('nan')):.4f} coef "
                            f"{a.anchor_coef}" + (" top " + ", ".join(f"{k} {v:.4f} ({n})" for v, k, n in top[:3])
                                                  if top else ""))
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
                       "nonfinite_skipped": counters["nonfinite_skipped"],
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
                if a.pool_ingest or counters.get("pool_refused"):
                    for k in ("pool_refused", "pool_dropped", "pool_dropped_steps"):
                        log.tb.add_scalar(f"pool/{k}", counters.get(k, 0), counters["steps"])
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
                        f"upd {row['update_s']}s reset {row['reset_frac']} rsp_skip {n_skip}"
                        + (f" demo_acc {ustats['demo_acc']:.2f}" if "demo_acc" in ustats else "")
                        + (f" sil {last_sil.get('sil_size', 0)} +{last_sil.get('sil_added', 0)} w {last_sil.get('sil_w', float('nan')):.2f}"
                           if sil is not None and last_sil else "")
                        + (f" wait {np.mean(wait_hist):.2f}s" if wait_hist else "")
                        + (" drops " + ",".join(f"{k}={v}" for k, v in sorted(drop_hist.items())) if drop_hist else "")
                        + (f" nonfinite_skipped {row['nonfinite_skipped']}" if row["nonfinite_skipped"] else ""))
                if wait_hist:
                    log.tb.add_scalar("train/collect_wait_s", float(np.mean(wait_hist)), counters["steps"])
                drop_hist.clear()
                wait_hist.clear()
                log.flush()
                win_eps, win_runs = [], []
            if u % a.ckpt_every == 0:
                save_ckpt()
            if guard is not None and u % 5 == 0:
                auto_rollback()
    except NonFiniteError as e:
        log.msg(f"STOPPED by the non-finite guard: {e}")
        raise
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
                m = in_q.get(timeout=5)
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
    try:
        rc = main()
    except SystemExit as e:  # a refusal after the hub / workers started must not leave a learner hanging
        import os as _os
        print(e, file=sys.stderr, flush=True) if e.code not in (None, 0) and not isinstance(e.code, int) else None
        sys.stdout.flush()
        _os._exit(e.code if isinstance(e.code, int) else 1)
    raise SystemExit(rc)

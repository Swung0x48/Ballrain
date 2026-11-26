"""Collect states for a policy merge, or play-test a policy (docs/policy-merge.md).

Lives are handed over at frames of the checkpoint's start record drawn from weighted ranges and played by a frozen
behaviour policy (temperature 1): one checkpoint, or a composite of teachers chosen by the sector the ball is in
(--teacher KEY=CKPT, keys are sector numbers or "default"). Every raw observation is saved with its sector and life.
Per life it prints: the hand-over frame, start sector, last sector, steps, the sectors left with every score / life
ball taken, whether the level was finished, and whether the life reached its next checkpoint.

    # composite teacher, full level, states for distillation
    python tools/merge/collect.py --teacher 3=ball.pt --teacher default=flight.pt --ranges 936-25300:3,4000-5800 \\
        --n 30 --copy hl-81 --seed 101 --out runtime/merge/full/r0_1.npz
    # play test: one checkpoint from sector-3 hand-overs until the next checkpoint (prints how many took every ball)
    python tools/merge/collect.py --behaviour merged.pt --ranges 15500-16000 --until next-cp --n 40 --copy hl-95 \\
        --seed 11 --out runtime/merge/eval/ball.npz

A life also ends after --stuck-s seconds of game time without 5 units of progress along the reference line, or after
--cap steps. TRUSTED CHECKPOINTS ONLY (unpickled)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import load_actor, make_env, parse_ranges, parse_teachers, pick_teacher, s_ranges


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--behaviour", default="", help="checkpoint that plays every step (else the --teacher composite)")
    p.add_argument("--teacher", action="append", default=[], help="KEY=CKPT, KEY a sector number, 's:LO-HI' (route arc s range, wins over sectors) or 'default'")
    p.add_argument("--ranges", required=True, help="hand-over frame ranges LO-HI[:weight],...")
    p.add_argument("--n", type=int, required=True, help="lives")
    p.add_argument("--copy", required=True, help="spare headless copy label, e.g. hl-81 (one per process)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="output .npz (raw, sector, life)")
    p.add_argument("--until", choices=("end", "next-cp"), default="end",
                   help="end = play on until the level / life ends or --cap; next-cp = stop at the next checkpoint")
    p.add_argument("--cap", type=int, default=4000, help="steps per life at most")
    p.add_argument("--stuck-s", type=float, default=8.0, help="end a life after this much game time without progress")
    p.add_argument("--hl-bin", default="", help="headless binary (default BALLRAIN_HL_BIN)")
    p.add_argument("--hl-root", default="", help="game root (default BALLRAIN_HL_ROOT)")
    p.add_argument("--ball-order", default="retail")
    a = p.parse_args(argv)
    import torch
    from ballrain.train.ppo import normalize
    torch.set_num_threads(1)
    teachers = parse_teachers(a.teacher)
    if bool(a.behaviour) == bool(teachers):
        raise SystemExit("give either --behaviour CKPT or --teacher KEY=CKPT entries")
    pols = {k: load_actor(v) for k, v in ({"behaviour": a.behaviour} if a.behaviour else teachers).items()}
    ck0 = next(iter(pols.values()))[0]
    for k, (ck, _) in pols.items():
        if (ck["n_obs"], ck["n_act"]) != (ck0["n_obs"], ck0["n_act"]) or ck["env_config"].get("delta_ms") != ck0["env_config"].get("delta_ms"):
            raise SystemExit(f"{k}: observation / action layout or frame rate differs from the first policy")
    env = make_env(ck0, a.copy, a.ball_order, a.hl_bin or None, a.hl_root or None)
    stuck_steps = int(a.stuck_s * 1000 / (env.cfg.delta_ms * env.cfg.action_repeat))
    ranges = parse_ranges(a.ranges)
    w = np.array([r[2] for r in ranges])
    rng = np.random.default_rng(a.seed)
    obs_all, sec_all, life_all, s_all, lives = [], [], [], [], []
    sr = s_ranges(teachers)
    try:
        for i in range(a.n):
            gen = torch.Generator().manual_seed(a.seed * 1000 + i)
            lo, hi, _ = ranges[int(rng.choice(len(ranges), p=w / w.sum()))]
            env.cfg.start_frame = int(rng.integers(lo, hi))
            obs, info = env.reset(options={"new_process": True})
            sec = sec0 = int(info.get("sector") or 1)
            clean, fin, reached = [], False, False
            mark_s, mark_t = None, 0
            t = 0
            for t in range(a.cap):
                obs_all.append(np.asarray(obs, np.float32))
                sec_all.append(sec)
                s_now0 = env.reward.s
                s_all.append(float(s_now0) if s_now0 is not None else float("nan"))
                life_all.append(i)
                key = "behaviour" if a.behaviour else pick_teacher(sec, s_all[-1], teachers, sr)
                if key is None:
                    raise SystemExit(f"no teacher for sector {sec} (add --teacher default=CKPT)")
                ck, m = pols[key]
                act, _, _ = m.act(normalize(obs, ck["rms"]), gen=gen)
                obs, r, term, trunc, info = env.step(int(act))
                if any(e[1] == "LevelFinish" for e in info.get("events") or []):
                    fin = True
                ns = int(info.get("sector") or sec)
                if ns != sec:
                    pk = getattr(env.session.tracker, "pickups", None) if env.session is not None else None
                    if pk and len(pk.present(sec)) == 0:
                        clean.append(sec)
                    if ns > sec0:
                        reached = True
                    sec = ns
                    if a.until == "next-cp" and reached:
                        break
                if term or trunc:
                    break
                s_now = env.reward.s
                if s_now is not None:
                    if mark_s is None or s_now > mark_s + 5:
                        mark_s, mark_t = s_now, t
                    elif t - mark_t > stuck_steps:
                        break
            lives.append({"frame": int(env.cfg.start_frame), "sector0": sec0, "sector": sec, "steps": t + 1,
                          "clean": clean, "finished": fin, "next_cp": reached or fin})
    finally:
        env.close()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, raw=np.asarray(obs_all, np.float32), sector=np.asarray(sec_all, np.int8), s=np.asarray(s_all, np.float32),
                        life=np.asarray(life_all, np.int32))
    n = len(lives)
    print(json.dumps({"behaviour": Path(a.behaviour).name if a.behaviour else {k: Path(v).name for k, v in teachers.items()},
                      "lives": n, "states": len(obs_all), "next_cp": sum(x["next_cp"] for x in lives),
                      "finished": sum(x["finished"] for x in lives),
                      "left_clean": {s: sum(s in x["clean"] for x in lives) for s in sorted({c for x in lives for c in x["clean"]})},
                      "per_life": lives}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Distil teachers into one actor (docs/policy-merge.md): a policy merge, or a capacity test.

Data: .npz files from tools/merge/collect.py (raw, sector, life). Each state gets a teacher:
  --label-by sector  the teacher of the state's sector (--teacher 3=ball.pt --teacher default=flight.pt)
  --label-by prefix  the teacher of the data file's prefix (files PREFIX_*.npz; --teacher flight=a.pt --teacher ball=b.pt)
Loss: KL(teacher || student) over the actions, every teacher's states weighted equally; 20% of the lives (by file and
life) are held out. The student either continues a checkpoint (--init: its weights, observation normalisation and critic;
only the actor trains) or starts fresh (--fresh --hidden 256,256: normalisation from the data; critic untrained, acting
only). The result is a checkpoint in the --init / first teacher's format that tools/merge/collect.py, scripts/eval_m2.py
and train_m2 --resume read.

    # merge: sector 3 from the ball teacher, the rest from the flight teacher, starting from the flight teacher
    python tools/merge/distill.py --data runtime/merge/full --rounds r0 r1 --label-by sector \\
        --teacher 3=ball.pt --teacher default=flight.pt --init flight.pt --steps 20000 --out runtime/merge/m1.pt
    # capacity: can a fresh 256x256 actor hold both? (compare with one teacher at a time and with 384 / 512)
    python tools/merge/distill.py --data runtime/merge/cap --rounds flight ball --label-by prefix \\
        --teacher flight=a.pt --teacher ball=b.pt --fresh --hidden 256,256 --steps 20000 --out runtime/merge/j256.pt

TRUSTED CHECKPOINTS ONLY (unpickled)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from common import load_actor, parse_teachers, teacher_key


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="directory of collect.py .npz files")
    p.add_argument("--rounds", nargs="+", required=True, help="file prefixes to use (PREFIX_*.npz)")
    p.add_argument("--label-by", choices=("sector", "prefix"), default="sector")
    p.add_argument("--teacher", action="append", default=[], help="KEY=CKPT (sector number / file prefix / default)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--init", default="", help="student starts from this checkpoint (actor trains, rest kept)")
    g.add_argument("--fresh", action="store_true", help="fresh student of --hidden width")
    p.add_argument("--hidden", default="256,256")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=4096, help="states per teacher per step")
    p.add_argument("--lr", type=float, default=None, help="default 3e-4 with --init, 1e-3 with --fresh")
    p.add_argument("--held-frac", type=float, default=0.2)
    p.add_argument("--device", default="", help="torch device (default: the last CUDA device, else cpu)")
    p.add_argument("--out", required=True, help="output checkpoint (.pt); a .json summary is written beside it")
    a = p.parse_args(argv)
    import torch
    import torch.nn.functional as F
    from ballrain.train.ppo import ActorCritic, normalize
    dev = torch.device(a.device or (f"cuda:{torch.cuda.device_count() - 1}" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(0)
    teachers = parse_teachers(a.teacher)
    if not teachers:
        raise SystemExit("at least one --teacher KEY=CKPT")
    tmods = {k: load_actor(v, dev) for k, v in teachers.items()}
    base = load_actor(a.init)[0] if a.init else next(iter(tmods.values()))[0]
    for k, (ck, _) in tmods.items():
        if (ck["n_obs"], ck["n_act"]) != (base["n_obs"], base["n_act"]):
            raise SystemExit(f"teacher {k}: observation / action layout differs from the student's")
    raw, key, life = [], [], []
    off = 0
    D = Path(a.data)
    for r in a.rounds:
        files = sorted(D.glob(f"{r}_*.npz"))
        if not files:
            raise SystemExit(f"no {r}_*.npz in {D}")
        for f in files:
            z = np.load(f)
            raw.append(z["raw"])
            keys = [teacher_key(s, teachers) for s in z["sector"]] if a.label_by == "sector" else [teacher_key(r, teachers)] * len(z["raw"])
            key += keys
            life.append(z["life"] + off)
            off = int(life[-1].max()) + 1
    raw, life = np.concatenate(raw), np.concatenate(life)
    key = np.array([k if k is not None else "" for k in key])
    if (key == "").any():
        raise SystemExit(f"{int((key == '').sum())} states have no teacher (add --teacher default=CKPT)")
    print("states", len(raw), "lives", off, {k: int((key == k).sum()) for k in teachers}, flush=True)
    # teacher log-probabilities, each teacher with its own normalisation
    tl = torch.empty((len(raw), base["n_act"]), device=dev)
    with torch.no_grad():
        for k, (ck, m) in tmods.items():
            idx = np.flatnonzero(key == k)
            for i in range(0, len(idx), 65536):
                j = idx[i:i + 65536]
                tl[torch.as_tensor(j, device=dev)] = m(torch.as_tensor(normalize(raw[j], ck["rms"]), device=dev))[0]
    tlogp = F.log_softmax(tl, -1)
    if a.fresh:
        hidden = tuple(int(x) for x in a.hidden.split(","))
        rms = {"mean": raw.mean(0).astype(np.float64), "var": raw.var(0).astype(np.float64), "count": float(len(raw))}
        st = ActorCritic(base["n_obs"], base["n_act"], hidden, base["args"]["activation"], base["args"].get("actor_heads", "single")).to(dev)
    else:
        hidden = tuple(int(x) for x in base["args"]["hidden"].split(","))
        rms = base["rms"]
        st = ActorCritic(base["n_obs"], base["n_act"], hidden, base["args"]["activation"], base["args"].get("actor_heads", "single")).to(dev)
        st.load_state_dict(base["model"])
    x = torch.as_tensor(normalize(raw, rms), device=dev)
    rng = np.random.default_rng(1)
    held = np.isin(life, rng.choice(np.unique(life), max(1, int(off * a.held_frac)), replace=False))
    train = {k: np.flatnonzero((key == k) & ~held) for k in teachers}
    hold = {k: np.flatnonzero((key == k) & held) for k in teachers}
    params = list(st.pi.parameters()) + (list(st.extra_heads.parameters()) if hasattr(st, "extra_heads") else [])
    opt = torch.optim.Adam(params, lr=a.lr or (1e-3 if a.fresh else 3e-4))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps, eta_min=1e-5)

    def fit(idx):
        if len(idx) == 0:
            return None
        kl, ag = [], []
        with torch.no_grad():
            for i in range(0, len(idx), 65536):
                j = torch.as_tensor(idx[i:i + 65536], device=dev)
                lp = F.log_softmax(st(x[j])[0], -1)
                tp = tlogp[j]
                kl.append((tp.exp() * (tp - lp)).sum(-1))
                ag.append((tp.argmax(-1) == lp.argmax(-1)).float())
        return {"kl": round(float(torch.cat(kl).mean()), 4), "argmax_agree": round(float(torch.cat(ag).mean()), 4)}

    print("before (held out)", json.dumps({k: fit(v) for k, v in hold.items()}), flush=True)
    t0 = time.time()
    active = [k for k in teachers if len(train[k])]
    for it in range(a.steps):
        loss = 0.0
        for k in active:
            j = torch.as_tensor(np.random.choice(train[k], a.batch), device=dev)
            lp = F.log_softmax(st(x[j])[0], -1)
            tp = tlogp[j]
            loss = loss + (tp.exp() * (tp - lp)).sum(-1).mean() / len(active)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if (it + 1) % max(1, a.steps // 5) == 0:
            print(it + 1, f"{time.time() - t0:.0f}s held out", json.dumps({k: fit(v) for k, v in hold.items()}), flush=True)
    res = {"rounds": a.rounds, "label_by": a.label_by, "teachers": {k: Path(v).name for k, v in teachers.items()},
           "init": Path(a.init).name if a.init else None, "hidden": list(hidden), "steps": a.steps, "states": int(len(raw)),
           "train": {k: fit(v[:300000]) for k, v in train.items()}, "held_out": {k: fit(v) for k, v in hold.items()}}
    print(json.dumps(res), flush=True)
    out = dict(base)
    out["model"] = {k: v.detach().cpu() for k, v in st.state_dict().items()}
    out["rms"] = rms
    out["args"] = dict(base["args"], hidden=",".join(str(h) for h in hidden))
    out.pop("anchor", None)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, a.out)
    Path(a.out).with_suffix(".json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

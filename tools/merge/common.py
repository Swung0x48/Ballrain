"""Shared helpers of the policy-merge tools (docs/policy-merge.md): loading checkpoints as actors, teacher maps, the
headless environment of a checkpoint."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for p in (REPO, REPO / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def parse_teachers(specs: list[str]) -> dict:
    """["3=a.pt", "default=b.pt"] -> {"3": "a.pt", "default": "b.pt"}. Keys are sector numbers (label by sector) or
    data-file prefixes (label by file); "default" covers everything not listed."""
    out = {}
    for s in specs or []:
        k, sep, v = s.partition("=")
        if not sep or not k.strip() or not v.strip():
            raise SystemExit(f"--teacher {s!r}: expected KEY=CHECKPOINT")
        if k.strip() in out:
            raise SystemExit(f"--teacher {k.strip()!r} given twice")
        out[k.strip()] = v.strip()
    return out


def teacher_key(key, teachers: dict):
    """The teacher entry for a state key (its sector, or its file prefix): the key itself, else "default", else None."""
    k = str(key)
    if k in teachers:
        return k
    return "default" if "default" in teachers else None


def parse_ranges(spec: str) -> list[tuple[int, int, float]]:
    """"4000-5800:2,936-25300" -> [(4000, 5800, 2.0), (936, 25300, 1.0)] (hand-over frame ranges with weights)."""
    out = []
    for part in spec.split(","):
        r, _, w = part.strip().partition(":")
        lo, hi = (int(x) for x in r.split("-"))
        if hi <= lo:
            raise SystemExit(f"range {part!r}: HI must be above LO")
        out.append((lo, hi, float(w or 1.0)))
    return out


def load_actor(path, device="cpu"):
    """(checkpoint dict, ActorCritic in eval mode) of a training checkpoint. TRUSTED INPUT ONLY (unpickled)."""
    import torch
    from ballrain.train.ppo import ActorCritic
    ck = torch.load(path, weights_only=False, map_location="cpu")
    a = ck["args"]
    m = ActorCritic(ck["n_obs"], ck["n_act"], tuple(int(x) for x in a["hidden"].split(",")), a["activation"],
                    a.get("actor_heads", "single"))
    m.load_state_dict(ck["model"])
    return ck, m.eval().to(device)


def make_env(ck: dict, copy: str, ball_order: str = "retail", binary: str | None = None, root: str | None = None):
    """A headless BallanceEnv for the checkpoint's own observation / frame-rate settings (scripts/eval_m2.py), on the
    spare copy label `copy`. The engine comes from --hl-bin / --hl-root or BALLRAIN_HL_BIN / BALLRAIN_HL_ROOT."""
    import eval_m2
    from ballrain import instances
    from ballrain.env import BallanceEnv
    instances.LAUNCH_RECIPE = instances.launch_direct
    hl = {"ball_order": ball_order}
    if binary:
        hl["binary"] = binary
    if root:
        hl["root"] = root
    cfg = eval_m2.build_env_config(ck, str(REPO / "runtime" / f"merge-{copy}"), 4300, 200000, backend="headless", headless=hl)
    cfg.copy = copy
    cfg.trace = False
    return BallanceEnv(cfg)

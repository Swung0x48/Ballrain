"""Command line for scripts/train_m2.py that starts a NEW run from a checkpoint with exactly that checkpoint's
arguments (ck["args"]: rehearsal / rest lists, curriculum, rewards, PPO settings ...), plus a few explicit changes.

    python scripts/derive_run_args.py <ckpt> --run r9 [--init-from PATH] [--contacts] [--no-pool] [--check]

Prints the train_m2.py argument list (POSIX shell quoting) with --init-from <ckpt> --init-full; the route branch
bundle is not an argument: train_m2 takes it from the --init-from checkpoint (make_env_config(init=...)).
--init-from PATH: the checkpoint path as the learner machine sees it (default: <ckpt>).
--contacts: the f8 contact / rail blocks. --no-pool: no volunteer-pool chunks (--pool-ingest off).
--check: build the new run's EnvConfig here (needs the reference line asset and the TAS record name only) and compare
it with the checkpoint's env_config: everything equal except the feature label (f7 -> f8) and out_root.
"""
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

# never copied: the new run's identity, how it starts (set below), one-shot actions, secrets
NOT_COPIED = ("run", "resume", "init_from", "init_full", "branch_update", "branch_dir", "remote_key")


def argv_from_args(args: dict, overrides: dict) -> tuple[list, list]:
    """(train_m2 argv reproducing args with overrides, keys of args train_m2 no longer knows)."""
    import train_m2
    p = train_m2.build_parser()
    vals = {k: v for k, v in args.items() if k not in NOT_COPIED}
    vals.update(overrides)
    known = {a.dest for a in p._actions}
    out = []
    for act in p._actions:
        if act.dest == "help" or act.dest not in vals:
            continue
        v = vals[act.dest]
        opt = next(o for o in act.option_strings if o.startswith("--"))
        if isinstance(act, argparse._StoreTrueAction):
            if v:
                out.append(opt)
        elif v is None or (v == act.default and act.dest != "run"):
            continue
        else:
            out += [opt, str(v)]
    return out, sorted(set(vals) - known)


def r9_argv(ck: dict, run: str, init_from: str, contacts: bool, no_pool: bool) -> tuple[list, list]:
    ov = {"run": run, "init_from": init_from, "init_full": True}
    if contacts:
        ov["contacts"] = True
    if no_pool:
        ov["pool_ingest"] = False
    return argv_from_args(ck["args"], ov)


def compare_env(ck: dict, argv: list) -> list:
    """Differences between the new run's EnvConfig (built from argv, branch bundle from ck) and ck's env_config,
    as (key, old, new); out_root and the features' version label are expected to differ."""
    import train_m2
    a = train_m2.parse(argv)
    env = train_m2.make_env_config(a, Path("unused"), None, init=ck)
    new = train_m2.checkpoint_config(env)
    old = {"branch_assets": ck.get("branch_assets"), "env_config": ck["env_config"]}
    diffs = []
    if new["branch_assets"] != old["branch_assets"]:
        diffs.append(("branch_assets", "...", "..."))
    for k in sorted(set(new["env_config"]) | set(old["env_config"])):
        if k == "features":
            fo, fn = old["env_config"].get(k) or {}, new["env_config"].get(k) or {}
            for f in sorted(set(fo) | set(fn)):
                if _norm(fo.get(f)) != _norm(fn.get(f)):
                    diffs.append((f"features.{f}", fo.get(f), fn.get(f)))
        elif _norm(old["env_config"].get(k)) != _norm(new["env_config"].get(k)):
            diffs.append((k, old["env_config"].get(k), new["env_config"].get(k)))
    return diffs


def _norm(v):
    if isinstance(v, (list, tuple)):
        return [_norm(x) for x in v]
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items()}
    return v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt")
    ap.add_argument("--run", required=True)
    ap.add_argument("--init-from", default=None)
    ap.add_argument("--contacts", action="store_true")
    ap.add_argument("--no-pool", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    import torch
    ck = torch.load(a.ckpt, weights_only=False, map_location="cpu")
    out, unknown = r9_argv(ck, a.run, a.init_from or a.ckpt, a.contacts, a.no_pool)
    if unknown:
        print(f"# checkpoint arguments train_m2 no longer has (dropped): {unknown}", file=sys.stderr)
    print("python scripts/train_m2.py " + shlex.join(out))
    if a.check:
        diffs = compare_env(ck, out)
        for d in diffs:
            print(f"# differs: {d[0]}: {d[1]!r} -> {d[2]!r}", file=sys.stderr)
        unexpected = [d for d in diffs if d[0] not in ("out_root", "features.version")]
        print(f"# env_config check: {len(diffs)} differences, unexpected {len(unexpected)}", file=sys.stderr)
        return 1 if unexpected else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

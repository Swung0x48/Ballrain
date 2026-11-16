"""Vector environments over N game copies (runtime/train-00 .. train-(N-1)), one copy per sub-env.

make_vec_env(config, n) returns a gymnasium.vector env (async: one worker process per copy, so
feature computation runs in parallel; sync: everything in this process). make_sb3_vec_env returns a
stable-baselines3 SubprocVecEnv / DummyVecEnv. Copies must exist (instances.create_training_copies).
"""
from __future__ import annotations

import copy as _copy
from functools import partial

import gymnasium as gym

from . import instances
from .env import BallanceEnv, EnvConfig


def _make(config: EnvConfig, copy_name: str, launcher_factory=None, seed=None):
    cfg = _copy.deepcopy(config)
    cfg.copy = copy_name
    if seed is not None:
        cfg.seed = seed
    launcher = launcher_factory() if launcher_factory is not None else None
    return BallanceEnv(cfg, launcher=launcher)


def env_fns(config: EnvConfig, n: int, copies=None, launcher_factory=None, seed: int | None = None):
    names = list(copies) if copies is not None else instances.training_copy_names(n)
    if len(names) != n or len(set(names)) != n:
        raise ValueError(f"need {n} distinct copies, got {names}")
    return [partial(_make, config, name, launcher_factory, None if seed is None else seed + i)
            for i, name in enumerate(names)]


def make_vec_env(config: EnvConfig, n: int = 8, copies=None, backend: str = "async", launcher_factory=None,
                 seed: int | None = None):
    fns = env_fns(config, n, copies, launcher_factory, seed)
    if backend == "sync":
        return gym.vector.SyncVectorEnv(fns)
    if backend == "async":
        return gym.vector.AsyncVectorEnv(fns, context="spawn")
    raise ValueError(f"unknown backend {backend!r}")


def make_sb3_vec_env(config: EnvConfig, n: int = 8, copies=None, subproc: bool = True, launcher_factory=None,
                     seed: int | None = None):
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    fns = env_fns(config, n, copies, launcher_factory, seed)
    return SubprocVecEnv(fns, start_method="spawn") if subproc else DummyVecEnv(fns)

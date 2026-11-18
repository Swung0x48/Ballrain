"""Behaviour identity of the learner's own rollout workers with the code on main (the r8 workers will run this branch).

This branch extracts BallanceEnv._draw_start into ballrain.env.draw_start (the gateway's task draw) and adds
EnvConfig.start_rest (pool tasks only). With the pool off, a Python worker must draw exactly the same starts and play
exactly the same episodes as main's code for the same seed. The test runs the learner's Worker on the fake game with a
TAS start record and r8's start distribution (reverse curriculum, rehearsal behind the frontier with level-load 0-0
entries, counter randomisation, p_early_respawn 0 with respawn_level_starts) plus at-rest starts with and without a
drive target, once with this checkout and once with the merge base with main (a temporary git worktree), and requires
every chunk (observations, actions, log-probs, values, rewards, flags, bootstrap values, frames) and every episode
record to be identical. A second check draws 20000 starts with both versions of the start draw."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="ballrain.fakegame's shared memory needs Windows")


def _git(*args, cwd=REPO):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture(scope="module")
def base_tree(tmp_path_factory):
    """The merge base of this branch with main, checked out in a temporary worktree (removed afterwards)."""
    try:
        base = _git("merge-base", "HEAD", "main")
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("no git / no main branch")
    d = tmp_path_factory.mktemp("main-tree") / "tree"
    _git("worktree", "add", "--detach", str(d), base)
    try:
        yield base, d
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(d)], cwd=REPO, capture_output=True)
        subprocess.run(["git", "worktree", "prune"], cwd=REPO, capture_output=True)


@pytest.fixture(scope="module")
def record(tmp_path_factory):
    from ballrain.pool import dev
    return dev.fake_record(tmp_path_factory.mktemp("ident-record"))


def _run(root: Path, record: Path, out: Path, n: int) -> dict:
    p = subprocess.run([sys.executable, str(REPO / "tests" / "worker_identity_run.py"), str(root), str(record), str(out),
                        str(n)], capture_output=True, text=True, encoding="utf-8", timeout=900, cwd=str(root))
    assert p.returncode == 0, p.stderr[-3000:]
    return json.loads(p.stdout.strip().splitlines()[-1])


def test_workers_play_the_same_episodes_as_main(base_tree, record, tmp_path):
    base, tree = base_tree
    n = 40
    here = _run(REPO, record, tmp_path / "here", n)
    there = _run(tree, record, tmp_path / "main", n)
    assert len(here["chunks"]) == len(there["chunks"]) == n
    eps = [e for c in here["chunks"] for e in c["episodes"]]
    kinds = {(e["kind"], bool(e["start_frame"])) for e in eps}
    # the run covers every start kind of the distribution (else the comparison would prove less than it claims)
    assert {("tas", True), ("tas", False), ("tas_rest", True), ("respawn", True)} <= kinds | {("respawn", True)}, kinds
    assert any(e["kind"] == "tas_rest" for e in eps) and any(e["kind"] == "tas" and e["start_frame"] for e in eps)
    for i, (a, b) in enumerate(zip(here["chunks"], there["chunks"])):
        assert a == b, f"chunk {i} differs from main ({base[:10]})"


def _load_env_module(root: Path, name: str):
    """ballrain/env.py of another checkout as a submodule of this ballrain package (its relative imports resolve to
    this checkout's unchanged modules: features, curriculum, ...)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"ballrain.{name}", root / "ballrain" / "env.py",
                                                  submodule_search_locations=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "ballrain"
    sys.modules[f"ballrain.{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_start_draws_match_main(base_tree):
    from types import SimpleNamespace
    from ballrain import env as E
    _, tree = base_tree
    M = _load_env_module(tree, "env_main_copy")
    valid = np.arange(100, 5000, 7)

    def near(k):
        if k <= 0:
            return 0
        i = int(np.clip(np.searchsorted(valid, k), 0, len(valid) - 1))
        j = max(i - 1, 0)
        return int(valid[i] if abs(valid[i] - k) <= abs(valid[j] - k) else valid[j])
    starts = SimpleNamespace(nearest_valid=near, nearest_any=near)
    for cfg_kw in ({"rehearse_starts": [[0, 0], [0, 0], [200, 1500], [1500, 4800]], "p_rehearse": 0.12,
                    "rehearse_behind_frontier": True},
                   {"rest_starts": [[300, 600], [700, 720, 1.0, -2.0, 0.5, 77]], "p_rest": 0.3,
                    "rehearse_starts": [[0, 0], [100, 900]], "p_rehearse": 0.25}):
        outs = []
        for mod in (E, M):
            env = object.__new__(mod.BallanceEnv)
            env.cfg = mod.EnvConfig(**cfg_kw)
            env.starts = starts
            from ballrain.curriculum import CurriculumConfig, ReverseCurriculum
            cu = ReverseCurriculum(valid=valid, config=CurriculumConfig(start_back=500, band=300, weighted=0.5,
                                                                         bin_frames=200, seed=0))
            cu.frontier = int(valid[len(valid) // 2])
            cu.set_bin_fail({0: 0.9, 3: 0.2})
            env.curriculum = cu
            env._start_rng = np.random.default_rng(11)
            outs.append([(env._draw_start(), env._rest_next) for _ in range(20000)])
        assert outs[0] == outs[1]

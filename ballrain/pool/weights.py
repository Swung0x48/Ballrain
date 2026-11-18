"""Policy weights for pool clients: learner checkpoint -> plain-array npz + JSON manifest, and a numpy-only forward.

Server side (trusted): export_checkpoint(<run>/ckpt_*.pt, out_dir) reads the learner's own checkpoint with torch (the
only torch use in ballrain.pool; never on data from a client) and writes

    <out_dir>/<policy_sha>.npz    the actor-critic tensors (float32) and the observation statistics (float64), a plain
                                  npz (task.save_npz_deterministic: no pickle, fixed zip timestamps)
    <out_dir>/<policy_sha>.json   the manifest: architecture, observation normalisation, action set, feature / reward /
                                  env / rollout configuration (allowlisted fields only: no paths, hosts or names), the
                                  npz file's sha256 and size

policy_sha is canonical: sha256 over the manifest's identity fields (canonical JSON) and the arrays' raw bytes in a
fixed order (not over the npz file, whose zip container is not part of the identity). The file sha256 is separate,
for download integrity.

Client side: load_policy(npz, manifest) checks the file sha256, loads with np.load(allow_pickle=False), checks names /
dtypes / shapes / finiteness against the manifest and recomputes policy_sha; NumpyPolicy reproduces
ballrain.train.ppo.ActorCritic + normalize (float64 statistics, the same clip and float32 cast, float32 layers).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .protocol import dumps

FORMAT = "ballrain-pool-weights/1"
NORM_EPS = 1e-8  # ballrain.train.ppo.normalize
NORM_CLIP = 10.0
ARRAY_ORDER_PREFIX = ("pi.", "v.")

# EnvConfig fields a client takes from the manifest (everything else is either the task's -- start frame, ball order
# -- or the client's own -- copy, out_root, headless paths -- or must stay off: curriculum / rest / rehearsal draws)
ENV_FIELDS = ("level", "mode", "delta_ms", "action_repeat", "camera_actions", "world_actions", "repress_stale_keys",
              "camera_quarter_tol", "camera_turn_grace_ms", "esc_action", "collect_action", "esc_frames",
              "collect_max_frames", "collect_chunk", "mask_keys_while_paused", "pause_keys", "esc_requires_control",
              "max_pause_frames", "max_episode_pause_frames", "per_life", "max_episode_frames", "max_process_frames",
              "idle_keys", "skip_tutorial", "start_record", "exclude_starts", "rest_speed", "rest_max_frames",
              "start_check", "start_retries", "timeout_ms", "obs_timeout_s", "hello_timeout_s", "sr_tail",
              "respawn_limit_frames", "body_radius", "skip_render")
# Worker-side rollout semantics (ballrain.train.worker.WorkerSpec; scripts/train_m2.py's mapping from its arguments):
# the chunk's "rewards" are the worker's training reward under these
ROLLOUT_FIELDS = ("chunk", "gamma", "cap_mult", "cap_extra_s", "stall_s", "respawn_stall_s", "stall_progress",
                  "stall_terminal", "stall_extra", "stall_compensate", "stall_speed", "stall_max_s", "route_probes",
                  "gates", "split_k", "split_min_ref_s", "split_rest_allow_s", "split_extra", "counter_rand",
                  "p_early_respawn", "respawn_level_starts")
_ROLLOUT_JSON_ARGS = ("route_probes", "gates", "split_extra", "counter_rand")  # train_m2 passes these as JSON strings
# WorkerSpec shaping the pool's task rollout does not implement: a run that uses any of them is not exported (its
# chunks would carry another reward than the learner's workers compute)
UNSUPPORTED_ARGS = ("zones", "wall_push_rate", "rail_gate")
# the ROLLOUT_FIELDS scripts/train_m2.py actually passes from its arguments to WorkerSpec; the others (cap_extra_s,
# stall_progress, split_rest_allow_s) are WorkerSpec defaults there even if an argument of that name appears later
_ROLLOUT_FROM_ARGS = ("chunk", "gamma", "cap_mult", "stall_s", "respawn_stall_s", "stall_terminal", "stall_extra",
                      "stall_compensate", "stall_speed", "stall_max_s", "route_probes", "gates", "split_k",
                      "split_min_ref_s", "split_extra", "counter_rand", "p_early_respawn",
                      "respawn_level_starts")
ACTION_NAMES = {0: "none", 1: "up", 2: "down", 4: "left", 8: "right"}
# HeadlessConfig fields that change the simulation (SimTool's --tas recipe), taken from the run's env_config.headless;
# binary / root / socket / host / port / log / extra_env are the client's own, ball_order is the task's
HEADLESS_FIELDS = ("boot_ticks", "hidden_ticks", "legacy_edit", "reset_nmc", "reset_seed", "reset_frame",
                   "reset_precision", "orig_random")


def _c(a) -> np.ndarray:
    """C-contiguous copy-if-needed that keeps 0-d arrays 0-d (np.ascontiguousarray makes them 1-d)."""
    return np.asarray(a, order="C")


# ============================================================================================== public-safety

def _path_like(s: str) -> bool:
    t = s.replace("\\", "/")
    return (len(t) >= 2 and t[1] == ":") or t.startswith(("/", "~", "//")) or "\\" in s or "/users/" in t.lower() \
        or "/home/" in t.lower()


def assert_public(obj, where: str = "$") -> None:
    """Refuse a manifest value that looks like a local path (absolute, a drive letter, a home directory, backslashes).
    Raises ValueError naming the field."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and _path_like(k):
                raise ValueError(f"{where}: key {k!r} looks like a path")
            assert_public(v, f"{where}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            assert_public(v, f"{where}[{i}]")
    elif isinstance(obj, str):
        if _path_like(obj):
            raise ValueError(f"{where}: {obj!r} looks like a local path")
    elif obj is not None and not isinstance(obj, (bool, int, float)):
        raise ValueError(f"{where}: {type(obj).__name__} is not a JSON value")


def _plain(v):
    """JSON-safe copy (tuples -> lists, numpy scalars -> Python)."""
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    return v


# ============================================================================================== identity

IDENTITY_FIELDS = ("format", "source", "arch", "obs_norm", "actions", "env", "engine_options", "features", "reward",
                   "rollout", "arrays")


def policy_sha(manifest: dict, arrays: dict) -> str:
    """sha256 over the manifest's identity fields (canonical JSON) and every array (sorted by name: name, dtype,
    shape, raw C-order bytes)."""
    h = hashlib.sha256(b"ballrain-pool/policy/v1\0")
    h.update(dumps({k: manifest[k] for k in IDENTITY_FIELDS}))
    for name in sorted(arrays):
        a = _c(arrays[name])
        h.update(b"\0" + name.encode() + b"\0" + a.dtype.str.encode() + b"\0" + dumps(list(a.shape)) + b"\0")
        h.update(a.tobytes())
    return h.hexdigest()


def _file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ============================================================================================== export (server)

def export_state(state: dict, out_dir, run: str) -> dict:
    """A learner checkpoint dict (scripts/train_m2.py save_ckpt) -> <out_dir>/<sha>.npz + <sha>.json; returns the
    manifest. Only allowlisted configuration fields are written, each checked by assert_public."""
    from dataclasses import fields

    from ballrain.env import ARROWS9, CAMERA_ACTIONS, EnvConfig
    from ballrain.features import FeatureConfig
    from ballrain.rewards import RewardConfig
    from ballrain.train.worker import WorkerSpec

    out_dir = Path(out_dir)
    sd = state["model"]
    args = dict(state.get("args") or {})
    n_obs, n_act = int(state["n_obs"]), int(state["n_act"])
    n_layers = sum(1 for k in sd if k.startswith("pi.") and k.endswith(".weight"))
    hidden = [int(sd[f"pi.{2 * i}.weight"].shape[0]) for i in range(n_layers - 1)]
    act = str(args.get("activation", "tanh"))
    if act not in ("tanh", "relu"):
        raise ValueError(f"activation {act!r}")
    arrays = {}
    for k, v in sd.items():
        if not k.startswith(ARRAY_ORDER_PREFIX):
            raise ValueError(f"unexpected model tensor {k!r}")
        v = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)  # torch tensors or numpy (learner link)
        arrays[k] = _c(v.astype("<f4"))
    rms = state["rms"]
    arrays["rms_mean"] = _c(np.asarray(rms["mean"], "<f8"))
    arrays["rms_var"] = _c(np.asarray(rms["var"], "<f8"))
    for k, a in arrays.items():
        if not np.isfinite(a).all():
            raise ValueError(f"{k}: non-finite values")
    _check_arch(arrays, n_obs, n_act, hidden)

    ec = dict(state.get("env_config") or {})
    from ballrain.headless import HeadlessConfig
    hl = dict(ec.get("headless") or {})
    hdef = HeadlessConfig(binary="", root="", socket="")
    engine_options = _plain({k: hl.get(k, getattr(hdef, k)) for k in HEADLESS_FIELDS})
    known_env = {f.name for f in fields(EnvConfig)}
    env = {k: _plain(ec[k]) if k in ec else _plain(getattr(EnvConfig(), k)) for k in ENV_FIELDS if k in known_env}
    fc = dict(ec.get("features") or {})
    feats = _plain({f.name: fc.get(f.name, getattr(FeatureConfig(), f.name)) for f in fields(FeatureConfig)})
    if feats.pop('branch_dir', None) or feats.pop('branch_sha256', None) or state.get('branch_assets') is not None:
        raise ValueError('pool clients do not yet implement ball-type branch assets')
    feats["path_dir"] = None  # the client installs the asset itself (task assets, sha256 below)
    rc = ec.get("reward") or {}
    reward = _plain({f.name: rc.get(f.name, getattr(RewardConfig(), f.name)) for f in fields(RewardConfig)})
    for k in UNSUPPORTED_ARGS:
        v = args.get(k)
        if isinstance(v, str):
            v = json.loads(v) if v.strip() else {}
        if v:
            raise ValueError(f"the run uses {k}, which pool tasks do not implement (not exported)")
    ws = {f.name: f.default for f in fields(WorkerSpec) if f.name in ROLLOUT_FIELDS}
    ws.update({"route_probes": {}, "gates": {}, "split_extra": [], "counter_rand": {}})
    for k in _ROLLOUT_FROM_ARGS:
        if k not in args:
            continue
        v = args[k]
        if k in _ROLLOUT_JSON_ARGS and isinstance(v, str):
            v = json.loads(v) if v.strip() else ([] if k == "split_extra" else {})
        ws[k] = v
    rollout = _plain(ws)
    keys = list(ARROWS9) + (list(CAMERA_ACTIONS) if env.get("camera_actions") else [])
    esc = env.get("esc_action") if env.get("esc_action") is not None else env.get("mode") == "hs"
    if esc or env.get("collect_action"):
        raise ValueError("Esc / collect actions are not supported by the pool prototype (SR runs only)")
    if len(keys) != n_act:
        raise ValueError(f"action set has {len(keys)} entries, the model {n_act}")
    names = ["+".join(ACTION_NAMES[b] for b in (1, 2, 4, 8) if k & b) or "none" for k in keys]
    if env.get("camera_actions"):
        names[-2:] = ["cam_left", "cam_right"]
    c = state.get("counters") or {}
    m = {
        "format": FORMAT,
        "source": {"run": str(run), "update": int(c.get("update", 0)), "steps": int(c.get("steps", 0))},
        "arch": {"kind": "mlp_actor_critic", "n_obs": n_obs, "n_act": n_act, "hidden": hidden, "activation": act},
        "obs_norm": {"eps": NORM_EPS, "clip": NORM_CLIP, "count": float(rms.get("count", 0.0))},
        "actions": {"keys": [int(k) for k in keys], "names": names, "world_actions": bool(env.get("world_actions"))},
        "env": env, "engine_options": engine_options, "features": feats, "reward": reward, "rollout": rollout,
        "arrays": {k: {"dtype": a.dtype.str, "shape": list(a.shape)} for k, a in sorted(arrays.items())},
    }
    assert_public(m)
    dumps(m)  # JSON-safe (raises on NaN / non-JSON values)
    sha = policy_sha(m, arrays)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / f"{sha}.npz"
    from .task import save_npz_deterministic
    save_npz_deterministic(npz, arrays)
    m = dict(m, policy_sha=sha, npz_sha256=_file_sha(npz), npz_bytes=npz.stat().st_size)
    mp = out_dir / f"{sha}.json"
    tmpj = out_dir / f".{sha}.{os.getpid()}.tmp.json"
    tmpj.write_text(json.dumps(m, indent=1, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(tmpj, mp)
    return m


def load_trusted_checkpoint(ckpt_path) -> dict:
    """The learner's checkpoint dict (model, rms, curriculum, args, env_config, ...). TRUSTED INPUT ONLY: torch.load
    with weights_only=False unpickles, so this is for the learner's own checkpoints (e.g.
    runtime/m2-train-mac/<run>/ckpt_*.pt), never for received files. The gateway uses it for the policy export and the
    curriculum state of the same checkpoint."""
    import torch
    return torch.load(Path(ckpt_path), weights_only=False, map_location="cpu")


def export_checkpoint(ckpt_path, out_dir, run: str | None = None) -> dict:
    """export_state for a checkpoint file (TRUSTED INPUT ONLY, see load_trusted_checkpoint)."""
    p = Path(ckpt_path)
    return export_state(load_trusted_checkpoint(p), out_dir, run or p.parent.name)


def _check_arch(arrays: dict, n_obs: int, n_act: int, hidden: list) -> None:
    dims = [n_obs] + list(hidden)
    for head, n_out in (("pi", n_act), ("v", 1)):
        for i, (a, b) in enumerate(zip(dims, dims[1:] + [n_out])):
            w, bias = arrays.get(f"{head}.{2 * i}.weight"), arrays.get(f"{head}.{2 * i}.bias")
            if w is None or bias is None or w.shape != (b, a) or bias.shape != (b,):
                raise ValueError(f"{head} layer {i}: expected weight ({b}, {a}) and bias ({b},)")
    want = {f"{h}.{2 * i}.{p}" for h in ("pi", "v") for i in range(len(hidden) + 1) for p in ("weight", "bias")}
    want |= {"rms_mean", "rms_var"}
    if set(arrays) != want:
        raise ValueError(f"array names {sorted(set(arrays) ^ want)} unexpected / missing")
    if arrays["rms_mean"].shape != (n_obs,) or arrays["rms_var"].shape != (n_obs,):
        raise ValueError("observation statistics shape")
    if (arrays["rms_var"] < 0).any():
        raise ValueError("negative observation variance")


# ============================================================================================== load (client)

MANIFEST_MAX_BYTES = 256 * 1024
NPZ_MAX_BYTES = 64 * 1024 * 1024


def read_manifest(path_or_bytes) -> dict:
    """A manifest file (or its bytes) -> dict (strict JSON; structural checks happen in load_policy)."""
    from .protocol import loads_strict
    data = Path(path_or_bytes).read_bytes() if isinstance(path_or_bytes, (str, os.PathLike)) else path_or_bytes
    m = loads_strict(data, MANIFEST_MAX_BYTES)
    if not isinstance(m, dict) or m.get("format") != FORMAT:
        raise ValueError(f"not a {FORMAT} manifest")
    return m


def load_policy(npz_path, manifest, expect_sha: str | None = None) -> "NumpyPolicy":
    """Check and load downloaded weights: file size and sha256 against the manifest, np.load(allow_pickle=False),
    array names / dtypes / shapes against the manifest's architecture, finiteness, policy_sha recomputed (and equal
    to expect_sha, the version the task names, when given). Raises ValueError."""
    m = manifest if isinstance(manifest, dict) else read_manifest(manifest)
    p = Path(npz_path)
    if p.stat().st_size > NPZ_MAX_BYTES or p.stat().st_size != int(m["npz_bytes"]):
        raise ValueError("weights file size differs from the manifest")
    if _file_sha(p) != m["npz_sha256"]:
        raise ValueError("weights file sha256 differs from the manifest")
    arch = m["arch"]
    if arch.get("kind") != "mlp_actor_critic" or arch.get("activation") not in ("tanh", "relu"):
        raise ValueError(f"unsupported architecture {arch!r}")
    arrays = {}
    with np.load(p, allow_pickle=False) as z:
        if set(z.files) != set(m["arrays"]):
            raise ValueError("weights file arrays differ from the manifest")
        for k in z.files:
            a = z[k]
            spec = m["arrays"][k]
            if a.dtype.str != spec["dtype"] or list(a.shape) != list(spec["shape"]) or a.dtype.hasobject:
                raise ValueError(f"{k}: dtype / shape differ from the manifest")
            if a.dtype.str not in ("<f4", "<f8"):
                raise ValueError(f"{k}: dtype {a.dtype.str} not allowed")
            if not np.isfinite(a).all():
                raise ValueError(f"{k}: non-finite values")
            arrays[k] = _c(a)
    _check_arch(arrays, int(arch["n_obs"]), int(arch["n_act"]), [int(h) for h in arch["hidden"]])
    sha = policy_sha(m, arrays)
    if sha != m.get("policy_sha") or (expect_sha is not None and sha != expect_sha):
        raise ValueError("policy_sha does not match the weights")
    return NumpyPolicy(arrays, m)


class NumpyPolicy:
    """ballrain.train.ppo.ActorCritic.act / value and ppo.normalize in numpy. raw observations in, as the worker."""

    def __init__(self, arrays: dict, manifest: dict):
        self.manifest = manifest
        self.sha = manifest.get("policy_sha")
        arch = manifest["arch"]
        self.n_obs, self.n_act = int(arch["n_obs"]), int(arch["n_act"])
        nl = len(arch["hidden"]) + 1
        self.pi = [(arrays[f"pi.{2 * i}.weight"].T.copy(), arrays[f"pi.{2 * i}.bias"]) for i in range(nl)]
        self.v = [(arrays[f"v.{2 * i}.weight"].T.copy(), arrays[f"v.{2 * i}.bias"]) for i in range(nl)]
        self.act_fn = np.tanh if arch["activation"] == "tanh" else (lambda x: np.maximum(x, np.float32(0)))
        self.mean = arrays["rms_mean"].astype(np.float64)
        self.std = np.sqrt(arrays["rms_var"].astype(np.float64) + manifest["obs_norm"]["eps"])
        self.clip = float(manifest["obs_norm"]["clip"])

    def normalize(self, raw) -> np.ndarray:
        """ppo.normalize: float64 statistics, clip, float32."""
        return np.clip((np.asarray(raw) - self.mean) / self.std, -self.clip, self.clip).astype(np.float32)

    def _mlp(self, layers, x):
        h = np.asarray(x, np.float32)
        for i, (w, b) in enumerate(layers):
            h = h @ w + b
            if i < len(layers) - 1:
                h = self.act_fn(h)
        return h.astype(np.float32, copy=False)

    def logits(self, x) -> np.ndarray:
        """Policy logits of normalised observations x ((n_obs,) or (B, n_obs))."""
        return self._mlp(self.pi, x)

    def value(self, x) -> np.ndarray:
        """Value of normalised observations: scalar array for (n_obs,), (B,) for a batch."""
        return self._mlp(self.v, x)[..., 0]

    @staticmethod
    def log_softmax(logits) -> np.ndarray:
        z = np.asarray(logits, np.float64)
        z = z - z.max(-1, keepdims=True)
        return (z - np.log(np.exp(z).sum(-1, keepdims=True))).astype(np.float32)

    def log_probs(self, x) -> np.ndarray:
        return self.log_softmax(self.logits(x))

    def act(self, x, rng: np.random.Generator | None = None, greedy: bool = False, u: float | None = None):
        """One normalised observation -> (action, behaviour log-prob, value), as ActorCritic.act (temperature 1).
        The random stream differs from torch's (the log-prob of the action taken is what the learner needs). u: the
        task's deterministic draw (ballrain.pool.sampling.sample_u) instead of rng."""
        lp = self.log_probs(x)
        if greedy:
            a = int(np.argmax(lp))
        elif u is not None:
            from .sampling import sample_action
            a = sample_action(lp, u)
        else:
            p = np.exp(lp.astype(np.float64))
            c = np.cumsum(p / p.sum())
            a = int(min(np.searchsorted(c, rng.random(), side="right"), self.n_act - 1))
        return a, float(lp[a]), float(self.value(x))

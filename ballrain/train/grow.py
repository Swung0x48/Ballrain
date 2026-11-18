"""Growing a checkpoint's observation (scripts/train_m2.py): a policy trained on n_old entries continues on a layout
with blocks appended at the end (f6 look-ahead, f7 reach).

The new input columns of the first layers (pi.0.weight, v.0.weight) are zero, so the actions' logits and the value
are unchanged for any values in the new entries; the Adam first moments of those columns are zero and their second
moments the mean of the same row's old columns (the other entries keep theirs, so the optimiser continues where it
was): with zero second moments and the checkpoint's large step count (no bias correction left) the first steps of the
new columns would be 3-6x the learning rate (exp_avg / sqrt(exp_avg_sq) = 0.1 g / (0.03 |g|)); seeded they move like
the old columns. The observation statistics of the new entries are mean 0, var 1 with the checkpoint's count. A
shrink, or an old layout that is not a prefix of the new one, is refused.
"""
from __future__ import annotations

import numpy as np
import torch

FIRST_LAYERS = ("pi.0.weight", "v.0.weight")


class GrowError(ValueError):
    pass


def check_prefix(old_features: dict | None, new_features: dict | None, n_old: int, n_new: int) -> None:
    """GrowError unless the checkpoint's layout (its env_config features) is the start of the new one. Without a
    feature config in the checkpoint (old checkpoints) only the sizes are checked."""
    if n_old > n_new:
        raise GrowError(f"checkpoint has {n_old} observation entries, this config {n_new}: a shrink is not supported")
    if old_features is None or new_features is None:
        return
    from dataclasses import fields

    from ballrain.features import FeatureConfig, make_layout
    known = {f.name for f in fields(FeatureConfig)}
    lo = make_layout(FeatureConfig(**{k: v for k, v in old_features.items() if k in known}))
    ln = make_layout(FeatureConfig(**{k: v for k, v in new_features.items() if k in known}))
    if sum(k for _, k in lo) != n_old:
        raise GrowError(f"checkpoint n_obs {n_old} does not match its own feature config ({sum(k for _, k in lo)})")
    if [tuple(x) for x in ln[:len(lo)]] != [tuple(x) for x in lo]:
        raise GrowError(f"the checkpoint's observation layout is not a prefix of this config's: "
                        f"{[n for n, _ in lo]} vs {[n for n, _ in ln]}")


def grow_model_state(sd: dict, n_old: int, n_new: int) -> dict:
    sd = dict(sd)
    for k in FIRST_LAYERS:
        w = sd[k]
        if w.shape[1] != n_old:
            raise GrowError(f"{k} has {w.shape[1]} inputs, the checkpoint says n_obs {n_old}")
        sd[k] = torch.cat([w, torch.zeros(w.shape[0], n_new - n_old, dtype=w.dtype)], 1)
    return sd


def grow_rms_state(st: dict, n_old: int, n_new: int) -> dict:
    return {"mean": np.r_[np.asarray(st["mean"], np.float64), np.zeros(n_new - n_old)],
            "var": np.r_[np.asarray(st["var"], np.float64), np.ones(n_new - n_old)], "count": st["count"]}


def _seed_columns(v: torch.Tensor, n_new: int) -> torch.Tensor:
    """Second-moment columns for the new inputs: per row the mean of its old columns (a row whose old columns are all
    zero takes the mean of the whole tensor; an all-zero tensor stays zero)."""
    n_old = v.shape[1]
    row = v.mean(1, keepdim=True)
    row = torch.where(row > 0, row, v.mean())
    return torch.cat([v, row.expand(v.shape[0], n_new - n_old).to(v.dtype)], 1)


def grow_optimizer_state(opt_sd: dict, model, n_old: int, n_new: int) -> dict:
    """Adam (or any per-parameter state) of a checkpoint for the grown model: tensors of the first layers' weights
    with n_old columns get columns appended: zero for exp_avg (and any other state), the row mean of the old columns
    for exp_avg_sq / max_exp_avg_sq (module docstring; the step count is per tensor and stays). The optimiser's
    parameter order is model.parameters() (torch indexes the state by position)."""
    names = [n for n, _ in model.named_parameters()]
    idx = {names.index(k) for k in FIRST_LAYERS}
    groups = opt_sd["param_groups"]
    n_params = sum(len(g["params"]) for g in groups)
    if n_params != len(names):
        raise GrowError(f"optimiser state has {n_params} parameters, the model {len(names)}")
    order = [p for g in groups for p in g["params"]]
    pos = {pid: i for i, pid in enumerate(order)}
    state = {}
    for pid, s in opt_sd["state"].items():
        s = dict(s)
        if pos[pid] in idx:
            for k, v in list(s.items()):
                if torch.is_tensor(v) and v.dim() == 2 and v.shape[1] == n_old:
                    if k in ("exp_avg_sq", "max_exp_avg_sq"):
                        s[k] = _seed_columns(v, n_new)
                    else:
                        s[k] = torch.cat([v, torch.zeros(v.shape[0], n_new - n_old, dtype=v.dtype)], 1)
        state[pid] = s
    return {"state": state, "param_groups": groups}


def load_grown(model, opt, rms, src: dict, n_obs: int, new_features: dict | None = None, load_opt: bool = True,
               log=None) -> bool:
    """Model, observation statistics and (load_opt) optimiser state of a checkpoint dict into model / opt / rms,
    grown to n_obs entries when the checkpoint has fewer. Returns True when it was grown."""
    sd, n_old = src["model"], int(src["n_obs"])
    st = src["rms"]
    old_features = (src.get("env_config") or {}).get("features")
    if n_old == n_obs:
        model.load_state_dict(sd)
        rms.load_state_dict(st)
        if load_opt:
            opt.load_state_dict(src["opt"])
        return False
    check_prefix(old_features, new_features, n_old, n_obs)
    model.load_state_dict(grow_model_state(sd, n_old, n_obs))
    rms.load_state_dict(grow_rms_state(st, n_old, n_obs))
    if load_opt:
        opt.load_state_dict(grow_optimizer_state(src["opt"], model, n_old, n_obs))
    if log is not None:
        log(f"policy grown from {n_old} to {n_obs} observation entries (zero input weights; Adam for the new columns: "
            f"zero first moments, second moments seeded with the row mean of the old columns; observation statistics "
            f"mean 0 / var 1)")
    return True

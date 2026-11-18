"""PPO pieces: actor-critic MLP, running observation normalisation, GAE over worker chunks, the clipped update.

A chunk is T consecutive steps of one worker (ballrain.train.worker): obs (normalised as the worker saw them),
raw obs, actions, behaviour log-probs and values, rewards, terminated / truncated flags, the value of the final
observation of a truncated step (bootstrap) and the value of the observation after the last step (bootstrap at
the chunk end). Terminated (BallOff, LevelFinish, GameOver) bootstraps 0; truncated (frame cap, stall, crash)
bootstraps the value of its final observation. Off-policy chunks (the volunteer pool) get V-trace targets instead of
GAE (vtrace) and are clipped around the policy at the start of the update (decoupled PPO: their "logp" in the batch is
the proximal policy's, the behaviour policy enters through the truncated importance weights of vtrace).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class RunningMeanStd:
    """Parallel (Chan et al.) running mean / variance, float64."""

    def __init__(self, shape, eps: float = 1e-4):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = eps

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, np.float64)
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        d = bm - self.mean
        tot = self.count + bc
        self.mean = self.mean + d * bc / tot
        m2 = self.var * self.count + bv * bc + d * d * self.count * bc / tot
        self.var = m2 / tot
        self.count = tot

    def state_dict(self) -> dict:
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": float(self.count)}

    def load_state_dict(self, d: dict) -> None:
        self.mean, self.var, self.count = np.asarray(d["mean"], np.float64), np.asarray(d["var"], np.float64), float(d["count"])


def normalize(x: np.ndarray, rms: dict, clip: float = 10.0) -> np.ndarray:
    return np.clip((x - rms["mean"]) / np.sqrt(rms["var"] + 1e-8), -clip, clip).astype(np.float32)


def _mlp(n_in, hidden, n_out, act, out_gain):
    layers, d = [], n_in
    for h in hidden:
        lin = nn.Linear(d, h)
        nn.init.orthogonal_(lin.weight, np.sqrt(2))
        nn.init.zeros_(lin.bias)
        layers += [lin, act()]
        d = h
    out = nn.Linear(d, n_out)
    nn.init.orthogonal_(out.weight, out_gain)
    nn.init.zeros_(out.bias)
    return nn.Sequential(*layers, out)


class ActorCritic(nn.Module):
    def __init__(self, n_obs: int, n_act: int, hidden=(256, 256), act: str = "tanh"):
        super().__init__()
        a = {"tanh": nn.Tanh, "relu": nn.ReLU}[act]
        self.pi = _mlp(n_obs, hidden, n_act, a, 0.01)
        self.v = _mlp(n_obs, hidden, 1, a, 1.0)

    def forward(self, x):
        return self.pi(x), self.v(x).squeeze(-1)

    @torch.no_grad()
    def act(self, x: np.ndarray, temperature: float = 1.0, greedy: bool = False, gen=None):
        """x: (n_obs,) or (B, n_obs) normalised. Returns (actions, logp, values) as numpy (behaviour log-probs
        of the temperature-1 policy are only meaningful for temperature 1)."""
        t = torch.as_tensor(x, dtype=torch.float32)
        single = t.dim() == 1
        if single:
            t = t[None]
        logits, v = self(t)
        if greedy:
            a = logits.argmax(-1)
        else:
            probs = torch.softmax(logits / max(temperature, 1e-6), -1)
            a = torch.multinomial(probs, 1, generator=gen).squeeze(-1)
        logp = torch.log_softmax(logits, -1).gather(-1, a[:, None]).squeeze(-1)
        out = a.numpy(), logp.numpy(), v.numpy()
        return tuple(o[0] for o in out) if single else out

    @torch.no_grad()
    def value(self, x: np.ndarray) -> float:
        return float(self.v(torch.as_tensor(x, dtype=torch.float32)[None]).squeeze())


def gae(chunk: dict, gamma: float, lam: float, reward_scale: float = 1.0):
    """Advantages and returns of one chunk (arrays of length T)."""
    r = chunk["rewards"].astype(np.float64) * reward_scale
    v = chunk["values"].astype(np.float64)
    term, trunc = chunk["terminated"], chunk["truncated"]
    tv = chunk["trunc_values"].astype(np.float64)
    T = len(r)
    adv = np.zeros(T, np.float64)
    last = 0.0
    for t in range(T - 1, -1, -1):
        if term[t]:
            nv, cont = 0.0, 0.0
        elif trunc[t]:
            nv, cont = tv[t], 0.0
        else:
            nv = v[t + 1] if t + 1 < T else float(chunk["last_value"])
            cont = 1.0
        delta = r[t] + gamma * nv - v[t]
        last = delta + gamma * lam * cont * last
        adv[t] = last
    return adv.astype(np.float32), (adv + v).astype(np.float32)


def vtrace(chunk: dict, logp_cur, v_cur, v_last: float, gamma: float, lam: float, reward_scale: float = 1.0,
           rho_bar: float = 1.0, c_bar: float = 1.0):
    """V-trace (Espeholt et al. 2018, IMPALA) targets and policy-gradient advantages of one off-policy chunk (pool data:
    collected by an older policy version). logp_cur: log pi(a_t | x_t) of the policy at the start of this update (the
    proximal policy), v_cur: its values V(x_t), v_last: V of the row after the chunk. With the behaviour log-probs mu of
    the chunk: rho_t = min(rho_bar, pi / mu), c_t = lam * min(c_bar, pi / mu) (lam as in GAE);
        v_s = V(x_s) + delta_s + gamma c_s (v_{s+1} - V(x_{s+1})),  delta_s = rho_s (r_s + gamma V(x_{s+1}) - V(x_s));
        adv_s = rho_s (r_s + gamma v_{s+1} - V(x_s)).
    Terminated steps bootstrap 0, truncated ones V of the row after them (no trace across either). Returns (adv, v_s,
    rho) as float32 arrays. On-policy (pi = mu, rho_bar = c_bar = 1) v_s is GAE(lam)'s return with the current values;
    the advantage is IMPALA's (r_s + gamma v_{s+1} - V(x_s)), the form APPO-style learners use with PPO's clipping."""
    r = chunk["rewards"].astype(np.float64) * reward_scale
    v = np.asarray(v_cur, np.float64)
    T = len(r)
    ratio = np.exp(np.asarray(logp_cur, np.float64) - chunk["logp"].astype(np.float64))
    rho = np.minimum(rho_bar, ratio)
    c = lam * np.minimum(c_bar, ratio)
    term, trunc = np.asarray(chunk["terminated"], bool), np.asarray(chunk["truncated"], bool)
    v_next = np.r_[v[1:], float(v_last)]
    nv = np.where(term, 0.0, v_next)
    cont = (~term & ~trunc).astype(np.float64)
    delta = rho * (r + gamma * nv - v)
    acc = 0.0
    vs_v = np.zeros(T, np.float64)
    for t in range(T - 1, -1, -1):
        acc = delta[t] + gamma * c[t] * cont[t] * acc
        vs_v[t] = acc
    vs = v + vs_v
    vs_next = np.where(term, 0.0, np.where(trunc, v_next, np.r_[vs[1:], float(v_last)]))
    adv = rho * (r + gamma * vs_next - v)
    return adv.astype(np.float32), vs.astype(np.float32), rho.astype(np.float32)


def ppo_update(model: ActorCritic, opt, batch: dict, epochs: int, minibatch: int, clip: float, vf_coef: float,
               ent_coef: float, max_grad_norm: float, device="cpu", gen=None, target_kl: float | None = None,
               teacher_logp=None, teacher_mask=None, teacher_coef: float = 0.0) -> dict:
    """teacher_logp (n, n_act) log-probabilities of a frozen teacher policy on the batch states, teacher_mask (n,) bool:
    the loss gets + teacher_coef * KL(teacher || policy) averaged over the masked states of each minibatch (e.g. keep the
    back half of a level near a faster earlier policy while the front changes); stats teacher_kl / teacher_n."""
    obs = torch.as_tensor(batch["obs"], device=device)
    act = torch.as_tensor(batch["actions"], dtype=torch.long, device=device)
    old_logp = torch.as_tensor(batch["logp"], device=device)
    adv = torch.as_tensor(batch["adv"], device=device)
    ret = torch.as_tensor(batch["ret"], device=device)
    n = obs.shape[0]
    stats = {"pg_loss": [], "v_loss": [], "entropy": [], "approx_kl": [], "clip_frac": [], "grad_norm": []}
    use_teacher = teacher_logp is not None and teacher_mask is not None and teacher_coef > 0
    if use_teacher:
        t_logp = torch.as_tensor(teacher_logp, dtype=torch.float32, device=device)
        t_mask = torch.as_tensor(teacher_mask, dtype=torch.bool, device=device)
        stats["teacher_kl"] = []
    stop = False
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen).to(device)
        for i in range(0, n, minibatch):
            idx = perm[i:i + minibatch]
            logits, v = model(obs[idx])
            logp_all = torch.log_softmax(logits, -1)
            logp = logp_all.gather(-1, act[idx, None]).squeeze(-1)
            ent = -(logp_all.exp() * logp_all).sum(-1).mean()
            a = adv[idx]
            a = (a - a.mean()) / (a.std() + 1e-8)
            ratio = (logp - old_logp[idx]).exp()
            pg = -torch.min(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a).mean()
            vl = 0.5 * ((v - ret[idx]) ** 2).mean()
            loss = pg + vf_coef * vl - ent_coef * ent
            if use_teacher:
                m = t_mask[idx]
                if bool(m.any()):
                    lt = t_logp[idx][m]
                    kl_t = (lt.exp() * (lt - logp_all[m])).sum(-1).mean()
                    loss = loss + teacher_coef * kl_t
                    stats["teacher_kl"].append(kl_t.item())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # total norm before clipping
            opt.step()
            with torch.no_grad():
                lr = logp - old_logp[idx]
                kl = ((lr.exp() - 1) - lr).mean().item()
                stats["pg_loss"].append(pg.item())
                stats["v_loss"].append(vl.item())
                stats["entropy"].append(ent.item())
                stats["approx_kl"].append(kl)
                stats["clip_frac"].append(((ratio - 1).abs() > clip).float().mean().item())
                stats["grad_norm"].append(float(gn))
            if target_kl is not None and kl > 1.5 * target_kl:
                stop = True
                break
        if stop:
            break
    out = {k: float(np.mean(v)) if v else float("nan") for k, v in stats.items()}
    out["early_stop"] = float(stop)
    if use_teacher:
        out["teacher_n"] = float(t_mask.sum().item())
        out["teacher_coef"] = float(teacher_coef)
    return out


def explained_variance(pred: np.ndarray, target: np.ndarray) -> float:
    vt = np.var(target)
    return float("nan") if vt == 0 else float(1 - np.var(target - pred) / vt)

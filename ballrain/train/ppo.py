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

import copy

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
    def __init__(self, n_obs: int, n_act: int, hidden=(256, 256), act: str = "tanh", actor_heads: str = "single"):
        super().__init__()
        a = {"tanh": nn.Tanh, "relu": nn.ReLU}[act]
        self.pi = _mlp(n_obs, hidden, n_act, a, 0.01)
        self.v = _mlp(n_obs, hidden, 1, a, 1.0)

        if actor_heads not in ("single", "ball_type"):
            raise ValueError(f"unsupported actor_heads {actor_heads!r}")
        self.actor_heads = actor_heads
        if actor_heads == "ball_type":
            if n_obs < 4:
                raise ValueError("ball_type needs observation columns 1:4")
            # Head zero stays at its legacy name; append parameters AFTER the critic for Adam migration.
            self.extra_heads = nn.ModuleList(copy.deepcopy(self.pi[-1]) for _ in range(2))

    @staticmethod
    def context(x):
        # make_layout always begins with valid, ball_type[3]. RMS of a binary column
        # preserves the sign: a present bit is positive, absent bits nonpositive.
        strength, index = x[..., 1:4].max(dim=-1)
        return torch.where(strength > 0, index, 0)

    def forward(self, x):
        if self.actor_heads == "single":
            logits = self.pi(x)
        else:
            ctx = self.context(x)
            h = self.pi[:-1](x)
            heads = (self.pi[-1], *self.extra_heads)
            # Keep the dense GEMM shape identical to legacy. Do not even call unused
            # heads: grad=None means Adam skips their moments as well as their weights.
            active = ctx.unique().tolist()
            logits = heads[active[0] if active else 0](h)
            for i in active[1:]:
                logits = torch.where((ctx == i).unsqueeze(-1), heads[i](h), logits)
        return logits, self.v(x).squeeze(-1)

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
               teacher_logp=None, teacher_mask=None, teacher_coef: float = 0.0,
               anchor_logp=None, anchor_weight=None, anchor_coef: float = 0.0, value_only: bool = False,
               demo_obs=None, demo_act=None, demo_coef: float = 0.0, demo_batch: int = 256) -> dict:
    """teacher_logp (n, n_act) log-probabilities of a frozen teacher policy on the batch states, teacher_mask (n,) bool:
    the loss gets + teacher_coef * KL(teacher || policy) averaged over the masked states of each minibatch (e.g. keep the
    back half of a level near a faster earlier policy while the front changes); stats teacher_kl / teacher_n.

    anchor_logp (n, n_act) teacher log-probabilities and anchor_weight (n,) row weights summing to 1 (the mastered-state
    anchor, ballrain.train.anchor.row_weights; 0 = not anchored): the loss gets + anchor_coef * sum_r w_r KL_r over the
    rows of each minibatch, scaled by n / minibatch rows, so that over an epoch it is the weighted batch sum (a weighted
    mean over families, not over rows); stats anchor_kl (that term) / anchor_n (rows with w > 0) / anchor_coef.
    Independent of the teacher term (both can be on).

    Non-finite guard (gradient clipping does not help: one NaN / inf gradient makes every parameter NaN): a minibatch
    whose loss or gradient norm is not finite is not stepped, the parameters and the optimiser state of the start of
    the update are restored (earlier minibatches of this update are undone) and the update ends; the same when a
    parameter is not finite after the last minibatch. The stats then carry "nonfinite": {"term" (pg_loss, v_loss,
    entropy, teacher_kl, grad_norm or params), "epoch", "minibatch", "rows" (batch rows of that minibatch), the values
    of the terms}, and the caller must treat the update as not done. A finite update is unchanged (no extra key)."""
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
    use_anchor = anchor_logp is not None and anchor_weight is not None and anchor_coef > 0
    if use_anchor:
        a_logp = torch.as_tensor(anchor_logp, dtype=torch.float32, device=device)
        a_w = torch.as_tensor(anchor_weight, dtype=torch.float32, device=device)
        use_anchor = bool((a_w > 0).any())
        if use_anchor:
            stats["anchor_kl"] = []
    # imitation term (train_m2 --demo): + demo_coef * cross-entropy of the policy against a demonstration's actions on
    # its states (normalised observations), a random demo_batch of them per minibatch, beside the PPO loss
    use_demo = demo_obs is not None and demo_act is not None and demo_coef > 0 and not value_only and len(demo_act) > 0
    if use_demo:
        d_obs = torch.as_tensor(demo_obs, dtype=torch.float32, device=device)
        d_act = torch.as_tensor(demo_act, dtype=torch.long, device=device)
        stats["demo_ce"], stats["demo_acc"] = [], []
    snap = {k: v.detach().clone() for k, v in model.state_dict().items()}  # restored when the update turns non-finite
    opt_snap = copy.deepcopy(opt.state_dict())
    stop, nonfinite, ep = False, None, 0
    # Minibatch bounds. A batch filled with pool chunks is not a multiple of the minibatch; a short remainder (under
    # half a minibatch) joins the previous minibatch instead of being stepped on its own: the per-minibatch advantage
    # normalisation divides by the std of the minibatch, which is NaN for a single row (the r8 NaN at u16870, a batch
    # of 8193 rows) and noise for a handful. A multiple of the minibatch is split exactly as before.
    bounds = [(i, min(i + minibatch, n)) for i in range(0, n, minibatch)]
    if len(bounds) > 1 and bounds[-1][1] - bounds[-1][0] < minibatch // 2:
        bounds[-2:] = [(bounds[-2][0], n)]
    for ep in range(epochs):
        perm = torch.randperm(n, generator=gen).to(device)
        for mb, (i, j) in enumerate(bounds):
            idx = perm[i:j]
            logits, v = model(obs[idx])
            logp_all = torch.log_softmax(logits, -1)
            logp = logp_all.gather(-1, act[idx, None]).squeeze(-1)
            ent = -(logp_all.exp() * logp_all).sum(-1).mean()
            a = adv[idx]
            a = (a - a.mean()) / (a.std() + 1e-8)
            ratio = (logp - old_logp[idx]).exp()
            pg = -torch.min(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a).mean()
            vl = 0.5 * ((v - ret[idx]) ** 2).mean()
            # value_only (a value warm-up after a reward change): the actor gets no gradient (its .grad stays None, so
            # Adam leaves it and its moments untouched); the policy terms below are skipped too
            loss = vf_coef * vl if value_only else pg + vf_coef * vl - ent_coef * ent
            if use_teacher and not value_only:
                m = t_mask[idx]
                if bool(m.any()):
                    lt = t_logp[idx][m]
                    kl_t = (lt.exp() * (lt - logp_all[m])).sum(-1).mean()
                    loss = loss + teacher_coef * kl_t
                    stats["teacher_kl"].append(kl_t.item())
            if use_demo:
                di = torch.randint(0, len(d_act), (min(demo_batch, len(d_act)),), generator=gen).to(device)
                dl, _ = model(d_obs[di])
                ce = nn.functional.cross_entropy(dl, d_act[di])
                loss = loss + demo_coef * ce
                stats["demo_ce"].append(ce.item())
                stats["demo_acc"].append(float((dl.argmax(-1) == d_act[di]).float().mean()))
            if use_anchor and not value_only:
                wa = a_w[idx]
                ma = wa > 0
                if bool(ma.any()):
                    la = a_logp[idx][ma]
                    kl_a = (wa[ma] * (la.exp() * (la - logp_all[ma])).sum(-1)).sum() * (n / len(idx))
                    loss = loss + anchor_coef * kl_a
                    stats["anchor_kl"].append(kl_a.item())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # total norm before clipping
            if not (torch.isfinite(loss) and torch.isfinite(gn)):  # no step: back to the start of the update (below)
                terms = {"pg_loss": pg.item(), "v_loss": vl.item(), "entropy": ent.item()}
                if use_teacher and bool(m.any()):
                    terms["teacher_kl"] = kl_t.item()
                if use_anchor and bool(ma.any()):
                    terms["anchor_kl"] = kl_a.item()
                terms["grad_norm"] = float(gn)
                nonfinite = {"term": next((k for k, x in terms.items() if not np.isfinite(x)), "loss"), "epoch": ep,
                             "minibatch": mb, "rows": idx.cpu().numpy(), **terms}
                break
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
        if stop or nonfinite is not None:
            break
    if nonfinite is None and not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        nonfinite = {"term": "params", "epoch": ep, "minibatch": -1, "rows": None}
    if nonfinite is not None:
        model.load_state_dict(snap)
        opt.load_state_dict(opt_snap)
        opt.zero_grad(set_to_none=True)
    out = {k: float(np.mean(v)) if v else float("nan") for k, v in stats.items()}
    out["early_stop"] = float(stop)
    if nonfinite is not None:
        out["nonfinite"] = nonfinite
    if use_teacher:
        out["teacher_n"] = float(t_mask.sum().item())
        out["teacher_coef"] = float(teacher_coef)
    if use_anchor:
        out["anchor_n"] = float((a_w > 0).sum().item())
        out["anchor_coef"] = float(anchor_coef)
    return out


def explained_variance(pred: np.ndarray, target: np.ndarray) -> float:
    vt = np.var(target)
    return float("nan") if vt == 0 else float(1 - np.var(target - pred) / vt)

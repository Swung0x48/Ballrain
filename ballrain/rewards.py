"""Reward functions as configuration (docs/observations.md, section "奖励").

SR mode: progress along the reference path per in-game time, finish bonus, death penalty and
potential-based shaping F = gamma * Phi(s') - Phi(s) with Phi = shaping_weight * arc length and
Phi(terminal) = 0. Progress made while the ball has no floor support (airborne: falls, jumps, flying off the road)
is held and paid at the next supported step (a landing) or at LevelFinish, and dropped at BallOff
(hold_unsupported), so flying into the void earns nothing (the projection of a falling ball can advance by up to the
window's 30 units of arc per step across a bend). At BallOff the progress paid since the sector started (in this
episode) is taken back (death_refund): the respawn is at the last checkpoint, so a death undoes it.
HS mode: delta of the score potential Points + Lifes * LifeBonus (time decay, +100 when a score ball
is touched, +20 per small ball reaching the player ball, life balls), at finish the level bonus plus
20 x small balls still following (the official readout, docs/benchmarks.md), a death penalty of one
LifeBonus at BallOff (the game subtracts the life later, while the RL episode is already over),
optional potential-based follower shaping and optional path shaping. Small balls lost at a checkpoint
never pay. Pausing costs no score (the game rule): the env caps pauses (ballrain.env).

Time always comes from the TIME section's SR double sum (the BML+ 0.3.12 timer), not frame counts.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .state import SMALL_BALL_POINTS, GameTracker


@dataclass
class RewardConfig:
    mode: str = "sr"  # "sr" | "hs"
    gamma: float = 0.99
    # SR
    progress_weight: float = 0.1  # per unit of arc length gained
    time_weight: float = 1.0  # per second of IGT
    finish_bonus: float = 10.0
    death_penalty: float = 5.0
    shaping_weight: float = 0.0  # Phi = shaping_weight * arc length
    backward_scale: float = 1.0  # multiplier for negative progress
    hold_unsupported: bool = True  # hold progress made without floor support until landing / finish, drop it at BallOff
    death_refund: float = 1.0  # at BallOff take back this fraction of the progress paid since the sector started
    checkpoint_bonus: float = 0.0
    # pay the arc a re-acquired projection moves (ballrain.path jump=True) like any other step (through
    # hold_unsupported): a shortcut that lands further along the line (dropping from an upper turn onto a lower
    # part of the route) is paid for the arc it skips, a fall back onto an earlier part is charged, and the progress
    # of an episode telescopes to s_end - s_start. False = the old behaviour (a jump step pays 0, s only moves)
    pay_jumps: bool = True
    # HS
    score_scale: float = 0.01  # reward per score point
    hs_death_lifebonus: bool = True  # penalise one LifeBonus at BallOff
    hs_progress_weight: float = 0.0
    # potential-based shaping Phi = hs_follower_shaping * score_scale * 20 * followers (0 at terminal):
    # pays the pending small balls when they start following, takes it back when they are lost
    hs_follower_shaping: float = 0.0

    def to_dict(self):
        return asdict(self)


class Reward:
    """Stateful reward over one RL episode (one life)."""

    def __init__(self, config: RewardConfig | None = None):
        self.cfg = config or RewardConfig()
        if self.cfg.mode not in ("sr", "hs"):
            raise ValueError(f"unknown reward mode {self.cfg.mode!r}")
        self.reset_state()

    def reset_state(self):
        self.s = None
        self.held = 0.0  # progress made without support, not paid yet
        self.sector_paid = 0.0  # progress paid since the episode start or the last checkpoint
        self.sr = None
        self.score = None
        self.checkpoints = None
        self.followers = 0
        self.total = 0.0
        self.terms: dict = {}

    def begin(self, tracker: GameTracker, s: float | None):
        """Call at the start of the episode (after the reset observation)."""
        self.reset_state()
        self.s = s
        self.sr = tracker.sr_ms or 0.0
        self.score = tracker.hs_partial()
        self.checkpoints = tracker.checkpoints
        self.followers = tracker.followers()

    def _add(self, name, v):
        self.terms[name] = self.terms.get(name, 0.0) + v
        return v

    def step(self, tracker: GameTracker, s: float | None, died: bool, finished: bool, terminal: bool,
             rebase: bool = False, supported: bool | None = None) -> tuple[float, dict]:
        """s: arc length of the ball's projection (None when unknown). rebase: the projection was re-acquired
        this step (ballrain.path, jump=True): paid like any step when pay_jumps (default), otherwise no progress
        reward for it and s only moves the reference.
        supported: the ball has floor support after this step (features info["supported"]); None = unknown,
        treated as supported. Returns (reward, terms)."""
        c = self.cfg
        terms = {}
        sr = tracker.sr_ms if tracker.sr_ms is not None else self.sr
        dt_s = (sr - self.sr) / 1000.0 if self.sr is not None else 0.0
        ds = (s - self.s) if (s is not None and self.s is not None and (c.pay_jumps or not rebase)) else 0.0
        if ds < 0:
            ds *= c.backward_scale
        r = 0.0
        if tracker.checkpoints > self.checkpoints:
            self.sector_paid = 0.0
        if c.mode == "sr":
            pay = ds
            if c.hold_unsupported:
                self.held += ds
                pay = 0.0
                if died:  # forfeited; backward moves are never forgiven
                    pay, self.held = min(self.held, 0.0), 0.0
                elif finished or supported is None or supported:
                    pay, self.held = self.held, 0.0
            terms["progress"] = c.progress_weight * pay
            self.sector_paid += c.progress_weight * pay
            terms["time"] = -c.time_weight * dt_s
            if c.shaping_weight and s is not None and self.s is not None:
                phi0 = c.shaping_weight * self.s
                phi1 = 0.0 if terminal else c.shaping_weight * s
                terms["shaping"] = c.gamma * phi1 - phi0
            if c.checkpoint_bonus and tracker.checkpoints > self.checkpoints:
                terms["checkpoint"] = c.checkpoint_bonus * (tracker.checkpoints - self.checkpoints)
            if finished:
                terms["finish"] = c.finish_bonus
            if died:
                terms["death"] = -c.death_penalty
                if c.death_refund:
                    terms["death_refund"] = -c.death_refund * max(self.sector_paid, 0.0)
        else:
            score = tracker.hs_partial()
            terms["score"] = c.score_scale * (score - self.score)
            self.score = score
            if c.hs_progress_weight:
                terms["progress"] = c.hs_progress_weight * ds
            if finished:
                terms["finish"] = c.score_scale * tracker.level_bonus
                pending = tracker.followers_at_finish
                if pending is None:
                    pending = tracker.followers()
                terms["followers_finish"] = c.score_scale * SMALL_BALL_POINTS * pending
            if died and c.hs_death_lifebonus:
                terms["death"] = -c.score_scale * tracker.life_bonus
            followers = tracker.followers()
            if c.hs_follower_shaping:
                phi0 = c.hs_follower_shaping * c.score_scale * SMALL_BALL_POINTS * self.followers
                phi1 = 0.0 if terminal else c.hs_follower_shaping * c.score_scale * SMALL_BALL_POINTS * followers
                terms["follower_shaping"] = c.gamma * phi1 - phi0
            self.followers = followers
        for k, v in terms.items():
            r += self._add(k, v)
        self.sr = sr
        if s is not None:
            self.s = s
        self.checkpoints = tracker.checkpoints
        self.total += r
        return r, terms

"""Non-finite guard of the learner (scripts/train_m2.py): the learner never applies or publishes a NaN / inf update.

Checks of the batch arrays before an update (check), ppo_update behind them (update: ppo.ppo_update restores its
parameters and optimiser state when a loss or gradient norm is not finite), a last check of everything saved or
published (nonfinite_keys). Every skip is one log line and one dump nonfinite-u<update>.npz in the run directory (at
most MAX_DUMPS per run directory); more than MAX_CONSECUTIVE skips in a row stop the learner (NonFiniteError).

The dump (np.load, no pickle) holds only arrays: the batch the update consumed (obs, actions, logp, adv, ret, values,
teacher terms), per row the chunk it came from and whether that is a pool chunk, the chunks' rewards / behaviour
log-probs / flags (the inputs of GAE and V-trace), per chunk the worker id, backend label, policy version and lag and
the pool task id / sequence number, what was non-finite (where, term, epoch, minibatch, the rows) and the parameters
and optimiser state the update started from. No hosts, copies, paths or keys.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ballrain.train.ppo import ppo_update

MAX_DUMPS = 5  # nonfinite-u*.npz per run directory: a persistent problem must not fill the disk
MAX_CONSECUTIVE = 20  # skipped updates in a row before the learner stops instead of spinning


class NonFiniteError(RuntimeError):
    """The learner must stop: too many non-finite updates in a row, or non-finite parameters to save / publish."""


def _leaves(tree, prefix: str = ""):
    """(name, array) of the tensors, arrays and numbers of a nested state (model / optimiser state_dict, rms)."""
    if isinstance(tree, dict):
        for k, v in tree.items():
            yield from _leaves(v, f"{prefix}{k}.")
    elif isinstance(tree, (list, tuple)):
        for i, v in enumerate(tree):
            yield from _leaves(v, f"{prefix}{i}.")
    elif tree is not None:
        a = np.asarray(tree.detach().cpu().numpy() if hasattr(tree, "detach") else tree)
        if a.dtype.kind in "biuf":
            yield prefix[:-1], a


def nonfinite_keys(state) -> list:
    """Names of the non-finite entries of a nested state, e.g. {"model": model.state_dict(), "rms": ...}."""
    return [k for k, a in _leaves(state) if a.dtype.kind == "f" and not np.isfinite(a).all()]


def nonfinite_rows(arrays: dict):
    """(row mask, names): the rows with a non-finite value in one of the float arrays ((n,) or (n, k) each) and the
    names of those arrays; (None, []) when everything is finite."""
    mask, names = None, []
    for k, v in arrays.items():
        v = np.asarray(v)
        if v.dtype.kind != "f":
            continue
        ok = np.isfinite(v)
        if ok.all():
            continue
        names.append(k)
        bad = ~ok.reshape(len(v), -1).all(1)
        mask = bad if mask is None else mask | bad
    return mask, names


def chunk_table(chunks: list, version: int) -> dict:
    """Where the rows of a batch came from: per row the chunk index and the pool flag, per chunk its first row, steps,
    pool / remote flags, worker id, backend label, policy version and lag behind the learner's version, pool task id
    and sequence number; and the chunk fields GAE / V-trace read. Whitelisted fields only (no host, copy or path)."""
    steps = np.array([len(c["rewards"]) for c in chunks], np.int64)
    pool = np.array([bool(c.get("pool")) for c in chunks])
    row_chunk = np.repeat(np.arange(len(chunks)), steps)
    ver = np.array([int(c.get("version", -1)) for c in chunks], np.int64)
    return {"row_chunk": row_chunk, "row_pool": pool[row_chunk],
            "chunk_start": np.cumsum(steps) - steps, "chunk_steps": steps, "chunk_pool": pool,
            "chunk_remote": np.array([bool(c.get("remote")) for c in chunks]),
            "chunk_wid": np.array([int(c.get("wid", -1)) for c in chunks], np.int64),
            "chunk_backend": np.array([str(c.get("backend", "")) for c in chunks]),
            "chunk_version": ver, "chunk_lag": version - ver,
            "chunk_pool_task": np.array([str(c.get("pool_task", "")) for c in chunks]),
            "chunk_pool_seq": np.array([int(c.get("pool_seq", -1)) for c in chunks], np.int64),
            "chunk_last_value": np.array([float(c.get("last_value", np.nan)) for c in chunks], np.float32),
            **{f"chunk_rows_{k}": np.concatenate([np.asarray(c[k]) for c in chunks])
               for k in ("rewards", "logp", "terminated", "truncated") if all(k in c for c in chunks)}}


def describe_rows(rows, table: dict, limit: int = 5) -> str:
    """'own rows 3 (workers 1, 4), pool rows 64 (task/seq lag: 1f0c/7 2)' of a row mask or an array of row indices."""
    idx, n = np.unique(table["row_chunk"][rows], return_counts=True)
    pool = table["chunk_pool"][idx]
    own = sorted({int(w) for w in table["chunk_wid"][idx[~pool]]})
    ids = [f"{table['chunk_pool_task'][i]}/{table['chunk_pool_seq'][i]} {table['chunk_lag'][i]}" for i in idx[pool]]

    def some(v):
        return ", ".join(str(x) for x in v[:limit]) + (f", +{len(v) - limit} more" if len(v) > limit else "")

    return (f"own rows {int(n[~pool].sum())}" + (f" (workers {some(own)})" if own else "")
            + f", pool rows {int(n[pool].sum())}" + (f" (task/seq lag: {some(ids)})" if ids else ""))


class NonFiniteGuard:
    """The learner's skips of non-finite batches / updates: log line, dump, the stop after too many in a row. The
    skips are counted in counters["nonfinite_skipped"] (the learner's counters: saved with the checkpoint);
    consecutive: skips since the last finite update."""

    def __init__(self, run_dir, log=print, counters: dict | None = None, max_dumps: int = MAX_DUMPS,
                 max_consecutive: int = MAX_CONSECUTIVE):
        self.run_dir, self.log = Path(run_dir), log
        self.counters = {} if counters is None else counters
        self.counters.setdefault("nonfinite_skipped", 0)
        self.max_dumps, self.max_consecutive = max_dumps, max_consecutive
        self.consecutive = 0

    @property
    def skipped(self) -> int:
        return self.counters["nonfinite_skipped"]

    def check(self, update: int, chunks: list, arrays: dict, model=None, opt=None) -> bool:
        """True (logged, dumped, counted: skip update number `update`) when a batch array has a non-finite value."""
        rows, names = nonfinite_rows(arrays)
        if not names:
            return False
        self._skip(update, "batch", ",".join(names), chunks, arrays, {"bad_rows": rows}, model, opt)
        return True

    def update(self, update: int, chunks: list, extra: dict, model, opt, batch: dict, *args, **kw):
        """ppo_update(model, opt, batch, *args, **kw) behind the checks; extra: further row arrays to check and dump
        (the old values). Returns its stats, or None when update number `update` was skipped: a non-finite batch
        (nothing was run) or a non-finite loss / gradient / parameter inside (ppo_update restored the parameters and
        the optimiser state); either way the model and the optimiser are as before the call."""
        arrays = {**batch, **extra, **{k: kw[k] for k in ("teacher_logp", "teacher_mask", "anchor_logp", "anchor_weight")
                                   if kw.get(k) is not None}}
        if self.check(update, chunks, arrays, model, opt):
            return None
        stats = ppo_update(model, opt, batch, *args, **kw)
        nf = stats.pop("nonfinite", None)
        if nf is None:
            self.consecutive = 0
            return stats
        rows = nf.pop("rows")
        detail = {"epoch": nf.pop("epoch"), "minibatch": nf.pop("minibatch"),
                  **({} if rows is None else {"minibatch_rows": rows}),
                  **{f"term_{k}": v for k, v in nf.items() if k != "term"}}
        self._skip(update, "update", nf["term"], chunks, arrays, detail, model, opt)
        return None

    def _skip(self, update, where, term, chunks, arrays, detail, model, opt):
        self.counters["nonfinite_skipped"] += 1
        self.consecutive += 1
        table = chunk_table(chunks, update - 1)  # update - 1: the learner's version while the chunks were collected
        rows = detail.get("bad_rows", detail.get("minibatch_rows"))
        at = f"epoch {detail['epoch']} minibatch {detail['minibatch']}, its " if "epoch" in detail else ""
        src = "" if rows is None else f"; {at}{describe_rows(rows, table)}"
        state = {"model": model.state_dict() if model is not None else None,
                 "opt": opt.state_dict() if opt is not None else None}
        dump = self._dump(update, {"update": update, "where": where, "term": term, **detail, **table, **arrays,
                                   **dict(_leaves(state))})
        self.log(f"NON-FINITE {where} at u{update} ({term}): update skipped, parameters unchanged{src}; {dump}; "
                 f"skipped {self.skipped}, in a row {self.consecutive}")
        if self.consecutive > self.max_consecutive:
            raise NonFiniteError(f"{self.consecutive} non-finite updates in a row (limit {self.max_consecutive}): "
                                 "the learner stops; see the nonfinite-u*.npz dumps of the run")

    def _dump(self, update: int, arrays: dict) -> str:
        if len(list(self.run_dir.glob("nonfinite-u*.npz"))) >= self.max_dumps:
            return f"no dump (the run has {self.max_dumps} already)"
        f = self.run_dir / f"nonfinite-u{update}.npz"
        tmp = f.with_name(f.name + ".tmp")
        try:  # numbers, strings and arrays only: nothing in the file needs pickle
            out = {k: np.asarray(v) for k, v in arrays.items()}
            with open(tmp, "wb") as fh:
                np.savez_compressed(fh, **{k: v for k, v in out.items() if v.dtype.kind in "biufU"})
            os.replace(tmp, f)
        except Exception as e:  # noqa: BLE001  (a failed dump must not stop the learner)
            tmp.unlink(missing_ok=True)
            return f"dump failed ({type(e).__name__})"
        return f"dump {f.name}"

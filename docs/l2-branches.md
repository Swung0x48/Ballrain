# Learner-owned multi-branch routes (wire v2)

A branch describes how to navigate, not a new observation layout or a new reward budget. With no branch pin the feature extractor follows its original path. A directory containing only the original `L<nn>.npz`, `L<nn>.json` and `drive.npz` uses the v1 reader, follower and digest unchanged; see `l2-stone-branch.md` for the station, timing, replay and rule formats.

## On disk and on the wire

`branches.json` is UTF-8, LF. Its object has `version: 2` and a nonempty **ordered** array `branches`. Each ID names a child directory containing the three v1 files. Every child must pair with the **same** main file (supplied separately). A line is a whole-level splice, not just a waypoint list: points are world `(x,y,z)` **ball centres**, `sector` labels each point, `t_ms` is a strictly increasing game-time clock. `main_s`, `branch_s`, `stations` in the child description are in metres, never TAS frames. Rules are per child; the v1 rule names, types, defaults and units still apply. The installed L2 example is `benchmarks/refpaths-alt/L02-multi/branches.json`.

| Member field | Meaning |
|---|---|
| `id` | Unique lower-case ASCII identifier matching `[a-z][a-z0-9-]*`; no Windows device names. No paths from the sender. |
| `phases` | Nonempty ordered sequence of `{when, local_s:[lo,hi]}`. Spans are contiguous and positive, start/end at vertices of the branch, and partition `branch_s`. The array index is the phase. Optional `from:[?]` names earlier eligible predecessor indices and must include the immediately previous one. |
| `when` | `ball` (wood=0, paper=1, stone=2), plus **at least one** of `sector`, `s:[lo,hi]` or `box:[[xmin,ymin,zmin],[xmax,ymax,zmax]]`. All supplied predicates are ANDed, bounds inclusive. `s` always refers to the currently active physical line **before** transition. No expressions, callbacks, host filenames or code. |
| `merge` | Same predicates (ball optional), tested in the final phase. On match resume the main reference. Leaving an explicitly gated sector without entering the next phase also releases the branch. |
| `legacy:true` | Alternative to `phases` and `merge`, not an extra flag on them. Use the original material/sector/branch-projection interval to select the child. Used for the south loop; v1 bundles need no manifest or conversion. |

The Python envelope is `{version:2, level, manifest:bytes, members:[v1-envelope,…]}`. The order is the manifest order, not directory sort order. Compute `H=SHA256`; use **raw 32-byte digests**, not their printed hex, in concatenations:

```
child[i] = H(H(line_npz) || H(desc_json) || H(drive_npz))
feature_pin = hex(H(H(manifest_bytes) || child[0] || ... || child[last]))
```

Validate all child hashes, level, main pin, branch line pin in both descriptor and drive, rules and feature pin. Do not canonicalize JSON, recompress NPZ or hash absolute paths. The learner sends original bytes and keeps them in the checkpoint; workers install into private digest-specific storage. The main reference path/sha handshake is separate. Python bridges advertise `branch_assets_version:1`, `branch_rules_version:1`, `multi_branch_v2:true`. Unknown readers must refuse before consuming a task; do **not** enable native jobs merely by advertising the hello bit.

## Per valid observation, per life

1. Start on main with all branches eligible. In manifest order, the first matching entrance wins. The first `when` sees the current main projection. While latched, evaluate only this branch. Invalid observations during a transform do not advance it. For a phase use that phase's vertex range for projection (nearest 3-D segment; prefer the continuity window if within `reacquire` of the global minimum). Waypoints, speed and lateral error are on the physical line, not the reward axis.
2. Test later phases in order on the old phase projection, accepting only an eligible predecessor (`from`, default: immediately previous phase). L2 can accept a genuine early return to the wood transformer without demanding that the ball first visit the eastern pit. Matching a later ball material in some other region is not sufficient. Consume at most one transition per update. Final merge marks this phased branch done for the life; jitter at the merge does not reopen it. Legacy selection remains reversible. Respawn, new life or a fresh TAS handover clears the branch, phase, projections, done set and ledger.
3. Keep **one** ledger across all lines. Piecewise-linearly map local station onto main station using `[0,0]`, the splice endpoints, intermediate pairs and the two line ends. Pay only increases beyond the previous high-water mark. When switching, absorb any projection jump without payment; interpolate from `(peak,credit)` to `(next common station,next common station)`. At a phase change or regular final merge, first account for physical travel on the **old** phase, then absorb the new projection. Never add each branch's cumulative progress together. Backtracking and subsequent retraversal do not repay the same interval. At full traversal the budget is the replaced main span. The curriculum frontier is still a TAS frame.
4. Elapsed reference time integrates only newly traversed local arc against its line's `t_ms`; no time payment for switching or reversing. Remaining reference time uses the active clock from the local equivalent of the peak to the finish. `branch_id`, `phase`, `local_s` and the two reference clocks are telemetry, not new observation components. Stall rules come from the active member, not the last member loaded.

## Rest, resume and native parity

`--rest-starts lo-hi@branch:<id>:<local_s>` selects a child; use its exact `prefix_frame` for both ends. The old `lo-hi@branch:<s>` selects the unique legacy member. Replay actual `(delta,keys)` from the record-hash-matched TAS prefix up to the nearest sampled station; at a phase boundary the newer phase owns the station. Drive/brake and verify actual material, sector, final support (if required), 3-D position, horizontal speed and phase before handover. Seed only the verified route cursor in the training extractor, never a game pose or velocity. Failure closes that process and obeys that child's `rest_fallback`.

For resume, use the original launch arguments and `--resume --path-dir benchmarks/refpaths-v2 --branch-dir benchmarks/refpaths-alt/L02-multi --branch-update` **once**. Old pin and old bytes must agree even during migration. New checkpoints contain the replacement bytes and pin. Later omit both branch flags. The change is printed and appended to `branch-updates.jsonl` in the run directory. This does not migrate a live learner or stale queued chunks.

Native port checklist: implement envelope/child validation, capability gating and pin-keyed installation; match closest-segment ties, continuity and gates; per-life reset, phase-restricted projection and ordered transitions; station ledger and clocks; identical physical features and active rules; record/drive verification, rest/fallback and restored cursor; update policy/task/cache identity and reject old or mixed samples. Use `tests/test_multi_branch.py`, `test_branch_distribution.py`, `test_stone_branch.py` and the two-build capture logs as parity fixtures. Native pool export remains closed until that port exists.

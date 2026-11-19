---
name: route-scout
description: Search a Ballance level offline for engine-observed route families, rank checkpoint splits for SR or HS, resume a scout archive, or independently replay its candidates. Use for route discovery, not live training or human-demo analysis.
---

Run from the repository root with its Python environment. The search needs the installed game data (`base.cmo`) and an **agent-capable BallanceMMOSimTool**. It does not start Player.exe or use a training pool. If the runtime is outside the checkout, set `BALLRAIN_HL_ROOT` to the game folder and `BALLRAIN_HL_BIN` to the search executable (or pass `--root` and `--build`).

### Choose the start and budget

Reserve one spare label with the local operator: `hlw-91`, `hlw-92`, `hlw-93`, `hlw-98` or `hlw-99`. These are fresh local TCP headless processes, not connections to an existing service. `hlw-91` is the search default. Never borrow a retail copy. Use a new, empty output directory per search; do not run simultaneous writers on it.

```console
python -m ballrain.routescout plan --level 2 --instance hlw-91 --budget-s 90 --candidates 6 --out runs/l2
python -m ballrain.routescout plan --level 2 --from-checkpoint 1 --to-checkpoint 2 --instance hlw-92 --budget-s 120 --out runs/l2-cp2
python -m ballrain.routescout plan --level 9 --spawn 0,18.65,0 --material wood --instance hlw-93 --budget-s 90 --out runs/l9
python -m ballrain.routescout plan --resume runs/l2 --instance hlw-91 --budget-s 180
python -m ballrain.routescout plan --map custom.nmo --out runs/custom
```

No start option means the first controllable state at the level's own spawn. Default target: the next checkpoint. A requested start is **physically approached**, not injected: the checkpoint count, requested material and optional position (within 0.5 world units) and camera yaw (degrees, within 2) must match. This can be a moving state. If it cannot be reached the command reports failure, not a substituted start. Preparation consumes search time but is excluded from the split clocks. For custom maps the last command deliberately errors: there is no file-loading agent handshake or validated custom-map collision, ordered reset/checkpoint and mechanism export.

The default is two minutes. To measure **cold discovery**, use a new directory and `--cold`; `--resume` is not a cold start. Allow 5 to 10 minutes for an initial coverage run, not as a promise of CP1. Without `--cold`, any matching packaged *engine-only* keys are proposed and measured afresh. On a map without matching keys, the default still has to discover the route. No human reference line is required.

```console
python -m ballrain.routescout plan --level 11 --instance hlw-91 --cold --budget-s 360 --candidates 8 --out runs/cold-11
```

Read **cells** as explored 4-unit position bins, separated by checkpoint, material and start tape. **Candidates** in the progress log are retained tradeoffs, not the cell count or a count of routes to CP. `archive.json` keeps the replayable cell prefixes and counts of exact restores, divergences and deaths. A new cell can be farther from the checkpoint; this is useful on switchbacks. If cells grow while every published candidate is partial, the scout found more of the map, **not** a checkpoint route. If coverage stalls, inspect the last supported positions and contacts in the prefixes before increasing the budget. Moving bridges, rail mounts and speed-dependent gaps can defeat a static support guide. Unknown checkpoint coordinates do not imply a checkpoint hit; measured distance is null and coverage still drives exploration.

In seven-minute first-sector trials on x86soft, L2 reached CP1 from cold; L1 and L11 expanded to over 180 cells each but produced only partial routes. These are individual runs, not success-rate estimates.

Field construction, a pending engine request and final writes may extend wall time. The map-key-checked fastreach geometry cache under `runtime/scout-fields/` is safe to reuse for a cold run: it stores geometry, not key proposals. Ctrl+C finalizes the archive; after a forced stop, resume the last atomic save. Resume adds search time; it does not verify against another build.

### Read the results

Open `summary.md` and `top-view.png`, then inspect `candidates.json`. Filter on `complete` **before comparing times**. `checkpoints` are `[number, frame relative to handover]` observed at PostCheckpoint. `segments` includes the attempted segment. A partial path is an exploration prefix, not a checkpoint route. Compare like starts and endpoints. Keys are `[physical mask, frames]` at the engine's default delta; replay `start_keys` from controllable spawn, then `keys`. The build and map are pinned by digest. `centre_path` samples are `[relative frame, [x,y,z]]`.

`sr_ms` is the difference of the engine's training SR timer; `game_ms` includes injected frames even during pauses. `hs.score_delta` is the change in points + lives × life bonus, not a finish score. `hs.pickups` records observed life events and estimated main-point pickups; `esc_banks` records follower drops with positive points across a pause. `mechanisms` lists observed contact name, reachmech type and material-specific rule. `transformer_events` records sampled material transitions, not guessed passages near a transformer. `family` encodes material sequence, touched mechanism types, rail observation, signed lateral excursion bin and pickup/bank style. No rail contact observed is not proof the entire route was on floor.

`difficulty.score` (0–100) sums cheap control-change, transformation and sampled-rail components; it is not a measured success probability. `pareto.sr` and `pareto.hs` indicate nondominance against difficulty among the returned comparable routes (partial progress is binned by remaining distance). No perturbation runs or second-build checks by default. A tiny budget may yield fewer distinct families than requested.

### Verify only when asked

```console
python -m ballrain.routescout plan --level 2 --out runs/l2-checked --instance hlw-91 --verify --second-instance hlw-99 --second-build runtime/hl-win/bin-x86p2002d/BallanceMMOSimTool.exe
python -m ballrain.routescout verify runs/l2 --second-instance hlw-99 --second-build runtime/hl-win/bin-x86p2002d/BallanceMMOSimTool.exe
```

Verification costs an additional cold launch and replay per candidate, **outside** the search budget. Use a different spare and a genuinely different executable. Review each `verification.status` and `mismatches`. Certification requires matching replay evidence and ordered terminal PostCheckpoint events. Matching an unfinished prefix does not certify a route. Do not promote a scout result to a whole-level time, deploy it, or run training as part of this workflow.

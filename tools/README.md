# tools/

Probe and analysis code behind the measurements in `docs/state-spec.md`, `docs/agent-protocol.md`,
`docs/observations.md` and `docs/benchmarks.md`. Not part of the `ballrain` package and not run by the
gates; kept so that every measured number can be reproduced. The large dumps these scripts read are not
in the repository; each section says how to regenerate them.

Scripts find the repository root from their own location (`BALLRAIN_REPO` overrides it). Game processes
are only ever started in `runtime/<copy>/Bin` through `ballrain.instances`.

## probe/

- `mod/`: the M0/M1 read-only probe mod (BallrainIO 0.2.0 replay path plus text dumps of the IVP
  synapses, mindists, contact BBs, arrays, cameras and watched objects; `state-spec.md` 0.1 and 3).
  Build it like `BallrainIO/` (x86 Release, BML+ 0.3.12 SDK and Virtools SDK paths given as the cache
  variables `BML_DIR` and `VirtoolsSDK_DIR`, e.g. from a gitignored local .bat). It uses
  `BallrainIO/3rdparty` (minhook). Install it as `BallrainIOProbe.bmodp` into a training copy instead of
  BallrainIO only for a probe run.
- `post_finish.py`: the M1 post-finish probe (needs BallrainIO >= 0.3.1). Replays a record in agent mode
  and keeps the game running idle frames after the SR tail (`BALLRAIN_POST_FINISH`), with the per-frame
  log `BALLRAIN_POST_LOG` (Energy row, followers, every data-array change, score-ball objects, ball).
  `uv run python tools/probe/post_finish.py HS01_4738_1_31_317 --copy train-00 --frames 4000 --out <dir>`

## analysis/

- `post_finish_credit.py`: reads `post.log` files of `probe/post_finish.py`; when the small balls
  still following at LevelFinish are credited, when the end-screen drain starts, and the HS readout
  (`docs/benchmarks.md`, "HS readout").
- `edge_fan_rule.py`: the f3 / f4 edge-fan rules on the TAS dataset (histogram of consecutive ray-fan
  height differences on floor contact, per-rule distributions and agreement with edge_mesh;
  `docs/observations.md` section 2). `uv run python tools/analysis/edge_fan_rule.py SR01_1_14_457_264 ...`;
  `--references` compares the f4 and f5 fan references on all control frames (floor support / airborne /
  side or overhead contact / rail / moving partner).
- `sr_reward_check.py`: the SR reward on the TAS dataset with env-like steps, old (no hold, no death refund)
  against the defaults per life (`records`), and void probes along a level's reference line (`probes`;
  `docs/observations.md` section 4). `uv run python tools/analysis/sr_reward_check.py records SR03_1_46_066_240`
- `hs_followers/`: follower rule and timing on the TAS dataset (`runtime/datasets/tas-v1`, built by
  `scripts/m1_gates.py --gates D`): `rule2.py` (visible small balls of a hidden score ball = followers;
  final count = name-score gap on every HS record), `align.py` (Points / visibility / ExtraPoint timing),
  `taildrop.py` (Points during the SR tail), `trafo.py` (transformation windows vs. the IVP ball).
- `hs_gap/`: the first HS gap analysis on the same dataset (`timeline.py` builds `tl.pkl`, then
  `perball.py`, `fate2.py`, `sections.py`, `show.py`, ...).
- `probe_m0/`: analyses of the probe mod's text dumps (layouts of the IVP structures, geometry vs.
  mindist, ray fan, cameras, SR); they read the probe output directories given in each script.
- `agentdev/`: the stub agent-mode client used before `ballrain.client` existed (`stub.py`) and its
  comparison with the probe dumps (`cmp_probe.py`, `agent-protocol.md` 8.6).
- `fastreach_fans/`: the L2 fans (P_Modul_18) in `ballrain.fastreach` (module docstring "Fans").
  `nmo_box.py` reads mesh extents from the game's `.nmo` files (the push box `Modul_18_Kollisionsquader_Mesh`
  in `3D Entities/PH/P_Modul_18.nmo`, the ball boxes in `Balls.nmo`); `fan_fit.py` checks the box test
  against the impulses in the r8 traced episodes (`runtime/m2-train-mac/r8/episodes-traced`) and fits the
  rotation-free radius; `fan_val.py <tag>` is the before/after validation (TAS doomed flags, the L2 pit,
  traced deaths and false alarms at the fans, vertical motion, latency; `FR_SRC` picks the code, `FR_EPISODES`
  freezes the episode set).

## merge/

Policy merge by distillation (`docs/policy-merge.md`): several checkpoints that each master a different part of a
level are distilled into one actor, the teacher of a state chosen by its sector (or, for a capacity test, by the data
file). `collect.py` plays a checkpoint or a composite of teachers on the headless engine from weighted record
hand-overs and saves the raw observations (also a play test: lives that reached their next checkpoint, sectors left
with every ball); `distill.py` fits the student (from a checkpoint's weights, or fresh at a given width) and writes a
checkpoint `train_m2 --resume` and `scripts/eval_m2.py` read. `models/L01-r10-m2` was made this way.

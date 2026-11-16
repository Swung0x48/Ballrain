# TASSupport (reference snapshot)

Source of the TAS replay mod used as Ballrain's acceptance environment.

- Upstream: https://github.com/doyaGu/BallanceMods (directory `TASSupport`), commit `5452e1129b53018d1eea5fec9e51e79c87b3ae86`, MIT licence (see `LICENSE`).
- This snapshot = that commit plus local changes (`local-changes.diff` is `git diff` against the commit):
  legacy `.tas` frames parsed as 8 bytes, legacy flag on auto-loaded records, diagnostic
  environment variables (`BMMO_TAS_TRACE`, `BMMO_TAS_CORE`, `BMMO_TAS_OFFSETS`, `BMMO_TAS_PC`).
- `minhook` and `miniz` are upstream submodules and are not copied here.
- `TASSupport.cpp` line 3 now includes the bare header name `"physics_rt_api.h"`; that include
  originally pointed, by an absolute local path, at BallanceMMO's
  `BallanceMMOCommon/include/physics/physics_rt_api.h` (the bridge for the x87-soft physics DLL,
  not used by BallrainIO), and the header is not part of this snapshot.
- `local-changes.diff` is the `git diff` against that commit, with the same `#include` line
  rewritten to the bare header name (it originally held an absolute local path).

BallrainIO re-implements the legacy-mode playback logic from these files. Exported `.tas`
files are accepted only when this mod (LegacyMode 1) replays them to the same trajectory.

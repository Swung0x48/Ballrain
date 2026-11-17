"""Volunteer compute pool (prototype, docs/pool-prototype.md, docs/volunteer-compute.md section 11).

Shared contract between the gateway (tools/pool/), the verifier and the player's controller:
- reasons:  rejection / revocation reason codes with the Chinese text the controller shows verbatim;
- protocol: JSON control messages (strict validation, explicit protocol version) and the binary chunk codec;
- weights:  learner checkpoint -> plain-array npz + JSON manifest, and a numpy-only policy forward;
- task:     the task description and how a client builds its BallanceEnv from it (plus the start-state assets).

Nothing here unpickles anything: client-side modules import neither pickle nor torch (torch is used only by
weights.export_checkpoint, on the learner's own trusted checkpoints).
"""

---
name: sac-train-293-diagnosis
description: "SAC run 293 diagnosis — 97 % of deaths by starvation, Q inflated ×25 by reward normalisation"
metadata:
  node_type: memory
  type: project
  originSessionId: 3b30ce76-ccc8-4012-84b6-30ff71fc73a6
  modified: 2026-08-17T08:20:09.310Z
---

Analysis of `outputs/train_293_sac` (376 k env steps, ~455 episodes, 2026-08-17):

- `s_μ` flat at 27.5 s while `max_frames_without_hit` = 30 s → **~97 % of tanks die of the
  starvation timer**, not in combat (1–2 kills/episode for 32 tanks). The dominant signal is a nearly
  constant, unpredictable −1.
- `hit/fire` = 0.42 % → 0.91 % over the run: real but very slow learning. 248 windows of 421
  contain zero hits.
- `π_μ` ≈ −Q̄ (α ≈ 0.002 makes the entropy term negligible) and rises to 15.5: Q is inflated by a
  factor ~25 by `reward_scale()` (life std of a sparse reward ≈ 0.04).

**Two constants in absolute units were silently detuned by this ×25**:
`GRAD_NORM_MAX = 1.0` (clip bit on every batch) and the α clamp `[1e-5, 1e-2]`
(entropy reduced to 0.04 % of the actor loss).

Fixes applied 2026-08-17 (branch `develop`): γ 0.997 → 0.99 and **removal of reward
normalisation** in `SacReplayBuffer` (raw rewards are already hand-calibrated
in [−1, +2]; dividing by the std of a sparse signal amplifies instead of normalising).

**How to apply:** next run, expect Q (hence `π_μ`) of order one, not 15.
Judge on `hit/fire` and `s_μ` (see ppo-reward-redesign-2026-08). Options not retained so far:
starvation timer in observation (proprioception-archi-boundary, refused), larger buffer
(300 k / 32 tanks = only ~11 episodes, ~0.02 % "hit" transitions), starvation as truncation
rather than terminal.

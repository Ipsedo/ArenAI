---
name: sac-train-411-diagnosis
description: "SAC run 411 (2026-11-08, 6 α, lr α 3e-5) at 33k steps: Q = pure entropy value (≈12, follows α), actor at max entropy (Hc 2.73 = σ≈0.9 on 4 axes, target 0.51), aim at random floor, fire 0.6→0.3/s because of the no-bootstrap cut on empty reserve"
metadata:
  node_type: memory
  type: project
  originSessionId: 51b10db1-fb50-4f0f-81d1-b1b5b214142f
  modified: 2026-11-08T17:31:10.689Z
---

Run `outputs/train_411_sac` (config: α₀ 0.1, lr α 3e-5, target_sigma 0.3, γ 0.99, batch 256, gradient_steps 1,
32 tanks, buffer 800k, image 128×64, D fixed at 250). Read 2026-11-08 at index 33k (≈1 M transitions, 33k updates, 75 min).

Findings:
- **Q ≈ 1 → 12.1 (peak at 17k) → 10.0 while α 0.10 → 0.037**: Q is only the entropy bonus
  (α·H ≈ 0.1 × 3.5 nat/step, asymptote 35); mean reward is −0.0003/step. Q follows α, not the task.
- **Hc = 2.73 flat** = exact maximum of 4 tanh-gaussians (0.683/axis at σ ≈ 0.9); target 4 × 0.129 = 0.51.
  The actor ignores ∂Q/∂a; hit 0.039 (< random floor 0.05), miss 22.7 m flat since the start.
- **fire 0.6 → 0.3 shots/s monotone, kill 1.2 → 0.44, s 60 → 40 s**: empty reserve is a no-bootstrap cut
  (commit 3a1cb788), so firing costs ≈ γ·V ≈ 10 of future entropy vs +1 per hit → the soft value
  acts as an anti-fire survival bonus. Prediction: fire → 0 as in run 392 before α is small enough.
- α speed: ÷e every ~33k steps (Adam, lr 3e-5); for α·H ≈ reward scale (α ~ 3e-4) ~190k steps (~7 h) are needed.
- UTD = 1 update / 32 transitions (gradient_steps 1 for 32 tanks); buffer already wrapped.

**Why:** Samuel wants to judge whether SAC unlocks aim after the PPO plateaus; here the lock is the
entropy/reward scale, not perception.

**How to apply:** α₀ set to 0.01 on 2026-11-08 (validated). Refused: bootstrap on empty reserve (PPO cuts too, `ppo_trainer.cpp` cuts = dones + truncateds). Validated 2026-11-08 evening:
gradient_steps 8 + batch 512 (defaults, run 414); SAC-only bootstrap would return to the table if fire → 0 is confirmed.
See sac-agent-2026-11, [ppg-train-392-diagnosis](ppg-train-392-diagnosis.md) (fire → 0), [sac-train-309-diagnosis](sac-train-309-diagnosis.md) (collapsed α, the opposite).

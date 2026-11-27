---
name: sac-train-309-diagnosis
description: "train_309_sac does not converge — hits only during the initial high-entropy phase, α collapsed in ~30k steps, 300k buffer = ~10 episodes (nearly on-policy), spray+starvation = net positive return"
metadata:
  node_type: memory
  type: project
  originSessionId: 5f4519c6-b897-4f67-9c43-2cd8ad263043
  modified: 2026-08-21T16:59:16.840Z
---

Diagnosis 2026-08-21 on `outputs/train_309_sac` (~295k env steps × 32 tanks, 330 episodes):

- hit/s: 0.02–0.06 early (α_c=0.046, Hc≈0.69) → ≈0 from ~50k steps, never back. r decreases monotonically 0.0013 → 0.0002. s_μ stuck at 27–30 s all run = starvation wall (as [sac-train-293-diagnosis](sac-train-293-diagnosis.md)).
- α_c: 5e-2 → ~3e-4 in ~30k steps (entropies converged to low targets: target_sigma, fire 2.5 %). The only hits of the run came from the near-uniform initial policy.
- Buffer: 300 000 / 32 tanks = 9 375 steps ≈ 10 episodes ≈ 3 % of the run → transitions with hits evicted from ~40k steps; SAC nearly on-policy.
- Critic: MSE ~1e-3 → RMSE ~0.03/step vs mean reward ~0.0005/step. Early on, Q targets were dominated by the entropy bonus (α·H ≈ 0.1/step vs r ≈ 0.001).
- Reward hacking: r stays positive with hit=0 — the aim gaussian (0.5×exp, ~18 shells/episode) + winner bonus offset the −1 starvation penalty → « spray without hitting then die » is a stable local optimum.
- γ=0.99 @30 Hz → effective horizon ~3 s, starvation (30 s) invisible to Q except at end of life.

**Proposed (none validated by Samuel at this stage)**: much larger buffer, α min clamp/lr to slow the collapse, curriculum to densify hits, rebalance gaussian vs starvation, higher γ. Note: gaussian φ and shell reserve already validated (ppo-reward-redesign-2026-08), entropy annealing already refused.

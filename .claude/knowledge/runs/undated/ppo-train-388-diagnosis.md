---
name: ppo-train-388-diagnosis
description: "Run 388 (γ 0.99, λ 0.95, σ₀ 0.3, compared to 385) — real gain: hit 0.10 at D=400, D rises to 550, ev 0.57; kill flat because the curriculum regulates hit and kill ∝ hit³ (5 hits on the same part to kill); periodic fire/hit dips; σ decreases and skip rises to 0.33"
metadata:
  node_type: memory
  type: project
  originSessionId: f4f60a9b-baf5-487c-aea1-2d1d583ff538
  modified: 2026-11-01T09:11:47.194Z
---

Run `outputs/train_388_ppo` (launched 2026-11-01 00h07, read at 502k steps, still running) =
385 with γ 0.994 → 0.99, λ 0.99 → 0.95 and also σ₀ 0.05 → 0.3 (so gain not attributable to γ/λ alone).

- Gain: hit 0.10 at D=350-400 vs 0.055 at D=350 for 385, D reaches 550, ev 0.25 → 0.57.
- `kill` = total over the 32 tanks per episode (mean of 16 episodes), ≈ 1.5/31 at best. Not readable
  alone: the curriculum raises D as soon as hit on probe episodes > 0.06, which brings hit down.
- kill ∝ hit^3 roughly: 0.07 at hit 0.04, 1.5 at hit 0.10 (correlation 0.89), because of HP: shell
  = 1 damage, wheels/turret/barrel 5 HP, chassis 10, a part at 0 kills. ~24 hits per kill at hit 0.10.
- Recharge +5 shells/hit: self-sustaining reserve only at hit > 0.20; lives ≈ 10-14 s, ~10 shots.
- Dips at 237k, 356k, 490k: fire → 0.5, hit 0.03-0.045, kill ≈ 0. Cause not established.
- σ 0.3 → 0.15 (Hc −0.49), skip 0 → 0.33, clip 0.36: slope seen before the regressions of [ppo-train-383-diagnosis](ppo-train-383-diagnosis.md).

- Compared to [ppo-train-345-diagnosis](ppo-train-345-diagnosis.md) (regen 1 shell/1.5 s until 2026-10-23, 16 tanks): same
  hit ~0.08, but ~38 shots per life vs ~9, hence ~2.9 hits per tank vs ~0.7, and kill 4.4 vs
  0.77. Regen raised kills through fire volume, not aim (345 hit flat).
- Options proposed 2026-11-01, not validated: recharge +10/hit (autonomy at hit > 0.10, above the
  random floor; +15 = 6.7 % ≈ floor), reduced part HP, return of regen.

**Why:** flat kill does not prove absence of learning, but the gain at fixed D is not measured.

**How to apply:** judge on a control run at fixed D (final_spawn = initial_spawn disables the
curriculum) rather than on `kill` at moving D. Any HP/curriculum change: to be validated by Samuel.

## From ElMatador memory

Distinct run from the 2026-11-01 one above (ElMatador run numbering; this 388 dates from early September 2026). Note dated 2026-09-05.
- Commit `6c7483ae`: spawn curriculum 500→2000 m linear over 6000 episodes + hit_received 0.15→0.3. Kills rise to ~5.2 (row 1000) then decline continuously to 3.1 (row 5000) while miss grows 38→72 m.
- Cause: curriculum linear per episode, blind to performance; hit-reward tolerance grows as √d while miss grows as d at constant angular precision ⇒ the task hardens faster than the agent learns ⇒ inverted-U unlearning.
- Reference 387 (fixed 500 m spawn, otherwise identical config): 7.4 kills at row 1800 vs 4.8 for 388.

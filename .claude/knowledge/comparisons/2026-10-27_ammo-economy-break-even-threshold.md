# Ammo economy break-even threshold (env since 2026-10-25)

Runs: 430, 437, 441

Env since 2026-10-25 (20 shells, +5 per hit, max 30, death −1 at empty reserve): break-even hit ≥ 20 % vs observed ceiling 5-6 %; ammo horizon (20 shells at 0.7/s ≈ 28 s) ≡ starvation timer 30 s ⟹ emptying the reserve is rational, the rule carries no gradient.

Env state (jolt_enemy_tank.cpp, commits 548b4117 of 2026-10-23 and ca4583a5 of 2026-10-25): 20 initial shells, +5 per hit, cap 30, empty reserve (no shell in flight) = suicide ⟹ death −1; starvation 30 s without hit (+3 s per hit, +15 s per kill) = truncation without penalty. Shell ratio/max is in the proprio.

**Arithmetic (2026-10-27, run 441):** for the reserve to be maintained hit ≥ 1/5 = 20 % is needed. Observed ceiling across all runs: 5-6 %. At 0.7 shot/s the 20 shells leave in ~28 s, i.e. the starvation timer: keeping shells does not save (truncation at 30 s), emptying them maximises the chance of a hit in the window. Fire 0.5-0.9/s constant from 430 to 441: neither the regen removal nor death at empty moved the cadence. The CSV does not distinguish the end cause (empty / starvation / killed) — fire×s ≈ 20-22 shells fired for 24-27 available, so unknown whether the agent keeps shells.

**Why:** Samuel expects regen-on-hit to teach ammo management; it cannot while the break-even threshold is 3-4× above the precision ceiling.
**Rule:** before touching the reward, bring the threshold close to the ceiling (e.g. +15-20 shells per hit ⟹ threshold 5-7 %) or conclude it is useless; do not propose a long cooldown (refused, see docs/decisions.md 2026-10-25). See runs 441, 430, 437. (starvation removed 2026-11-06 per docs/decisions.md)

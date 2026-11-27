# Per-episode efficiency levers: target density and aim tolerance

Runs: 419, 447, 452, 453 (applied)

2026-10-29 — to converge faster at equal number of episodes: target density (419 = 32 tanks in the same 250 zone ⟹ hit ×2.5 at 0 k) and aim tolerance (2° ⟹ tol 4-8 m for miss 27 m: aim reward in the exponential tail, gradient ~2e-4/m); λ 0.99, epochs, critic are not levers.

Signal per episode = ~22 shells (reserve 20, +15/hit at 2 % hit) × informativeness of each shell.
- Density: 419 = 32 tanks in the initial zone 250×250 (same zone as 16 tanks today). Hit at 0 k 6 % vs 2-2.5 %; kill/episode ×8 (with regen). GPU-cost-free equivalent: `--initial_spawn_width/height 175` (area ÷2). CLI flags, zero code.
- Aim tolerance (jolt_enemy_tank.cpp:62-63, aim_tolerance_min 3.25 m, angle 2°): tol = 3.25 + 0.035·d ⟹ 4-8 m for shots of 30-130 m, mean miss 27 m ⟹ exp(−27/6) ≈ 0.01, gradient 0.1·q/tol ≈ 2e-4 per metre: the typical shell learns almost nothing. Angle 6° + min 10 m ⟹ tol ≈ 18 m, q ≈ 0.22, gradient ≈ 1.2e-3/m (×6), max 0.1 unchanged < hit 0.2.
- Non-levers: gae_lambda already 0.99 (~1 s flight intact), epochs > 4 ⟹ KL regime (run 449), critic at ev ≈ 0 legitimate (quasi-constant value).

**Why:** 452 showed the optimiser healthy (skip 0, Hc 0.62) and the game unchanged at 64 k; what is missing is the per-episode signal, not the number of steps (see comparisons/2026-10-29_event-density-collapsed-since-10-21.md).

**Decision 2026-10-29 (Samuel):** all three applied for run 453 — default nb_tanks 32 (main.cpp), tolerance 10 m + 6°, reserve 40 / max 50 (jolt_enemy_tank.cpp), model tests adapted. Comparison reference: 452 (config 447 + empty-reserve truncation, 16 tanks, 2°, reserve 20). Proposal to Samuel at first; he decided.

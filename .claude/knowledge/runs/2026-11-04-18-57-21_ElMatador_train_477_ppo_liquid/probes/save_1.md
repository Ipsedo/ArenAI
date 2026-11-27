# Probe 477 save_1 vs save_0 — 2026-11-04

`build/arenai_agent/arenai_agent_probe` (the cmake-build-release one dated 2026-11-03, old constructor), on train_477_ppo_liquid, 4 stochastic episodes, spawn 250 (D = 250 all run), save_1 (~1 h 25 of training) compared with save_0 (init) with the same protocol. Analysis script in session scratchpad (analyze_477.py: p_fire by angle/distance/reserve/time bins, turret, hit by angle).

- **Trigger**: mean p_fire 0.060 (init 0.100), median 0.032, no decision > 0.5 ⟹ the (deterministic) game would never fire. Flat vs angle: 0.062 under 2°, 0.056 at 45-90°, 0.049 > 90°; aligned < 5° and < 40 m: 0.076 vs 0.060. Spawn reflex 0.19 in the first second.
- **Reserve**: emptied for 52 % of tanks (init 84 %), median 12.8 s (10.4 s), 26 % of tank-steps unarmed (48 %). Effective shots 0.45/s unchanged.
- **Turret**: first run since 470 where it progresses. Best enemy < 5°: 11.7 % of time (init 5.1 %), < 15°: 38 % (25 %). corr(mu_canon_x, yaw) ≈ 0, σ canon still 0.30.
- **Shells**: hit 12.2 % (init 6.2 %), median miss 13.3 m (16.2 m), 34 % of shots under 10° (18 %). The gain comes from the turret, not a selective trigger.

Criterion of the centred shot reward (run 477): reserve lasting longer without extinguished trigger = reached; p_fire depending on angle = not yet.

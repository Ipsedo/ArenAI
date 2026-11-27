# train_472_ppo_liquid

| | |
|---|---|
| machine | ElMatador |
| user | samuel |
| launch | 2026-11-03 14:12:01 |
| commit at launch | `682bc8f3` — potential reward shaping |
| commits ≤ 90 min after | none |
| diff at launch | unknown (identity recorded after the run ended) |
| raw data | `outputs/train_472_ppo_liquid` on ElMatador |

## Established facts


- 2026-11-03: launched 14:12 (config 471 + aim-potential shaping, per-shell aim removed, shot_cost 0.03 at impact; see potential-shaping note). At 155 k steps (56 k/h): fire 2.19 → 0.75 (7 k) → 0.02-0.07/s from 50 k, kill 0 since 7 k, r_hit 7e-5, r_aim −1.5e-4 constant (= tax −(1−γ)Φ·0.1 ⟹ mean Φ ≈ 0.1, flat), s 27 s (starvation). D 250 → 300 at 129 k and hit 5-7 % are artefacts: ~0.04 shot/s, curriculum ratio read on a handful of shells, like 465.
- 2026-11-03: probe save_1 (108 k, 4 episodes side 550, `--features`), at step ≥ 10: trigger p_fire 0.003 flat vs best_angle (0.0030 at 0-2°, 0.0025 at 20-45°), 0 line > 0.5, spawn reflex 0.057 at step < 10; saturated sigmoid (p(1−p) ≈ 0.003) ⟹ will not come back even if aim improves. Turret: |mu_cx| 0.15, corr(yaw, mu_cx) −0.03, enemy in field 13 %, best_angle < 5° 3.7 %, Φ estimated 0.048 — **worse than untrained 470 save_0** (17 %, 6.9 %, 0.070): rotation unrelated to the enemy. Perception: yaw RMSE 24.8° (std 23.5°), AUC aligned < 6° 0.58-0.65 ≡ 470/471 (see non-decodable-bearing perception note).
- 2026-11-03: cause (implementation fault): AimPotential reuses AimScorer(2.5 m, 3°) of the real hit. Measured on the probe: |F| > 0.01 on 5 % of decisions, p90 |F| = 0.0035 (10× under shot_cost), |F| > 0.001 on 18 %; median Φ 0.01. Beyond ~10° slope is zero and the turret is never there: the « exponential tail » of the leverage note reproduced. With a potential the width of Φ is free (policy invariance): tol 20° ⟹ |F| > 0.001 on 56 %, mean Φ 0.40; tol 10° ⟹ 38 %.
- 2026-11-03 (Samuel: « Oui, prépare 473 »): applied for 473: own potential tolerance (wide, ~15-20°, hit keeps 3°), shot_cost 0 while aim does not exist (cost sound in principle — selectivity if P(hit|aligned)·0.3 > c > P(hit|random)·0.3, i.e. 0.006 < c < 0.15 — but inexpressible by a blind policy, so it closes the trigger one-way), mean-Φ metric in CSV only (r_aim measures only the variation). Gate unchanged: yaw decodable at save_1, otherwise privileged auxiliary head.
- 2026-11-03 17:30 (uncommitted): jolt_enemy_tank.cpp `AIM_POTENTIAL_TOLERANCE_MIN 2.5 m` + `AIM_POTENTIAL_TOLERANCE_DEGREES 20` (simple Cauchy; value to be confirmed by Samuel; alternative = mean 20° + 3° to keep a premium on final precision), `shot_cost 0` (mechanism kept), CSV column `phi` (train_environment, living tanks only, flag print_in_progress_bar back on MeanMetric), tests adapted (0.5 thresholds of off-axis / obstacle tests, MissedShellGivesNothing). Model 86/86, agent 2278 ok. Hit no longer has a tolerance: it is a physical contact.

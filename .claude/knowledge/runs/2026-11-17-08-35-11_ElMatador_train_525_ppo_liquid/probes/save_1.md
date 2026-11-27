# Probe 525 save_1: start point

Run 525 (ppo_liquid, launched 2026-11-17 08:35): code of 524 (single advantage, flat CNN, target_kl 0.10) without aim potential, purely event reward; working tree uncommitted on 68c41e20. save_1 = 27 k, D 250. Binary `build/arenai_agent/arenai_agent_probe` of 2026-11-17 08:32, no inference diff. Sides 550 and 1000, 4 episodes, stochastic.

| | tour<5° | tour<20° | ratio | p_al | p_off | hit/sh | shots/s | 1st empty | disarmed | σ canon | n aligned & close |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 525 s1 @550 | 8.5 % | 38.6 % | 1.03 | 0.548 | 0.535 | 4.4 % | 0.91 | 2.9 s | 72 % | 0.38-0.39 | 206 |
| 525 s1 @1000 | 6.7 % | 33.7 % | 1.02 | 0.546 | 0.533 | 1.7 % | 0.83 | 2.8 s | 76 % | 0.38-0.40 | 148 |
| 522 s1 @550 | 17.0 % | 54.0 % | 1.01 | 0.633 | 0.629 | 5.4 % | 1.59 | 2.7 s | 57 % | 0.23 | 388 |
| 523 s2 @550 (54 k) | 13.0 % | 49.4 % | 1.04 | 0.621 | 0.596 | 7.2 % | 1.14 | 3.0 s | 70 % | 0.47 | 581 |
| 524 s5 @550 (135 k) | 15.7 % | 54.8 % | 4.15 | 0.737 | 0.177 | 11.4 % | 1.10 | 6.8 s | 36 % | 0.46 | 493 |

- Trigger: flat, 0.53-0.55 at all angles = init (0.5). Dump in 2.9 s, all tank-episodes empty, 72-76 % of decisions disarmed.
- Aim: nothing learned. tour<5° 8.5/6.7 %, under 522 s1 (17 %, separate credit) and 523 s2 (13 %). Median |μ| canon_x 0.18, σ 0.38: weak commands, turret barely moves.
- CSV at same index: hit 0.065-0.076 and kill 7.6-9.2 at D 250. The CSV rise since start (hit 0.03, kill 2.6) has no counterpart in trigger or turret at this save.
Markers: save_5 (135 k) side 550: tour<5° above 20 % = aim starts; 13-16 % = 524's level (no aim); ratio > 2 = trigger starts. Separate-credit control at save_5: 29 %, ratio 3.2.

# Probe 539 save_3 and save_5: current image vs 538

Run `train_539_ppo_liquid` = 538 config + fix of the one-decision image lag (`offscreen_renderer.cpp`, uncommitted). Training binary `cmake-build-release` of 2026-11-19 16:12. Saves: save_3 (81 k, D 350), save_5 (135 k, D 400). Probe binary `build/arenai_agent/arenai_agent_probe` of 2026-11-19 13:49 (with the fix), no inference diff. Sides 250 and 550, 4 episodes, stochastic. Control: 538 s3 probed in the morning with the lagged image (as in training), 538 s8 (216 k).

| run save | side | ratio | p_al | p_off | hit/sh | h<5° | shots<5° | empty_s | %disarmed | shots/s | σx | \|μ\|cx |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 538 s3 | 250 | 1.13 | 0.75 | 0.67 | 14.3 % | 32 % | 22 % | 4.0 | 35 % | 2.21 | 0.10 | 0.14 |
| 539 s3 | 250 | 1.06 | 0.81 | 0.76 | 13.6 % | 39 % | 23 % | 4.5 | 29 % | 2.42 | 0.09 | 0.10 |
| 539 s5 | 250 | 1.14 | 0.83 | 0.73 | 13.2 % | 38 % | 22 % | 3.2 | 27 % | 2.34 | 0.04 | 0.07 |
| 538 s3 | 550 | 1.23 | 0.75 | 0.61 | 7.9 % | 19 % | 23 % | 2.8 | 51 % | 1.78 | 0.15 | 0.18 |
| 539 s3 | 550 | 1.11 | 0.82 | 0.73 | 7.5 % | 22 % | 23 % | 2.4 | 53 % | 1.78 | 0.11 | 0.11 |
| 539 s5 | 550 | 1.24 | 0.84 | 0.67 | 9.1 % | 25 % | 27 % | 2.8 | 38 % | 2.20 | 0.08 | 0.08 |

## Aim (yaw toward nearest enemy, < 100 m, step ≥ 10, `aim_lin.py`)

| | 538 s3 (250 / 550) | 539 s3 (250 / 550) | 539 s5 (250 / 550) | 538 s8 (250 / 550) |
|---|---|---|---|---|
| dwell < 5° (s) | 0.34 / 0.32 | 0.50 / 0.62 | 0.66 / 0.73 | 0.30 / 0.29 |
| passes ≥ 1 s | 4.0 / 3.8 % | 12.7 / 19.4 % | 21.7 / 24.1 % | 2.0 / 2.6 % |
| number of passes | 668 / 1124 | 393 / 463 | 309 / 474 | 1471 / 2565 |
| yaw < 5° | 6.9 / 7.0 % | 6.9 / 8.0 % | 7.5 / 9.1 % | 6.4 / 7.9 % |
| corrective turret μ, 5-40° | 0.586 / 0.575 | 0.556 / 0.589 | 0.561 / 0.587 | 0.592 / 0.604 |
| mean command (°/decision) | 1.49 / 1.75 | 1.12 / 1.20 | 0.85 / 0.99 | 1.58 / 1.63 |

Kills per probe episode, 539: s3 25-28 (250), 23-25 (550); s5 26-29 (250), 16-22 (550).

## CSV at equal index (identical D milestones)

| index | D | kill 538 / 539 | hit 538 / 539 | fire 538 / 539 |
|---|---|---|---|---|
| 60 k | 300 | 22.4 / 23.3 | 0.115 / 0.121 | 2.40 / 1.96 |
| 105 k | 350 | 23.1 / 25.4 | 0.118 / 0.124 | 1.58 / 2.38 |
| 120 k | 400 | 22.7 / 26.1 | 0.113 / 0.119 | 1.04 / 2.79 |
| 150 k | 450 | 21.6 / 26.4 | 0.125 / 0.115 | 1.01 / 1.71 |

Reading: established: the turret holds the target. Dwell under 5° doubled (0.3 → 0.5-0.7 s), passes of at least 1 s from 4 % to 13-24 %, calmer command. 538 never exceeded 0.34 s from s1 to s8. Unchanged: total aligned time (yaw < 5° 7-9 %), off-axis corrective gesture (0.56-0.59), touches per shell (13-14 % / 7.5-9 %). Passes are twice fewer and twice longer. CSV kills +15 to +20 % from 105 k, by firing cadence and not touch rate. One run per arm: kill and hit/shell gaps are seed-noise size; dwell is outside the whole range seen on 538.
To follow: dwell and hit/shell at D 600 and D 1000; trigger selectivity (ratio 1.06-1.24, no better than 538); kills at D 800 vs 538 (18.9 at 451 k) and vs separate-credit runs (17.8-22 at D 1000).

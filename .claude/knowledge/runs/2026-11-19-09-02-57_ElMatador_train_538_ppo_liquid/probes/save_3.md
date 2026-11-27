# Probe 538 save_1 and save_3: white-noise control of 537

Run `train_538_ppo_liquid` = 537 + `--noise_correlation 0` (sole difference, same training binary of 2026-11-19 07:31). Saves: save_1 (27 k, D 250), save_3 (81 k, D 350). Probe binary `build/arenai_agent/arenai_agent_probe` of 2026-11-19 09:33 (recompiled after adding export fields to `AimGeometry`, no inference diff). Sides 250 and 550, 4 episodes, stochastic. Controls: 537 s1/s3, 534 s3.

| run save | side | ratio | p_al | p_off | hit/sh | h<5° | shots<5° | empty_s | %disarmed | shots/s | σx | \|μ\|cx |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 538 s1 | 250 | 1.05 | 0.60 | 0.57 | 10.9 % | 26 % | 19 % | 3.2 | 43 % | 1.79 | 0.17 | 0.11 |
| 538 s1 | 550 | 1.05 | 0.59 | 0.57 | 5.7 % | 14 % | 20 % | 2.9 | 58 % | 1.48 | 0.18 | 0.13 |
| 538 s3 | 250 | 1.13 | 0.75 | 0.67 | 14.3 % | 32 % | 22 % | 4.0 | 35 % | 2.21 | 0.10 | 0.14 |
| 538 s3 | 550 | 1.23 | 0.75 | 0.61 | 7.9 % | 19 % | 23 % | 2.8 | 51 % | 1.78 | 0.15 | 0.18 |
| 537 s3 | 250 | 1.21 | 0.66 | 0.55 | 12.3 % | 30 % | 14 % | 3.3 | 50 % | 1.41 | 0.16 | 0.19 |
| 537 s3 | 550 | 1.23 | 0.64 | 0.52 | 6.5 % | 18 % | 13 % | 3.0 | 66 % | 1.18 | 0.18 | 0.17 |

## Aim (yaw toward nearest enemy, < 100 m, step ≥ 10)

| | 538 s1 (250 / 550) | 538 s3 (250 / 550) | 537 s3 (250 / 550) | 534 s3 (250 / 550) |
|---|---|---|---|---|
| corrective turret μ, 5-40° | 0.561 / 0.553 | 0.586 / 0.575 | 0.569 / 0.572 | 0.624 / 0.636 |
| corr(command, yaw) | −0.26 / −0.23 | −0.22 / −0.19 | −0.16 / −0.15 | −0.28 / −0.26 |
| yaw < 5° | 4.5 / 4.5 % | 6.9 / 7.0 % | 5.8 / 5.4 % | 7.0 / 6.4 % |
| dwell < 5° (s) | 0.32 / 0.35 | 0.34 / 0.32 | 0.28 / 0.33 | 0.29 / 0.36 |
| mean command (°/decision) | 1.13 / 1.24 | 1.49 / 1.75 | 1.94 / 1.83 | 1.54 / 1.31 |

## CSV at equal index

| index | run | D | kl | clip | skip | kill | hit | fire |
|---|---|---|---|---|---|---|---|---|
| 27 k | 537 / 538 | 250 | 0.039 / 0.008 | 0.31 / 0.11 | 0 / 0 | 6.9 / 13.7 | 0.070 / 0.094 | 1.32 / 1.18 |
| 66 k | 537 / 538 | 300 | 0.069 / 0.011 | 0.36 / 0.16 | 0.05 / 0 | 19.7 / 23.2 | 0.115 / 0.117 | 1.11 / 1.66 |
| 80 k | 537 / 538 | 350 | 0.066 / 0.014 | 0.38 / 0.16 | 0.08 / 0 | 19.4 / 22.8 | 0.111 / 0.109 | 1.35 / 1.78 |

Reading: at equal config white noise does at least as well as coloured noise on each aim measure (yaw < 5° 6.9-7.0 % vs 5.4-5.8 %, share of aligned shots 22-23 % vs 13-14 %, hit/shell 14.3/7.9 % vs 12.3/6.5 %), with kl five times lower and null skip. Trigger selectivity at s3 (ratio 1.13-1.23) is there without coloured noise too: it comes from batch 2400 or the linear command, not the noise. Gaps of seed-noise size on a single run per arm; the direction is the same everywhere.

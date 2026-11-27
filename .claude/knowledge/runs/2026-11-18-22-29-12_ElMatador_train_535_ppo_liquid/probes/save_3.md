# Probe 535 save_3: signal present, weaker than at lr 1e-4

Run `train_535_ppo_liquid` = 534 + `--actor_learning_rate 5e-5` (sole change), commit b83da585. save_3, index 81 k, D 350. Binary `build/arenai_agent/arenai_agent_probe` of 2026-11-18 21:24:57 (run code, `--force` for the dating artefact, see 534 s3). Sides 250 and 550, 4 episodes, stochastic. Controls: 534 s3, 533 s3, 529/532 s3.

| side | ratio | p_al | p_off | hit/sh | h<5° | shots<5° | empty_s | %disarmed | shots/s | tour<5° | σx | \|μ\|cx |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 250 | 1.01 | 0.61 | 0.60 | 11.2 % | 28 % | 17 % | 3.5 | 36 % | 2.01 | 14.6 % | 0.18 | 0.28 |
| 550 | 1.01 | 0.61 | 0.60 | 6.7 % | 17 % | 18 % | 2.7 | 51 % | 1.78 | 14.1 % | 0.19 | 0.25 |

## Aim (yaw toward nearest enemy, < 100 m, step ≥ 10)

| | 535 s3 (250 / 550) | 534 s3 (250 / 550) | 533 s3 (250 / 550) | 529 / 532 s3 (250) |
|---|---|---|---|---|
| corrective turret μ, 5-40° | 0.61 / 0.61 | 0.62 / 0.64 | 0.63 / 0.65 | 0.55 / 0.55 |
| gain (° command per ° yaw) | −0.018 / −0.018 | −0.031 / −0.027 | −0.024 / −0.025 | −0.005 / −0.004 |
| corr(command, yaw) | −0.21 / −0.22 | −0.28 / −0.26 | −0.25 / −0.26 | −0.11 / −0.12 |
| yaw < 5° | 5.9 / 5.7 % | 7.0 / 6.4 % | 6.9 / 6.4 % | 4.7 / 4.3 % |
| yaw < 20° | 20.1 / 18.3 % | 22.2 / 19.6 % | 21.4 / 19.1 % | 16.2 / 15.0 % |
| dwell < 5° (s) | 0.36 / 0.37 | 0.29 / 0.36 | 0.31 / 0.31 | 0.50 / 0.61 |
| mean command (°/decision) | 1.19 / 1.01 | 1.54 / 1.31 | 1.39 / 1.23 | 0.67 / 0.47 |

- The coloured-noise signal is there for the third time (corrective 0.61 vs 0.55 in white noise, gain ×4), but behind 534/533 on all measures: gain ×0.6, yaw < 5° −1 point. Consistent with a twice-slower actor; gap of 1.5-2.5 points for s.e. 0.5, one seed.
- Trigger flat 0.60, dump 2.7-3.5 s; hit/shell 11.2/6.7 % (534: 11.9/6.5). CSV at equal index ≡ 534: kill 20.8 / hit 0.100 at 80 k (534: 20.5 / 0.107).

## Skip

| index | 60 k | 80 k | 88 k | 97 k |
|---|---|---|---|---|
| 534 | 0.51 | 0.58 | 0.64 | – |
| 535 | 0.29 | 0.37 | 0.43 | 0.48 |

535's slope since 80 k: ~0.65 pt / k, same as 534 at same age. lr ÷2 shifts the curve by about 40 k without bending it; 0.75 expected near 140 k if the slope holds.
Conclusion: established: coloured noise reproduces its effect at lr 5e-5; lr does not change the skip dynamics. In the noise or not settled: that the deficit vs 534 is due to lr (one seed). Next: s5 (135 k) and s9; judge holding vs 528 s9 and 529 s9.

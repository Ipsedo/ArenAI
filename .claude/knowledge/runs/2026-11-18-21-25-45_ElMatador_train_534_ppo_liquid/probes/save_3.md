# Probe 534 save_3: 533's sign holds without the short head

Run `train_534_ppo_liquid` (coloured AR(1) noise ρ 0.9 alone), effective commit b83da585. save_3, index 81 k, D 350. Binary `build/arenai_agent/arenai_agent_probe` compiled 2026-11-18 21:24:57, no source modified since, clean tree ⟹ run code. The diff flagged by the script comes from dating (commit made at 22:01), probe run with `--force`. Sides 250 and 550, 4 episodes, stochastic. Control: 533 s3, same sides.

| side | ratio | p_al | p_off | hit/sh | h<5° | shots<5° | empty_s | %disarmed | shots/s | tour<5° | σx | \|μ\|cx |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 250 | 1.02 | 0.61 | 0.60 | 11.9 % | 28 % | 16 % | 3.5 | 50 % | 1.60 | 13.7 % | 0.19 | 0.35 |
| 550 | 1.02 | 0.61 | 0.60 | 6.5 % | 17 % | 16 % | 2.8 | 58 % | 1.54 | 15.0 % | 0.19 | 0.30 |

## Aim (yaw toward nearest enemy, < 100 m, step ≥ 10)

| | 534 s3 (250) | 534 s3 (550) | 533 s3 (250) | 533 s3 (550) | 529 / 532 s3 (250) |
|---|---|---|---|---|---|
| corrective turret μ, 5-40° | 0.62 | 0.64 | 0.63 | 0.65 | 0.55 / 0.55 |
| gain (° command per ° yaw) | −0.031 | −0.027 | −0.024 | −0.025 | −0.005 / −0.004 |
| corr(command, yaw) | −0.28 | −0.26 | −0.25 | −0.26 | −0.11 / −0.12 |
| yaw < 5° | 7.0 % | 6.4 % | 6.9 % | 6.4 % | 4.7 / 4.3 % |
| yaw < 20° | 22.2 % | 19.6 % | 21.4 % | 19.1 % | 16.2 / 15.0 % |
| dwell < 5° (s) | 0.29 | 0.36 | 0.31 | 0.31 | 0.50 / 0.61 |
| corrective beyond 40° | 0.51 | 0.51 | 0.51 | 0.50 | – |
| mean command (°/decision) | 1.54 | 1.31 | 1.39 | 1.23 | 0.67 / 0.47 |

- 534 s3 ≡ 533 s3 on all aim measures, both sides (gaps ≤ 1 point, s.e. ≈ 0.5 point). The short value head was not the lever: coloured noise alone reproduces the sign (corrective 0.62-0.64 vs 0.55 in white noise, gain ×6).
- Same limits as 533: nothing beyond 40° gap, short dwell (command crosses the target), weak gain in absolute.
- Trigger flat 0.60 at any angle (533: 0.67), dump 2.8-3.5 s, disarmed 50-58 %. Shot: hit/shell 11.9/6.5 % (533: 13.7/7.1).
CSV around the save: 88 k (D 350): kill 20.3, hit 0.109, fire 1.77/s, kl 0.071. **skip: 0.54 at 67 k, 0.57 at 78 k, 0.61 at 83 k, 0.64 at 88 k**: slope doubled (1 pt / 2 k); at this rate the 0.75 threshold falls near 110 k.
Conclusion: established (two runs, same result): coloured noise gives at 81 k a turret that answers yaw better than 528/529/532; the short head does not contribute. Not established that it holds beyond s3; skip accelerates. Useful next: probe s5 (135 k) and s9 vs 528 s9 (dwell 0.26 s) and 529 s9 (0.53 s).

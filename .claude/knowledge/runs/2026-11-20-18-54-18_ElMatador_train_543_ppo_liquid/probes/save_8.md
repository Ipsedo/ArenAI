# Probe 543 save_8

Run 543 (`ppo_liquid`, 541 + constant empty penalty 0.5), save_8, index 216 k, D 550. Binary `build/arenai_agent/arenai_agent_probe` compiled 2026-11-20 18:39 on `b83da585` + uncommitted tree (incl. `train_environment.cpp` at 0.5). No inference diff with the run's code. Sides 550 and 1000, 4 episodes, 32 tanks, stochastic, economy 10 / 5 / 30.

| | 543 s8 c550 | 541 s9 c550 | 542 s9 c550 | 542 s7 c550 | 543 s8 c1000 |
|---|---|---|---|---|---|
| index / D | 216 k / 550 | 243 k / 600 | 243 k / 600 | 189 k / 500 | 216 k / 550 |
| ratio (aligned close / off-axis) | 1.99 | 2.28 | 1.90 | 1.27 | 2.27 |
| p_fire aligned close | 0.80 | 0.80 | 0.72 | 0.84 | 0.78 |
| p_fire off-axis | 0.40 | 0.35 | 0.38 | 0.66 | 0.35 |
| n aligned & close | 374 | 469 | 350 | 280 | 165 |
| hit/shell | 0.114 | 0.158 | 0.109 | 0.085 | 0.066 |
| hit/shell fired < 5° | 0.353 | — | — | — | 0.251 |
| shots/tank/s | 1.10 | 1.71 | 1.07 | 1.17 | 0.84 |
| 1st empty reserve (median) | 3.6 s | 4.7 s | 3.2 s | 2.4 s | 3.3 s |
| tank-episodes that empty | 82 % | 73 % | 94 % | 96 % | 95 % |
| disarmed steps | 49 % | 29 % | 56 % | 63 % | 65 % |
| turret < 5° | 12.9 % | 23.1 % | 12.4 % | 14.7 % | 11.9 % |
| median angle to target | 31° | 19° | 34° | 26° | 29° |
| σ canon x / y (median) | 0.077 / 0.090 | 0.083 / 0.104 | 0.125 / 0.133 | 0.079 / 0.097 | 0.094 / 0.103 |
| p_zoom | 0.57 | 0.36 | 0.53 | 0.49 | 0.55 |

- Trigger: selective at side 550: p_fire 0.71 at < 2°, 0.64 at 2-5°, 0.40 beyond 20°, 0.80 aligned and close. Deterministic it fires on 82 % of aligned decisions and 25 % of off-axis decisions. Same shape at side 1000 (ratio 2.27 on 165 aligned close decisions, noisy). Level comparable to 541 s9 and 542 s9.
- Dump: intact: 1st empty reserve at 3.6 s median (p10 2.3 s), 82 % of tank-episodes empty, half the steps disarmed. At side 1000 worse: 95 % empty, 65 % disarmed steps. 43 % of shells leave at more than 20° from any target, with 1.2 % touches.
- Shot: hit/shell 11.4 % at side 550, carried by aligned shots (35 % touches at < 5°, a quarter of shots). Aim: turret within 5° of a target 12.9 % of time, median angle 31°. Heads: σ canon 0.08-0.09, never at ceiling, no μ saturation.
Comparison: at D 550 the constant penalty (0.5) is almost 541's (0.55): no reward gap expected at this point; the difference between the two runs holds only to the D 250-450 history where 543 paid more. 543 s8 ≡ 542 s9 on all lines (dump, aim, hit/shell, cadence). 543 s8 under 541 s9: turret 12.9 vs 23.1 %, hit/shell 11.4 vs 15.8 %, disarmed steps 49 vs 29 %. One save of gap (27 k), one point per run.
Conclusions: established: the penalty doubled from D 250 to D 450 did not resorb the dump; trigger selective and not extinct. In the noise: the gap with 541 s9, same order as the 541 / 542 gap at same index. Not settled: trigger holding at D 750-950, the only zone where 543 pays less than 541. Useful next: side-550 probe of saves at D 800-950 (about save_13 to save_16 if milestones follow 541's), compared with 541's saves 12-16 already probed.

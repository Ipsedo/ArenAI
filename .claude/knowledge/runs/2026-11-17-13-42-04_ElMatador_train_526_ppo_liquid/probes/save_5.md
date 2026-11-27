# Probe 526 save_5: both markers missed

Run 526 (ppo_liquid, launched 2026-11-17 13:42): 525 + `actor_batch_size` 2400. save_5 = 135 k, D 400. Binary of 2026-11-17 08:32 (525's code, uncommitted on 68c41e20), no inference diff. Sides 550 and 1000, 4 episodes, stochastic.

| | tour<5° | tour<20° | ratio | p_al | p_off | hit/sh | h<5° | shots<5° | shots/s | touches/tank-s | 1st empty | disarmed | σ canon x/y | \|μ\| cx | n aligned & close |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **526 s5 @550** | 14.4 % | 44.3 % | 1.67 | 0.63 | 0.375 | 7.5 % | 17 % | 21 % | 1.19 | 0.089 | 4.3 s | 54 % | 0.43 / 0.39 | 0.25 | 507 |
| **526 s5 @1000** | 12.4 % | 40.7 % | 1.91 | 0.68 | 0.355 | 3.1 % | 9 % | 18 % | 1.08 | 0.034 | 3.7 s | 56 % | 0.45 / 0.40 | 0.24 | 289 |
| 526 s3 @550 (81 k) | 15.4 % | 50.1 % | 1.19 | 0.71 | 0.592 | 6.8 % | 20 % | 20 % | 1.72 | 0.117 | 3.1 s | 51 % | 0.23 / 0.27 | 0.24 | 606 |
| 526 s3 @1000 | 16.4 % | 51.4 % | 1.17 | 0.68 | 0.580 | 2.4 % | 7 % | 19 % | 1.54 | 0.037 | 2.7 s | 59 % | 0.26 / 0.28 | 0.24 | 188 |
| 525 s5 @550 | 11.8 % | 51.3 % | 3.6 | 0.66 | 0.186 | 9.7 % | 21 % | 17 % | 0.90 | 0.087 | 6.1 s | 49 % | 0.60 | 0.29 | 568 |
| 525 s5 @1000 | 10.6 % | 50.1 % | 3.8 | 0.64 | 0.170 | 4.1 % | 11 % | 14 % | 0.75 | 0.031 | 6.1 s | 51 % | 0.61 | 0.27 | 195 |
| 524 s5 @550 | 15.7 % | 54.8 % | 4.2 | 0.74 | 0.177 | 11.4 % | 24 % | 24 % | 1.10 | 0.125 | 6.8 s | 36 % | 0.46 | 0.47 | 493 |
| control s5 @550 | 29.3 % | — | 3.2 | 0.27 | — | 20.0 % | — | 45 % | — | 0.171 | — | 16 % | 0.34-0.42 | 0.18 | — |

CSV at same index (121-139 k, D 400): hit 0.111, kill 20.3, fire 1.33, kl 0.011, skip 0 (525: 0.125 / 15.7 / 0.83, skip 0.28).

- Trigger: late. Descended en bloc (off-axis 0.59 → 0.375, mean level 0.60 → 0.40) but still twice as high off-axis as 525 and 524 at same index (0.18). Ratio 1.67/1.91, under marker 2; angle-only 1.16/1.11. Deterministic almost mute (25 %/16 % of aligned & close states, 12 %/7 % off-axis).
- Dump: barely resorbed. Empty at 4.3/3.7 s (s3: 3.1/2.7; 525 s5: 6.1), 95-99 % of tank-episodes empty, 54-56 % of decisions disarmed.
- Aim: not started, slight regression. tour<5° 14.4/12.4 % vs 15.4/16.4 % at save_3; tour<20° 50 → 44/41 %, median angle 20° → 25-28°. Still under the 20 % marker. No rotation bias (μ canon_x +0.07/+0.08), no saturation.
- σ canon: back up. 0.23-0.28 → 0.39-0.45 (CSV Hc back to 0), under 525's 0.60.
- Shot: 39-42 % of shells still leave at ≥ 20°. Touches per tank-second 0.089/0.034 = 525 s5's (0.087/0.031), under 524 s5 (0.125/0.058).
Established: at 135 k batch 2400 unblocked nothing on aim (tour<5° within 524/525 range) and slows trigger selectivity (p_off 0.375 vs 0.18 for 524 and 525). Useful touch throughput is 525's; extra CSV kills come from extra shells fired. Not settled: tour<5° drop between s3 and s5 (1 to 4 points) is seed-noise order; trigger delay may be a matter of gradient step count (one step per epoch instead of five): slope between s3 and s5 goes the right way.
Next: save_9 (243 k, D ≈ 600) side 550: p_off < 0.1 and empty > 6 s = trigger caught up with 525; tour<5° vs 525 s9 (6.6 %), 524 s9 (12.4 %) and control (34 %), with dwell < 5°. CSV: kill and fire at D 600-800, where 525 fell.

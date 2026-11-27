# Probe 528 save_5: same start as 524, half as disarmed

Run 528 (ppo_liquid, kills/tank curriculum + `shells_per_hit` 5, tree uncommitted on 68c41e20). save_5 = 135 k, D 400. Binary `build/arenai_agent/arenai_agent_probe` of 2026-11-17 08:32, no inference diff; sources modified since (curriculum, defaults) do not touch the probe. Sides 550 and 1000, 4 episodes, stochastic, run economy (10 / 5 / 30).

| | tour<5° | tour<20° | ratio | p_al | p_off | hit/sh | h<5° | shots<5° | shots/s | touches/tank-s | 1st empty | disarmed | σ canon | \|μ\| cx | n aligned & close |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **528 s5 @550** | 11.4 % | 50.7 % | 4.7 | 0.58 | 0.124 | 11.7 % | 28 % | 20 % | 1.01 | 0.121 | 12.7 s | **24 %** | 0.51-0.59 | 0.52 | 683 |
| **528 s5 @1000** | 10.9 % | 47.0 % | 7.2 | 0.64 | 0.090 | 6.8 % | 21 % | 19 % | 0.62 | 0.043 | 14.1 s | **34 %** | 0.56-0.63 | 0.59 | 345 |
| 525 s5 @550 | 11.8 % | 51.3 % | 3.6 | 0.66 | 0.186 | 9.7 % | 21 % | 17 % | 0.90 | 0.089 | 6.1 s | 49 % | 0.60-0.61 | 0.29 | 568 |
| 525 s5 @1000 | 10.6 % | 50.1 % | 3.8 | 0.64 | 0.170 | 4.1 % | 11 % | 14 % | 0.75 | 0.031 | 6.1 s | 51 % | 0.61-0.62 | 0.27 | 195 |
| 524 s5 @550 | 15.7 % | 54.8 % | 4.2 | 0.74 | 0.177 | 11.4 % | 24 % | 24 % | 1.10 | 0.131 | 6.8 s | 36 % | 0.46 | 0.47 | 493 |
| 524 s5 @1000 | 13.5 % | — | 6.0 | 0.77 | 0.129 | 7.4 % | 20 % | 20 % | 0.78 | 0.059 | — | 39 % | 0.44-0.53 | 0.54 | — |

CSV at same index: hit 0.14, kill ~23, fire 0.65-1.3 (525: 0.125 / 15.7; 524: 0.139 / 21.5).

## By distance band (best-aligned target)

Touches per tank-second:

| | total | < 50 m | 50-100 m | 100-200 m | > 200 m |
|---|---|---|---|---|---|
| **528 s5 @550** | 0.121 | 0.081 | **0.016** | 0.008 | 0.015 |
| 525 s5 @550 | 0.089 | 0.059 | 0.007 | 0.008 | 0.016 |
| 524 s5 @550 | 0.131 | 0.080 | 0.019 | 0.014 | 0.019 |
| 522 s5 @550 (aims) | 0.122 | 0.073 | 0.023 | 0.011 | 0.014 |
| **528 s5 @1000** | 0.043 | 0.025 | **0.009** | 0.002 | 0.007 |
| 525 s5 @1000 | 0.031 | 0.018 | 0.003 | 0.002 | 0.008 |
| 524 s5 @1000 | 0.059 | 0.038 | 0.008 | 0.002 | 0.011 |

hit/shell all shots — aligned shots < 5°:

| | < 50 m | 50-100 m | 100-200 m | > 200 m |
|---|---|---|---|---|
| **528 s5 @550** | 47 — 77 % | 10 — **31 %** | 3 — 11 % | 4 — 8 % |
| 525 s5 @550 | 42 — 72 % | 6 — 22 % | 3 — 7 % | 4 — 6 % |
| 524 s5 @550 | 50 — 72 % | 11 — 27 % | 5 — 13 % | 4 — 8 % |
| **528 s5 @1000** | 41 — 75 % | 14 — **40 %** | 2 — 8 % | 2 — 5 % |
| 525 s5 @1000 | 37 — 64 % | 7 — 29 % | 2 — 7 % | 1 — 2 % |
| 524 s5 @1000 | 54 — 79 % | 12 — 34 % | 2 — 9 % | 2 — 5 % |

Armed p_fire on a target aligned < 5° (count):

| | off-axis | < 50 m | 50-100 m | 100-200 m | > 200 m |
|---|---|---|---|---|---|
| **528 s5 @550** | 0.124 | 0.55 (984) | **0.38 (1 321)** | 0.23 | 0.20 |
| 525 s5 @550 | 0.186 | 0.64 (736) | 0.38 (829) | 0.28 | 0.26 |
| 524 s5 @550 | 0.177 | 0.71 (762) | 0.50 (941) | 0.30 | 0.27 |
| **528 s5 @1000** | 0.090 | 0.59 (551) | **0.36 (871)** | 0.17 | 0.12 |
| 525 s5 @1000 | 0.170 | 0.59 (294) | 0.38 (375) | 0.23 | 0.20 |
| 524 s5 @1000 | 0.129 | 0.74 (495) | 0.45 (568) | 0.20 | 0.16 |

## Reading

- Dump: clearly lower. Disarmed 24/34 % vs 49/51 % (525) and 36/39 % (524); 1st empty at 12.7-14.1 s vs 6-7 s. 77-93 % of tank-episodes still empty at some point. Part of the gap is mechanical (5 shells refunded per touch), part from a lower off-axis trigger (0.124/0.090 vs 0.17-0.19).
- Trigger at 50-100 m: same level as 525 (0.36-0.38), not more open. 50-100 m / < 50 m ratio 0.68 side 550 (525: 0.59; 524: 0.71). 525's closing happened between s5 and s9 (0.38 → 0.14): this save predates the lock, does not say whether it will close.
- Aligned states at 50-100 m: +40 to +60 % (1 321 vs 829-941 side 550, 871 vs 375-568 side 1000), mostly because tanks are armed longer (armed 58 % vs 38 %).
- Precision at 50-100 m: best of the three (31 % and 40 % of aligned shots); gap to 524 (27/34 %) is noise size; 522 was at 38 % at s5.
- Aim: not started. tour<5° 11.4/10.9 %, identical to 525 (11.8/10.6), under 524 (15.7). σ canon 0.5-0.6 no ceiling, |μ| canon_x 0.52-0.59 (as 524), μ not saturated. Deterministic: 21 % of aligned & close states fire side 550, 9 % side 1000.
Established: at 135 k 528 is at 524's level on touch throughput and precision, at 525's on aim, and half as disarmed as both. The 5-shell economy did not bring the dump back. Noise: touches/tank-s at 50-100 m (0.016 vs 0.019 for 524), aligned hit, ratio. Not settled: the run's hypothesis: trigger at range and aim only separated from 522 between s5 and s9 in 524/525.
Markers for save_9 (243 k): p_fire aligned 50-100 m side 550 ≥ 0.25 = trigger held open at range; ≤ 0.15 = 525's (0.14) and 524's (0.07) trajectory; touches/tank-s at 50-100 m rising toward 0.025+ (522 s9: 0.027) or falling toward 0.005 (525 s9); aligned hit 50-100 m > 35 % and rising = aim at range learned; tour<5° side 550 > 15 %, disarmed continuing to fall.

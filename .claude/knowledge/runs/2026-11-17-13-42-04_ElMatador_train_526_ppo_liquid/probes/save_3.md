# Probe 526 save_3

Run 526 (ppo_liquid, launched 2026-11-17 13:42): 525 + `actor_batch_size` 2400. save_3 = 81 k, D 350. Binary `build/arenai_agent/arenai_agent_probe` of 2026-11-17 08:32 (525's code, uncommitted on 68c41e20), no inference diff. Sides 550 and 1000, 4 episodes, stochastic.

| | tour<5° | tour<20° | ratio | p_al | p_off | hit/sh | h<5° | shots/s | 1st empty | disarmed | σ canon x/y | n aligned & close |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **526 s3 @550** (81 k) | 15.4 % | 50.1 % | 1.19 | 0.706 | 0.592 | 6.8 % | 20.1 % | 1.72 | 3.1 s | 51 % | 0.23 / 0.27 | 606 |
| **526 s3 @1000** | 16.4 % | 51.4 % | 1.17 | 0.676 | 0.580 | 2.4 % | 6.6 % | 1.54 | 2.7 s | 59 % | 0.26 / 0.28 | 188 |
| 525 s1 @550 (27 k) | 8.5 % | 38.6 % | 1.03 | 0.548 | 0.535 | 4.4 % | | 0.91 | 2.9 s | 72 % | 0.38 | 206 |
| 525 s5 @550 (135 k) | 11.8 % | | 3.6 | | 0.19 | 9.7 % | | | 6.1 s | 49 % | | |
| 524 s5 @550 (135 k) | 15.7 % | 54.8 % | 4.15 | 0.737 | 0.177 | 11.4 % | | 1.10 | 6.8 s | 36 % | 0.46 | 493 |
| 522 s1 @550 (27 k) | 17.0 % | 54.0 % | 1.01 | 0.633 | 0.629 | 5.4 % | | 1.59 | 2.7 s | 57 % | 0.23 | 388 |
| control s5 @550 (135 k) | 29 % | | 3.2 | | | | | | | | | |

No probe of the other runs at the same index (81 k): references only bracket.

- Trigger: almost flat, 0.58-0.63 at all angles (init 0.5), ratio 1.19/1.17, gap 0.10-0.11 carried by the aligned-and-close cell alone. It rose uniformly, hence CSV fire ×1.9. Deterministic it fires 81-92 % of the time, aligned or not.
- Dump: confirmed. Empty in 3.1/2.7 s, 96-99 % of tank-episodes empty, 51-59 % of decisions disarmed. The CSV firing volume is a reserve emptying.
- Shot: hit/shell 6.8/2.4 %. 39 % of shells leave at ≥ 20° (hit 0.2-0.4 %), 19-20 % at < 5° (hit 20 % side 550, 6.6 % side 1000).
- Aim: tour<5° 15.4/16.4 %, above 525 s1 and s5 (8.5 then 11.8 %), at 524 s5 and 522 s1's level, far from control s5 (29 %). No rotation bias (μ canon_x mean +0.05; 525 s9: +0.33), no saturation.
- σ canon: 0.23-0.28, tightened from 0.30; 525 went toward 0.38 then 0.60. This is the CSV's negative Hc.
Established: CSV kill gain comes from a trigger raised en bloc and a 3-s dump, not a selective trigger. Turret does not sweep and σ canon stays low, opposite of 525. Noise/not settled: tour<5° 15-16 % does not distinguish 526 from 524; at 81 k 524 and 525 had no selective trigger either (it appeared between 27 k and 135 k).
Next: save_5 (135 k) side 550: ratio > 2 = trigger starts as 524/525; tour<5° > 20 % = aim exceeds 524. Add dwell < 5° (525: 0.2 s, control: 1.0 s).

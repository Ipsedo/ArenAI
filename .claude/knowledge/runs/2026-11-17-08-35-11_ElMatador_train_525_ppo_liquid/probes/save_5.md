# Probe 525 save_5: trigger starts, aim does not

Run 525 (ppo_liquid, pure events, single advantage, flat CNN, target_kl 0.10; tree uncommitted on 68c41e20). save_5 = 135 k, D 400. Binary of 2026-11-17 08:32, no inference diff. Sides 550 and 1000, 4 episodes, stochastic.

| | tour<5° | tour<20° | ratio | p_al | p_off | hit/sh | h<5° | shots<5° | shots/s | touches/tank-s | 1st empty | disarmed | σ canon | \|μ\| cx | n aligned & close |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 525 s5 @550 | 11.8 % | 51.3 % | 3.6 | 0.66 | 0.186 | 9.7 % | 21 % | 17 % | 0.90 | 0.087 | 6.1 s | 49 % | 0.60-0.61 | 0.29 | 568 |
| 525 s5 @1000 | 10.6 % | 50.1 % | 3.8 | 0.64 | 0.170 | 4.1 % | 11 % | 14 % | 0.75 | 0.031 | 6.1 s | 51 % | 0.61-0.62 | 0.27 | 195 |
| 525 s1 @550 | 8.5 % | 38.6 % | 1.03 | 0.55 | 0.535 | 4.4 % | — | — | 0.91 | 0.040 | 2.9 s | 72 % | 0.38-0.39 | 0.18 | 206 |
| 524 s5 @550 | 15.7 % | 54.8 % | 4.2 | 0.74 | 0.177 | 11.4 % | 24 % | 24 % | 1.10 | 0.125 | 6.8 s | 36 % | 0.46 | 0.47 | 493 |
| 524 s5 @1000 | 13.5 % | — | 6.0 | 0.77 | 0.129 | 7.4 % | 20 % | 20 % | 0.78 | 0.058 | — | 39 % | 0.44-0.53 | 0.54 | — |
| control 430 s5 @550 | 29.3 % | — | 3.2 | 0.27 | — | 20.0 % | — | 45 % | — | 0.171 | — | 16 % | 0.34-0.42 | 0.18 | — |

CSV at same index: hit 0.125, kill 15.7, fire 0.83, skip 0.28 (524: 0.139 / 21.5 / 1.12 / 0.34).

- Trigger: started. Off-axis 0.535 → 0.186, aligned & close 0.55 → 0.66, ratio 1.03 → 3.6 (524 s5: 4.2; control: 3.2). Selectivity from distance: angle-only ratio 1.7/1.3. Empty at 6.1 s vs 2.9 s, disarmed 72 → 49 %. Deterministic: 20 % of aligned & close states fire side 550, 8 % side 1000.
- Aim: not started. tour<5° 11.8/10.6 %, under the 20 % marker and under 524 s5 (15.7/13.5). Rise since save_1 (8.5/6.7) mostly on tour<20° (39 → 51 %), at 524 s5's level (55 %). σ canon rises to 0.60-0.62 (save_1: 0.38), no ceiling.
- Shot: 17 % of shots aligned (524: 24 %, control: 45 %), aligned-shot hit 21 % (524: 24 %). hit/shell 9.7/4.1 % vs 11.4/7.4 % for 524.
- CSV hit plateau (0.12 since 60 k) consistent with probe. Gain since save_1 comes from the trigger (fewer off-axis shots), not the turret; 524 had the same plateau at 0.14.
Established: at 135 k, 525 ≈ 524 with slight delay on all axes (aim, precision, dump). Removing Φ neither unblocked nor broke anything visible at this stage. Not settled: the 525/524 gap on tour<5° (4 points) is seed-noise order; one save does not give slope (524 started at 15.7 % and fell; 525 starts at 8.5 % and rises to 11.8 %); the target chosen by the turret was not examined.
Markers for save_9 (243 k, D ~600): tour<5° side 550 > 15 % and rising = aim learned by events; ≤ 12 % = 524's trajectory (12.4 % at s9); trigger: ratio > 10 expected (524 s9: 25; control: 10); disarmed < 5 %.

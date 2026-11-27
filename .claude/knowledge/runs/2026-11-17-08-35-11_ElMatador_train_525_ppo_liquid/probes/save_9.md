# Probe 525 save_9: very selective trigger, sweeping turret

Run 525 (pure events, single advantage, flat CNN, target_kl 0.10; tree uncommitted on 68c41e20). save_9 = 243 k, D 600. Binary of 2026-11-17 08:32, no inference diff. Sides 550 and 1000, 4 episodes, stochastic.

| | tour<5° | tour<20° | ratio | p_al | p_off | hit/sh | h<5° | shots<5° | shots/s | touches/tank-s | 1st empty | disarmed | σ canon | \|μ\| cx | mean μ cx | n aligned & close |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 525 s9 @550 | 6.6 % | 37.5 % | 42 | 0.43 | 0.010 | 31.3 % | 50 % | 23 % | 0.24 | 0.076 | 43 s | 3 % | 0.58-0.59 | 0.49 | +0.33 | 731 |
| 525 s9 @1000 | 4.1 % | 27.7 % | 92 | 0.47 | 0.005 | 26.9 % | 48 % | 21 % | 0.11 | 0.030 | 77 s | 2.5 % | 0.59-0.60 | 0.50 | +0.41 | 322 |
| 525 s5 @550 | 11.8 % | 51.3 % | 3.6 | 0.66 | 0.186 | 9.7 % | 21 % | 17 % | 0.90 | 0.087 | 6.1 s | 49 % | 0.60-0.61 | 0.29 | −0.01 | 568 |
| 524 s9 @550 | 12.4 % | — | 25 | 0.31 | 0.012 | 32.4 % | 59 % | 28 % | 0.21 | 0.068 | — | 2 % | 0.59-0.61 | 0.53 | — | — |
| 524 s9 @1000 | 14.8 % | — | 54 | 0.24 | 0.004 | 22.3 % | 43 % | 22 % | 0.07 | 0.016 | — | 1 % | — | — | — | — |
| control 430 s9 @550 | 33.7 % | — | 10 | 0.14 | — | 34.9 % | — | 61 % | — | 0.101 | — | 1 % | 0.36-0.50 | 0.20-0.22 | — | — |
| control 430 s9 @1000 | 22.2 % | — | 14 | 0.15 | — | 27.8 % | — | 50 % | — | 0.046 | — | 1 % | — | — | — | — |

CSV at same index: hit 0.218, kill 16.5, fire 0.28, skip 0.31 (524: 0.245 / 18.1 / 0.33 / 0.22).

- Trigger: very selective, dump settled. Off-axis 0.186 → 0.010/0.005, aligned & close 0.43/0.47, ratio 42/92 (524 s9: 25/54; control: 10/14). Angle-only: 9.3/11.8 (s5: 1.7/1.3): angle now counts, not only distance. Empty at 43/77 s, disarmed 3 %. Deterministic 8/5 % of aligned & close states fire.
- Aim: regresses. tour<5° 11.8 → 6.6 % (550) and 10.6 → 4.1 % (1000); tour<20° 51 → 38 % and 50 → 28 %. Under 524 s9 (12.4/14.8) and far from control (34/22).
- Sweeping turret: mean μ canon_x +0.33/+0.41 (s5: −0.01) with median |μ| 0.49: yaw command biased to one side, turret rotates rather than settling on a target. dir_y mean −0.83/−0.91, saturated 22-23 % of time.
- Shot: precise when it leaves: aligned-shot hit 50/48 %, hit/shell 31/27 % (524 s9: 32/22; control: 35/28). But only 21-23 % of shots aligned (control 50-61 %) and 53/48 % leave at 5-20° with hit/shell 34/29 %. Touch throughput 0.076/0.030 per tank-s, between 524 (0.068/0.016) and control (0.101/0.046).
Established: CSV hit rise (0.12 → 0.22) comes from the trigger, not the turret; turret does not settle on targets: marker "tour<5° > 15 %" missed, s5 → s9 trend downward on both sides; touch throughput above 524 at same save (×1.1 side 550, ×1.9 side 1000) despite lower measured aim.
Not settled: whether sweep + fire in passing is a stable mode or transition (at D 1000 on-axis passes at range become rare: where 524 broke off, kill 18 → 9); tour<5° measures holding on target, not firing in passing: underestimates a sweeping policy; touch throughput per tank-s is the judge at equal D.
Markers for save_16 (432 k, D 950): side 1000 touches/tank-s: control 0.060, 522 0.035, 524 0.020; CSV kill at D 1000: control 17.8, 524 6-10; mean μ canon_x staying > 0.3 = sweep installed.

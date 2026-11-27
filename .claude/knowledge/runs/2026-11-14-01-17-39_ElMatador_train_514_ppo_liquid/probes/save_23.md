# Probe train_514_ppo_liquid / save_23

- Index ≈ 621 k, D = 1000 (reached at 429 k, as in 510 and 513).
- Binary and compatibility identical to save_3: `build/` compiled from the working tree of d1bb4020, a·|a| present, launched in `--force`.
- Sides 550 and 1000, 4 episodes, stochastic, economy 10/3/30.
- CSV 600-630 k: hit 17-18 %, kill 22-23.6, skip 0, skip_canon 0.19-0.21, Hc −0.2 to 0. At same index, 510 gives hit 16-20 % and kill 20-24 (its 590-600 k window is the best). 513 stops at 565 k with hit 14 %.

## Summary

| run / save | side | ratio | p_al/p_off | hit/sh | h<5° | shots<5° | enrich | empty_s | %disarmed | shots/s | turret<5° | σ cx/cy | median life |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **514 s23** | 550 | 4.9 | 0.47/0.10 | 22.0 % | 33.6 % | 53 % | 2.40 | 9.6 | 12 % | 1.37 | **25.9 %** | 0.43/0.30 | 25.8 s |
| 510 s28 | 550 | 5.1 | 0.66/0.13 | 21.0 % | 34.5 % | 48 % | 2.16 | 6.4 | 16 % | 1.59 | 28.8 % | 0.21/0.22 | 26.6 s |
| 510 s16 | 550 | 13.6 | 0.45/0.03 | 24.9 % | 34.6 % | 60 % | 2.60 | 7.9 | 26 % | 0.59 | 32.6 % | 0.54/0.40 | 62.2 s |
| 513 s18 | 550 | 7.3 | 0.49/0.07 | 17.6 % | 28.2 % | 44 % | 2.27 | 9.2 | 13 % | 1.09 | 22.8 % | 0.44/0.41 | 39.1 s |
| **514 s23** | 1000 | 11.1 | 0.59/0.05 | 18.4 % | 32.2 % | 49 % | 2.95 | 16.0 | 15 % | 0.67 | 19.4 % | 0.54/0.39 | 77.7 s |
| 510 s28 | 1000 | 10.5 | 0.66/0.06 | 19.5 % | 34.4 % | 50 % | 2.68 | 8.6 | 14 % | 0.86 | 22.7 % | 0.59/0.51 | 52.8 s |
| 510 s16 | 1000 | 16.9 | 0.41/0.02 | 20.8 % | 30.8 % | 56 % | 2.63 | 11.5 | 22 % | 0.37 | 26.1 % | 0.60/0.46 | 180 s |
| 513 s18 | 1000 | 15.4 | 0.59/0.04 | 15.1 % | 26.3 % | 43 % | 2.26 | 23.2 | 9 % | 0.55 | 20.9 % | 0.55/0.48 | 144 s |

Aligned and <40 m counts: 478 / 308. μ canon saturated on 1.8 % / 0.7 % of decisions, σ never at ceiling.

## Reading

- **Trigger**: selective and monotone vs angle, no plateau. At 550, p_fire is 0.61 under 2°, 0.41 at 2-5°, 0.21 at 5-10° and 0.09-0.10 beyond 20°. Deterministic: fires aligned (0.47-0.23) and almost never off-axis (0.08-0.04): the game would not be mute. At 1000, aligned <5° alone gives p_fire 0.24, lower than the aligned-and-near combination (0.59): trigger also accounts for distance.
- **Fire**: 49-53 % of shots are aligned, with 32-34 % hits on these shots. hit/shell 22/18 %, at 510 save_28's level and above 513 save_18 (17.6/15.1). Off-axis shots weigh only 20-25 %.
- **Dump**: resolved. Median 1st empty reserve at 9.6/16 s, 12-15 % of steps disarmed, 59-76 % of tank-episodes empty (510 s28: 73-77 %).
- **Aim**: turret<5° 25.9/19.4 %, between 513 s18 and 510 s28. Probe shows no turret gain attributable to a·|a| nor centred potential.
- **Heads**: σ canon wide and dispersed (p10 0.12-0.18, p90 0.6), no ceiling. Movement saturated in a fixed direction (μ dir_x −0.99, dir_y −0.87 on average, |μ| ≈ 1): constant gait regime, like 510 s16 (dir_y +0.99) but on another axis. Median life at 550: 26 s, as 510 s28.

## Established / noise

- Established: 514 has the selective trigger and leaves the dump at D=1000. At 510 save_28's level on all fire metrics (ratio, hit/shell, h<5°, enrichment) and ahead of 513 save_18 in precision. The 10/3/30 economy did not reproduce 513's (5/30) less demanding fire at this stage.
- Noise: 514/510 reserve gap (16 s vs 8.6 s at 1000), turret (±3 points) and ratio at 550 (4.9 vs 5.1). Single save: 513 showed save-to-save relapses (17 vs 18).
- Not established: an own effect of a·|a| or centred potential. Turret did not exceed 510 and h<5° is identical.

## Useful next

- Probe 2-3 neighbouring saves (save_20/21/22) to place save_23 in the oscillating regime before concluding.
- To isolate a·|a|: compare fine aim error (miss of aligned shots, yaw/pitch at 50-200 m) with 510 save_28.

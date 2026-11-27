# Analysis train_506_ppo_liquid

- Agent: ppo_liquid, 32 tanks, launched 11/11 21:59, RUNNING
- Last row: 12/11 07:15, index 594 k, D 800, 61.8 k steps/h
- Effective commit: c02137c6 (21:59 binary, 22:00:01 sources, commit at 22:05), high confidence.

## What the run tests

Only diff vs 505: `continuous_target_entropy` 0 → [0, 0, −1, −1, −1] (dir_x, dir_y, yaw_cos, yaw_sin, pitch). 505 stopped at 198 k: beyond, the reference is the velocity lineage 500-504.

## CSV reading

D milestones (k):

| run | 350 | 400 | 500 | 600 | 650 | 700 | 800 | 1000 |
|---|---|---|---|---|---|---|---|---|
| 504 | 47 | 114 | 180 | 232 | 267 | 281 | 328 | 461 |
| 500-502 | 67 | 114 | 180 | 232 | 267 | 281-304 | 328-461 | 538-595 |
| 505 | 47 | 67 | 180 | n/a | n/a | n/a | n/a | n/a |
| 506 | 47 | 144 | 207 | 328 | 366 | 510 | 566 | n/a |

- 506 is the slowest of the lineage since D 400 (excluding 503). Stuck 144 k at D 650 (366 → 510 k), reaches D 800 with 200-240 k delay vs 501/502/504.
- At equal D, CSV hit is below 500-504 at every level: D 400: 8.5 % vs 10-14 %; D 500: 9.2 % vs 11-14 %; D 600: 10.0 % vs 12-16 %; D 650: 9.4 % vs 11-16 %. 505 was already at 9.3-9.9 % ⟹ a trait of the absolute setpoint, not of the target. fire higher (1.3-1.8 vs 0.6-1.1 up to D 500), miss larger (34-40 m vs 27-34 m).
- At D 700-800, 506 rejoins the lineage: hit 12.5 / 10.5 %, kill 7.9 / 9.4, ev 0.57 / 0.47.
- Trough 405-460 k (D 650): kill 11.7 → 2.2, fire 1.0 → 0.5, r_hit ÷3, Hd 0.38. alpha_fire leaves floor at 405 k and rises to 6.3e-3 at 553 k. Recovery (record hit 13.3 % at 516 k) coincides with this rise.
- Continuous α: all at floor 1e-4 since 55 k, except alpha_dir_y (peak 3.6e-3 at 168 k, between 140 and 240 k). The −1 yaw targets are never active.
- Healthy regime: kl 0.012-0.018, skip ≤ 0.03, ev 0.45-0.57.

## Diagnosis (probes: `probes/timeline.md`)

- **Established — yaw setpoint is a fixed bias.** From save_5, setpoint concentrates in a rear sector of the chassis (127-165°, resultant length 0.6-0.9), with μ_dir_y < 0: reversing, turret toward travel direction. Correlation with target position null globally and weak in a ±45° cone (+0.22 to +0.34). turret <5° stays flat at 16-18 % from save_1 to save_22.
- **Established — target −1 did not tighten aim.** σ_yaw dropped briefly (0.22 at save_5) then rose to 0.61 without entropy pressure (α at floor). The σyaw° decline (81 → 21°) comes from |μ| growing toward that bias, up to saturation (49 % of |μ_yaw_sin| > 1.5 at save_22).
- **Established — trigger selectivity but persistent dump.** Ratio 4.2-4.9 from save_13 to save_22, but reserve empties in 3.5-3.9 s in 95-98 % of tanks at every save. hit/shell below 504 at same D (8.7 vs 16.1 % at D 600).
- **Undecided:** cause of the 405-460 k trough. Trigger at its lowest at save_17 (p_arm 0.08) with alpha_fire reacting: partial extinction caught by discrete α, plausible but unproven.

## Options (for Samuel to decide)

1. Let run to D 1000 to compare probes with 504 save_16/18 at same D.
2. Consider that the absolute setpoint (505/506) brings no aim: turret <5° flat over 594 k and hit/shell below. Would be the 3rd failure of absolute after 411/413, this time with cos/sin and per-dim target.
3. Measure target tracking in the chassis frame (probe does not write turret angle; rear bias could be reverse-driving tracking unseen here).

## Landmarks

- D 1000 (504: 461 k); probe at D 950-1000 vs 504 save_16 (11.2 %, 9.1 s, ratio 4.4).
- turret <5° above 20 % and setpoint/target corr beyond ±45°: only signs of true absolute aim.

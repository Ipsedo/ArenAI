# Probe 505 save_2

- Run: train_505_ppo_liquid (turret + canon absolute setpoint, yaw (cos, sin), MIN_ALPHA 1e-4)
- Save: save_2, index 54 k, D 350
- Binary: `build/arenai_agent/arenai_agent_probe`, compiled 11/11 18:48, commit f9df436a + uncommitted 505 changes (= run code), no inference diff
- Sides 550 / 1000, 4 episodes, stochastic, economy 10 / +2 / 20 (config.json)

## Summary

| side | ratio | p_al | p_off | p_arm | hit/sh | h<5° | shots<5° | enrich | empty_s | %empty | %disarmed | shots/s | turret<5° | σyaw° | \|μ\| yaw cos/sin |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 550 | 1.43 | 0.73 | 0.51 | 0.53 | 4.2 % | 10.4 % | 15.3 % | 1.03 | 3.5 | 98 % | 47 % | 1.82 | 15.9 % | 76.8 | 0.13 / 0.08 |
| 1000 | 1.40 | 0.69 | 0.49 | 0.49 | 1.5 % | 3.4 % | 15.2 % | 0.97 | 3.7 | 100 % | 48 % | 1.74 | 16.2 % | 79.7 | 0.13 / 0.07 |

Closest landmark: 504 save_5 (135 k, side 550 / 1000): ratio 3.5 / 5.0, p_arm 0.20 / 0.15, hit/sh 8.4 / 3.9 %, empty 5.9 / 5.7 s, shots/s 0.86 / 0.69, turret<5° 13.8 / 12.7 %. No 500-504 probe at 54 k.

## Reading

- **Trigger**: flat at init (p_fire 0.50-0.55 in all angle slices, angle-only ratio 1.03 / 0.95). Ratio 1.4 comes from n = 383 / 132 aligned-near decisions: noisy, and shot enrichment is 1.0. No selectivity.
- **Dump**: reserve empties in 3.5 s in 98-100 % of tanks, 47 % of steps disarmed, 1.8 shots/s (2× 504 save_5). Inflates CSV `fire` and `r_hit`.
- **Aim**: turret <5° = 16 % (504 save_5: 13.8 / 12.7 %), median angle 13°. Slightly better although setpoint is nearly random: |μ| in (cos, sin) 0.13 / 0.08 for σ ≈ 0.28 / 0.25 ⟹ angular dispersion of setpoint 77-80°. Limited slew speed (π rad/s) low-pass filters this noise: mechanism of 413.
- **Heads**: σ free (0 % at ceiling), μ never saturated. σ_yaw has not moved from init (0.3); |μ_yaw| must grow for yaw to become precise (≈ σ/|μ| rad).
- **Hit/shell**: 4.2 % side 550 vs 8.4 % for 504 at 135 k. Aligned shots hit at 10 %.

## Established vs noise

- Established: trigger at init level without selectivity, full dump, yaw setpoint at ~78° dispersion.
- Not established: the slight turret <5° gain (16 vs 14 %), at an index 2.5× smaller than the landmark.
- CSV lead at 20 k (hit 9.4 %, r_hit ×1.7) is mostly explained by the dump's fire rate, not precision.

## Next

Probe save_5 (135 k) to compare to 504 save_5 at same index. Indicators: |μ_yaw| (must exceed σ), σyaw° (must fall well under 77°), turret<5° and appearance of a p_fire vs angle slope.

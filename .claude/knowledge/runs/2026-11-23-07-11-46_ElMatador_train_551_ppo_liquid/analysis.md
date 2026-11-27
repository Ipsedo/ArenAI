# Run 551 — ppo_liquid

Launched 2026-11-23 07:11, last line 12:46, **running**, index 508 k, 91 k steps/h (2.9 M tank-steps/h). Code: `65899e8a` + uncommitted direct branch encoding → heads (actor + critic) (`liquid_actor.*`, `liquid_critic.*`); binary `cmake-build-release` dated 07:11, same minute as launch. Good confidence.

## What the run tests

Twin of 550: identical config, sole change = the direct branch in the liquid cell.

## CSV

D milestones: 500 @ 145 k, 700 @ 267 k, 900 @ 366 k (550: 900 @ ~405 k, gap within seed noise). D blocked at 900 since 142 k.

| D | fire (/s) | hit | kill | r_hit | Hd |
|---|---|---|---|---|---|
| 350-550 | 1.5-1.7 | 0.11-0.14 | 22-26 | 8.4-8.9e-3 | 0.67 → 0.56 |
| 600-700 | 1.2 → 0.78 | 0.15-0.17 | 26 → 23 | 9.1 → 7.0e-3 | 0.47 → 0.43 |
| 750-850 | 0.89 → 0.57 | 0.16-0.20 | 22 → 21 | 6.0 → 5.1e-3 | 0.40 → 0.32 |
| 900 (366-508 k) | 0.39 | 0.24 | 17.7 | 3.7e-3 | 0.27 |

At D 900, by 20 k slices: kill 20.1 → 15.4 (440 k) → 18.3 → 17.3; fire 0.57 → 0.28 → 0.49 → 0.40 → 0.18 (last slice, 9 rows); hit 0.23 → 0.28 (run max at 505 k). 550 at D 900 (420-660 k): kill 13.7-19.8, fire 0.13-0.50, hit 0.25-0.32.

## Diagnosis

- Established: the trigger rarefies (fire ÷4 from D 550 to D 900) while hit doubles (0.12 → 0.25-0.28). kill loses a third (26 → 17), does not tend to 0.
- Established: not the "extinct trigger" regime (fire → 0.02, kill → 0 of runs 465/471/479). kill is stable at 17-18 over the last 60 k.
- In the noise: 551 ≡ 550 at D 900 (kill, fire, hit in the same ranges). The direct branch does not change the trigger trajectory at CSV.
- Not settled: the fire of 0.10-0.20 on the last 10 rows. 550 had the same trough (0.13 at 600 k) then rose to 0.42-0.50. To confirm over 30-50 k more.
- Not visible at CSV: trigger closed > 100 m and turret sweep (550's finding).
Markers: real extinction if kill goes under 12 and fire stays < 0.15 over 40 k. Probe save_17 or save_18 at side 900 to compare with `550/probes/save_21_side900`: p_fire by distance, hit/shell, sweep.

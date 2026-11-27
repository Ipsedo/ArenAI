# train_556_ppo_liquid

| | |
|---|---|
| machine | ElMatador |
| user | samuel |
| launch | 2026-11-27 02:00:15 |
| commit at launch | `5367a45e` — docs(harness): skip commit hooks once final-check passed |
| commits ≤ 90 min after | none |
| diff at launch | none (clean tree, run in progress) |
| raw data | `outputs/train_556_ppo_liquid` on ElMatador |

## Established facts

- 2026-11-27, 568 k: D 1000 at 461 k (554: 950 max, 555: stuck 850), kill 21 vs 18 over 430-546 k, but hit 0.17 vs 0.27-0.29 and fire ×2.
- 2026-11-27, matches s17 and s21 (`matches/save_21.md`): below 554 s17 / 555 s17 in game — dry at 10 s 0.27 / 0.48 (0.03-0.05), hit/shell 0.18-0.26 (0.32-0.39), armed gun < −5° 70 % of the time; 553 s36 profile. One seed, confounded.
- `mask_loss` 0.28 → 0.18-0.21, no clear plateau.
- 2026-11-27, probe s21 (`probes/save_21.md`) and shots by range vs 554 s17 / 555 s17, side 1000 (`comparisons/2026-11-27_shots-by-range-554-555-556.md`): share of shells with best-aligned target > 200 m ≡ in the three runs (0.53 / 0.51 / 0.55) ⟹ not the cause. Specific to 556: under 100 m, 57 % of yaw-aligned shells leave with pitch error > +2° (554 38 %, 555 24 %) and hit 0.36 there (0.69 / 0.63); close-range hit/shell 0.42 vs 0.58 / 0.56; volume 0.46 shots/tank-s (0.37 / 0.21); first empty reserve median 18 s (43 / 35). Armed gun median −11.7° (−1.9 / −3.1).
- 2026-11-27, gun elevation diagnosis s21, side 1000 (`probes/save_21_gun_elevation.md`): pitch servo has the right sign (corr −0.31); firing setpoint follows the camera-parallax curve (gun +6.5 / +2.0 / +0.7° below the target at < 40 / 40-70 / 70-110 m, as 554; 555 holds the gun above); resting posture has no return to the horizon (μ_canon_y ≥ 0 above −3°, 554 / 555 raise the gun between −25° and −3°). Oracle pitch only: hit/shell 0.185 → 0.281, close-range 0.42 → 0.68, first empty reserve 18 → 35 s ⟹ elevation alone accounts for the gap to 554 / 555 on train rules.

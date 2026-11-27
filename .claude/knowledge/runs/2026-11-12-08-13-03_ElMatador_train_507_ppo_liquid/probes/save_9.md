# Probe 507 save_9 (243 k, D 650) vs 504 save_9 (D 600)

`build/` binary of 12/11 08:01, compiled from the tree committed afterwards as 25a75a34 (`--force`). 4 episodes per side, stochastic, economy 10 / +2 / 20. 504 save_9 re-analysed from its CSVs.

| run | side | p_arm | ratio | p_al / p_off | hit/sh | h<5° | empty_s | %disarmed | shots/s | turret<5° | \|μ\|cx | corr(yaw, μ_cx) ±40° (slope /10°) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 507 | 550 | 0.17 | 4.5 | 0.56 / 0.13 | 9.9 % | 21.1 % | 4.3 | 41 % | 0.87 | 14.9 % | 0.33 | −0.29 (−0.076) |
| 504 | 550 | 0.05 | 6.1 | 0.19 / 0.03 | 16.1 % | 29.1 % | 22.2 | 16 % | 0.43 | 12.5 % | 0.30 | −0.30 (−0.062) |
| 507 | 1000 | 0.12 | 4.7 | 0.46 / 0.10 | 5.1 % | 10.8 % | 4.1 | 38 % | 0.66 | 13.4 % | 0.32 | −0.20 (−0.050) |
| 504 | 1000 | 0.03 | 7.0 | 0.18 / 0.03 | 7.6 % | 15.6 % | 41.7 | 15 % | 0.28 | 10.1 % | 0.30 | −0.22 (−0.039) |

CSV at 243 k: 507 D 650, hit 14 %, kill 14, fire 0.84-1.05; 504 D 600, hit 16 %, kill 10, fire 0.38-0.43.

- **Turret: slightly better than 504, far from threshold.** turret<5° +2.4 / +3.3 points (14.9 / 13.4 % vs 12.5 / 10.1 %), corrector gain +23-28 % (0.076 vs 0.062 μ per 10°), equal correlation. First gap in the right direction, but of a size comparable to seed noise and below the 18 % threshold fixed before the run.
- **Trigger: 504 learned economy, 507 did not.** 504 lowered its fire level (p_arm 0.05, 0.3-0.4 shots/s) and keeps its reserve 22-42 s; 507 still empties it in ~4 s (97 % of tanks), at 3× the p_fire level. Relative selectivity exists (ratio 4.5-4.7) but stays under 504 (6.1-7.0).
- **Consequence: hit/shell falls back under 504** (9.9 / 5.1 % vs 16.1 / 7.6 %), whereas 507 led at save_2 and save_5. CSV kill lead (×1.4) comes from fire volume, not precision.

## Verdict

At 243 k the 10° aim potential produced only a marginal turret gain (not established, within noise), and trigger trajectories diverge: 504 left the dump between save_5 and save_9, 507 did not. Unverified hypothesis: the potential pays aim independently of fire, so a shot's net reward stays higher and delays learning restraint (or seed noise on transition timing, which 500/501/502 also showed). Useful next save: save_13 (351 k) to see whether the dump eventually fades.

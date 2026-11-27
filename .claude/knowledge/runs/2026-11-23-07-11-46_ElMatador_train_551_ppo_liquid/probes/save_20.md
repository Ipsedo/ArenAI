# Probe 551 save_20

Run 551 (ppo_liquid, 550 + direct branch), save_20, index 540 k, D 900. Binary `build/arenai_agent/arenai_agent_probe` of 2026-11-23 06:58, commit `65899e8a` + uncommitted direct branch (the 4 sources date from 06:55-06:56). No inference diff. Side 900, 4 episodes, 32 tanks, stochastic, 165 k decisions. Reference: `550/probes/save_21_side900` (567 k, D 900). Reserve: that 550 probe was made with a binary at commit `10211f76`, with an environment diff vs the run's code.

## Summary

| | 551 s20 | 550 s21 |
|---|---|---|
| ratio (aligned & < 40 m / off-axis) | 43.9 (n 634) | 31.0 (n 413) |
| p_fire aligned close / off-axis | 0.520 / 0.012 | 0.544 / 0.018 |
| armed p_fire | 0.022 | 0.026 |
| hit/shell | 0.312 | 0.257 |
| median miss (m) | 19.6 | 52.2 |
| shots/tank/s | 0.207 | 0.240 |
| 1st empty median (s), % empty | 63.3 / 48 % | 43.9 / 52 % |
| disarmed | 2.2 % | 4.8 % |
| tour < 5° / < 10° | 19.5 % / 39.9 % | 17.5 % / 37.2 % |
| σ canon_x / canon_y (p50) | 0.269 / 0.172 | 0.599 / 0.307 |
| p_zoom | 0.848 | 0.572 |

## Trigger by distance (armed, aligned < 5°)

| target distance | 551 p_fire (n) | 550 p_fire (n) | 551 hit of aligned shots |
|---|---|---|---|
| < 40 m | 0.520 (634) | 0.544 (413) | 0.91 |
| 40-60 | 0.433 (846) | 0.430 (878) | 0.69 |
| 60-100 | 0.193 (1362) | 0.121 (1634) | 0.40 |
| 100-150 | 0.036 (1428) | 0.050 (1254) | 0.21 |
| 150-200 | 0.013 (1655) | 0.026 (1424) | 0.32 |
| > 200 | 0.021 (23856) | 0.022 (19091) | 0.25 |

Touches: 70 % at < 60 m, 18 % at > 100 m (550: 71 % / 18 %).

## Turret

| | 551 s20 | 550 s21 |
|---|---|---|
| mean Δ (°/decision) | −10.9 | −8.6 |
| median \|Δ\| | 13.0 | 10.9 |
| share \|Δ\| > 4° | 88 % | 82 % |
| share \|Δ\| < 1.5° | 4.8 % | 7.0 % |
| mean tanh(μ canon_x) | +0.63 | +0.55 |
| tanh(μ) aligned target / out of field | +0.38 / +0.67 | +0.39 / +0.58 |

## Reading

- Trigger: selective and open at short range (0.52 at < 40 m, 0.43 at 40-60 m), closed beyond 100 m (0.01-0.04). Not extinct. Profile identical to 550.
- Shot: hit/shell 0.31, median miss 19.6 m. Dump absent (disarmed 2.2 %).
- Turret: continuous sweep in one direction at ~110°/s, 88 % of time at more than 4°/decision. μ does not cancel on an aligned target (+0.38): partial brake, no stop.
- Heads: σ canon_x 0.27 vs 0.60 on 550, p_zoom 0.85 vs 0.57.
Conclusions: established: the direct branch removes neither the sweep nor the trigger closing beyond 100 m. In the noise or to confirm: hit/shell +0.05, ratio 44 vs 31 (n 634 and 413), later empty reserve; one checkpoint per run, 550 binary not identical. Clear difference but effect not established: σ canon_x ÷2, median miss ÷2.7, p_zoom +0.28.
Next useful: probe 551 save_12 at side 550 to compare with `550/probes/save_12_side550` and date the fall of σ canon_x. Re-probing 550 save_21 with the current binary is not possible as is (the binary contains the direct branch).

# Probe 533 save_3: first sign of aim

Run `train_533_ppo_liquid` (coloured AR(1) noise ρ 0.9 **and** short value head γ 0.9: two confounded levers), uncommitted on 68c41e20. save_3, index 81 k, D 350. Run stopped at 88 k (2026-11-18 21:13) at Samuel's request. Binary `build/arenai_agent/arenai_agent_probe` (2026-11-18 20:04), same commit, no inference diff. Sides 250 and 550, 4 episodes, stochastic. Controls at same age and side 250: 529 s3 and 532 s3 (sigma_gradients exports), 528 s1; 533 s0/s1.

| side | ratio | p_al | p_off | hit/sh | h<5° | shots<5° | empty_s | %disarmed | shots/s | tour<5° | σx | \|μ\|cx |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 250 | 1.02 | 0.67 | 0.66 | 13.7 % | 32 % | 18 % | 3.4 | 34 % | 1.95 | 14.8 % | 0.18 | 0.34 |
| 550 | 1.03 | 0.68 | 0.66 | 7.1 % | 19 % | 19 % | 2.8 | 56 % | 1.66 | 15.1 % | 0.19 | 0.30 |

## Aim (yaw toward nearest enemy, < 100 m, step ≥ 10)

| | 533 s1 (250) | 533 s3 (250) | 533 s3 (550) | 529 s3 (250) | 532 s3 (250) | 528 s1 (250) |
|---|---|---|---|---|---|---|
| corrective turret μ, 5-40° | 0.59 | 0.63 | 0.65 | 0.55 | 0.55 | 0.585 |
| corrective turret μ, 5-20° | 0.59 | 0.65 | 0.69 | – | – | – |
| gain (° command per ° yaw) | −0.008 | −0.024 | −0.025 | −0.005 | −0.004 | −0.007 |
| corr(command, yaw) | −0.25 | −0.25 | −0.26 | −0.11 | −0.12 | −0.20 |
| yaw < 5° | 4.4 % | 6.9 % | 6.4 % | 4.7 % | 4.3 % | 5.1 % |
| yaw < 20° | 15.9 % | 21.4 % | 19.1 % | 16.2 % | 15.0 % | 17.6 % |
| dwell < 5° (s) | 0.55 | 0.31 | 0.31 | 0.50 | 0.61 | 0.36 |
| mean command (°/decision) | 0.45 | 1.39 | 1.23 | 0.67 | 0.47 | 0.50 |
| σ canon_x | 0.20 | 0.18 | 0.19 | 0.22 | 0.21 | 0.43 |

- The criterion set at save_1 is met: the share of corrective commands rises from s1 to s3 (0.59 → 0.63-0.65; 0.65-0.69 near the target), where 529 and 532 were at 0.55 at s3. s.e. ≈ 0.5 point: gap with controls (8-10 points) beyond sampling noise; one seed.
- Gain tripled (−0.008 → −0.025) and is 5-6× 529/532's at s3. At 20° gap the command is 0.5°/decision: still weak in absolute (an aiming turret answers within a few decisions).
- Time on target: yaw < 5° 6.9 % vs 4.3-4.7 % for controls; yaw < 20° 21 % vs 15-16 %.
- Beyond 60° gap: nothing (0.49-0.51 corrective). The turret corrects what is in field, does not search.
- Dwell falling (0.55 → 0.31 s): the command grew (1.2-1.4°/decision, bias +0.2 to +0.4°) and crosses the target; read with the gain, not as regression. Point to watch (sweep of 528/529).
- Trigger: still flat (0.66-0.68, all distances), dump 2.8-3.4 s, disarmed 34-56 %. Shot: hit/shell 13.7 % at 250 (s1: 9.5 %), aligned shell 32 %.
CSV: 81-88 k kill 21.1-21.3, hit 0.108-0.113, fire 1.9/s, D 350 (528: kill 18.7, hit 0.136, D 350; 529: kill 15.4-16.9, D 250); kill flat at 21-22 since 40 k, D 300 at 47 k, 350 at 70 k (528's calendar); **skip KL 55-58 %**, kl 0.07, Hc −0.33: more than half of actor updates cut (528: 3-7 %).
Conclusion: established: at 81 k 533's turret answers yaw better than 528/529/532 at same age (sign, gain, time on target), sides 250 and 550 concordant. Not attributable: 533 cumulates coloured noise and short value head; 532 (short head alone, white noise) was at 529's level at s3, designating coloured noise without proving it (532 had the head optional, other code). Not established that it holds (529 had its best point at s9 then tipped; skip 55 % is a regime the controls did not have). One seed, one save.

# Probe 537 save_1 and save_3: the aim sign of coloured noise is not reproduced

Run `train_537_ppo_liquid` (coloured noise ρ 0.9 + actor batch 2400 + a·|a| removed). Saves: save_1 (27 k, D 250), save_3 (81 k, D 350). Binary `build/arenai_agent/arenai_agent_probe` of 2026-11-19 07:30, same working tree as training binary (07:31), no inference diff. Sides 250 and 550, 4 episodes, stochastic. Controls: 534 s3, 533 s1. Command in °: 9.2 × tanh(μ) for 537 (linear), 9.2 × tanh(μ)·|tanh(μ)| for 533/534.

| save | side | ratio | p_al | p_off | hit/sh | h<5° | shots<5° | empty_s | %disarmed | shots/s | σx | \|μ\|cx |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s1 | 250 | 1.03 | 0.58 | 0.56 | 10.7 % | 24 % | 16 % | 4.5 | 47 % | 1.66 | 0.17 | 0.13 |
| s1 | 550 | 1.03 | 0.57 | 0.55 | 5.1 % | 13 % | 14 % | 3.2 | 61 % | 1.37 | 0.17 | 0.12 |
| s3 | 250 | 1.21 | 0.66 | 0.55 | 12.3 % | 30 % | 14 % | 3.3 | 50 % | 1.41 | 0.16 | 0.19 |
| s3 | 550 | 1.23 | 0.64 | 0.52 | 6.5 % | 18 % | 13 % | 3.0 | 66 % | 1.18 | 0.18 | 0.17 |

534 s3: ratio 1.02 / 1.02, hit/shell 11.9 / 6.5 %, h<5° 28 / 17 %.

## Aim (yaw toward nearest enemy, < 100 m, step ≥ 10)

| | 537 s1 (250 / 550) | 537 s3 (250 / 550) | 534 s3 (250 / 550) | 533 s1 (250) | 529 / 532 s3 |
|---|---|---|---|---|---|
| corrective turret μ, 5-40° | 0.605 / 0.593 | 0.569 / 0.572 | 0.624 / 0.636 | 0.593 | 0.55 / 0.55 |
| corr(command, yaw) | −0.25 / −0.25 | −0.16 / −0.15 | −0.28 / −0.26 | −0.25 | −0.11 / −0.12 |
| gain (°/°) | −0.019 / −0.017 | −0.018 / −0.016 | −0.031 / −0.027 | −0.008 | −0.005 / −0.004 |
| yaw < 5° | 4.9 / 4.4 % | 5.8 / 5.4 % | 7.0 / 6.4 % | 4.4 % | 4.7 / 4.3 % |
| yaw < 20° | 17.4 / 16.0 % | 19.6 / 17.1 % | 22.2 / 19.6 % | 15.9 % | 16.2 / 15.0 % |
| dwell < 5° (s) | 0.32 / 0.31 | 0.28 / 0.33 | 0.29 / 0.36 | 0.55 | 0.50 / 0.61 |
| mean command (°/decision) | 1.37 / 1.21 | 1.94 / 1.83 | 1.54 / 1.31 | 0.45 | 0.67 / 0.47 |
| μ autocorr at 0.1 s | 0.94 / 0.94 | 0.64 / 0.56 | 0.83 / 0.83 | 0.95 | – |

- Turret: s1 at 533 s1's level (corrective 0.59-0.60, corr −0.25). From s1 to s3 the corrective share **falls** (0.60 → 0.57) and correlation degrades (−0.25 → −0.15), where 533/534 rose to 0.62-0.65. This is the trajectory of white-noise runs (529/532: 0.55 at s3). Yaw < 5° 5.4-5.8 %, between white (4.3-4.7) and 534 (6.4-7.0).
- μ chopped: autocorr of μ at 0.1 s fell from 0.94 to 0.56-0.64 (534 s3: 0.83); mean command 1.8-1.9°/decision. Coloured noise is indeed active (noise autocorr 0.89 / 0.56 / 0.32).
- Trigger: first selectivity at s3, ratio 1.21-1.23 (gap 0.117, p_fire monotone from 0.54 to 0.63 from 45° to 0°), vs flat 1.02 for 533/534/535 at same age. Deterministic: aligned 0.92 / off-axis 0.65 (side 250).
- Shot: hit/shell ≡ 534; dump unchanged (empty at 3.0-3.3 s, disarmed 50-66 %).
Conclusion: established: at s3 the aim gain attributed to coloured noise (three runs, 533/534/535) does not appear in 537; the trigger, in contrast, leads the three. Confounded: actor batch 2400 and removal of a·|a|. 526 (batch 2400, white noise) had gained nothing on aim; the linear command gives 1.7° of command for |μ| 0.19 vs 0.3° with a·|a|. Useful next: s5 (135 k); control `--actor_batch_size 512` judged at s3.

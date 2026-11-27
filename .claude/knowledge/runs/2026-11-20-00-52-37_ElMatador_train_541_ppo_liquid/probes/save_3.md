# Probes 541 save_1, save_2, save_3: canon credit on the critic's H head

Run `train_541_ppo_liquid`, launched 2026-11-20 00:52 from `cmake-build-release` (uncommitted tree above b83da585). Config identical to 539. Only difference: canon credited at one step on γ·σ(H(s′)) − σ(H(s)), H = `to_shot_value` (linear head of the critic, auxiliary BCE ×0.1), group KL stop. Saves: save_1 (27 k, D 250), save_2 (54 k, D 300), save_3 (81 k, D 350). Probe binary `build/arenai_agent/arenai_agent_probe` of 2026-11-20 01:39, no inference diff. Sides 250 and 550, 4 episodes, stochastic. Controls: 539 s3 (current image, single advantage), 540 s3 (H on a third CNN), 538 s1.

## Script summary (250 / 550)

| run save | ratio | p_al | p_off | hit/sh | h<5° | shots<5° | empty_s | %disarmed | shots/s | tour<5° | σx | \|μ\|cx | kills |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 538 s1 | 1.05 / 1.05 | 0.60 / 0.59 | 0.57 / 0.57 | 10.9 / 5.7 % | 26 / 14 % | 19 / 20 % | – | – | 1.79 / 1.48 | 14.6 / 15.2 % | 0.17 / 0.18 | 0.11 / 0.13 | – |
| 541 s1 | 1.04 / 1.04 | 0.64 / 0.64 | 0.61 / 0.61 | 10.8 / 4.4 % | 34 / 19 % | 18 / 14 % | 2.8 / 2.7 | 64 / 74 % | 1.09 / 0.84 | 7.9 / 6.4 % | 0.12 / 0.12 | 0.06 / 0.05 | 18.5 / 7.3 |
| 541 s2 | 1.20 / 1.19 | 0.76 / 0.77 | 0.63 / 0.64 | 11.6 / 4.8 % | 39 / 20 % | 19 / 15 % | 2.7 / 2.5 | 59 / 75 % | 0.99 / 0.76 | 7.3 / 5.5 % | 0.09 / 0.07 | 0.04 / 0.04 | 18.3 / 8.8 |
| 541 s3 | 1.22 / 1.20 | 0.82 / 0.82 | 0.67 / 0.68 | 11.7 / 7.1 % | 39 / 26 % | 21 / 22 % | 2.9 / 2.6 | 45 / 65 % | 1.39 / 1.09 | 9.7 / 10.1 % | 0.08 / 0.07 | 0.05 / 0.05 | 21.3 / 15.3 |
| 539 s3 | 1.06 / 1.11 | 0.81 / 0.82 | 0.76 / 0.73 | 13.6 / 7.5 % | 39 / 22 % | 23 / 23 % | 4.5 / 2.4 | 29 / 53 % | 2.42 / 1.78 | 17.1 / 16.3 % | 0.09 / 0.11 | 0.10 / 0.11 | 27.8 / 19.8 |
| 540 s3 | 1.04 / 1.06 | 0.70 / 0.70 | 0.67 / 0.66 | 12.2 / 7.6 % | 35 / 22 % | 23 / 26 % | – | – | 1.47 / 1.07 | 12.1 / 12.3 % | 0.05 / 0.05 | 0.06 / 0.05 | 23.0 / 14.3 |

## Aim (yaw toward nearest enemy, < 100 m, step ≥ 10, `aim_lin.py`, 250 / 550)

| | 541 s1 | 541 s2 | 541 s3 | 539 s3 | 540 s3 | 538 s1 |
|---|---|---|---|---|---|---|
| dwell < 5° (s) | 0.59 / 0.62 | 0.71 / 0.83 | 0.81 / 1.13 | 0.50 / 0.62 | 0.84 / 1.28 | 0.32 / 0.35 |
| passes ≥ 1 s | 17 / 17 % | 23 / 28 % | 26 / 40 % | 13 / 19 % | 27 / 37 % | 3 / 5 % |
| number of passes | 346 / 327 | 228 / 188 | 174 / 186 | 393 / 463 | 193 / 193 | 658 / 802 |
| yaw < 5° | 4.8 / 3.9 % | 4.2 / 3.2 % | 3.9 / 4.2 % | 6.9 / 8.0 % | 4.3 / 5.1 % | 4.5 / 4.5 % |
| corrective turret μ, 5-40° | 0.538 / 0.526 | 0.546 / 0.517 | 0.527 / 0.511 | 0.556 / 0.589 | 0.544 / 0.514 | 0.561 / 0.553 |
| μ / yaw gain, 5-40° | −0.003 / −0.002 | −0.003 / −0.002 | −0.002 / −0.002 | −0.010 / −0.012 | −0.003 / −0.002 | −0.013 / −0.012 |
| mean command (°/decision) | 0.63 / 0.60 | 0.49 / 0.42 | 0.60 / 0.56 | 1.12 / 1.20 | 0.65 / 0.64 | 1.13 / 1.24 |
| yaw drift over 1 s, start 5-20° | +15.7 / +15.7 | +16.5 / +13.2 | +20.1 / +16.4 | +24.6 / +24.2 | +19.3 / +16.0 | +24.1 / +24.5 |

## Reading

- The turret does not turn toward the target. Corrective gesture at chance (0.51-0.55) and falling from s1 to s3 at side 550 (0.526 → 0.517 → 0.511). μ gain on yaw 5× weaker than 538/539 from s1 and does not move.
- It moves less. Command 0.4-0.6 °/decision vs 1.1-1.2 (538 s1, 539 s3), |μ| canon_x 0.04-0.06 vs 0.10-0.13. The long dwell (0.8-1.1 s at s3) comes from there: an almost immobile turret stays longer on a passing target and leaves it more slowly (drift +13 to +20° vs +24°).
- Less often aligned. Yaw < 5° 3-5 % vs 7-8 % (539 s3); turret within 5° of any target 6-10 % vs 15-17 %. Passes 2.5× fewer than 539 s3.
- Same signature as 540 s3 (H on a third CNN) on all aim lines: the effect comes from the one-step credit on H, not where H is plugged.
- What learns: the trigger on angle. p_fire 0.76-0.82 at < 10° vs 0.62-0.68 at > 20°, ratio 1.04 → 1.20 → 1.22 (539 s3: 1.06-1.11, flat). Stays high off-axis and dump whole (empty at 2.5-2.9 s, 45-75 % disarmed steps).
- Shot: hit per aligned shell equal or better than 539 (39 / 26 %), but fewer shots and less aligned time ⟹ probe kills 21 / 15 vs 28 / 20.
Established over 3 saves and 2 sides: no aim gesture learned, turret calmed, identical to 540. Noise: kill and hit/shell gaps (one run per arm, 4 episodes). Not settled: whether the angle-selective trigger ends up paying the canon via H later; whether elevation (|μ| canon_y 0.09 → 0.20 → 0.13) is a learned gesture. To follow: save_5 (135 k, D 400 expected): corrective and yaw < 5° vs 539 s5 (0.56-0.59, 7.5-9 %); CSV kill at D 400-450 vs 539 (25-26).

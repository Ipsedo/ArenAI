# Probe 522 save_9 (243 k, D 650)

Run 522 = 521 + penalty −D/1000 at the step the reserve empties (uncommitted at that time). Binary `cmake-build-release` of 2026-11-15 20:07, commit 9d843717 + uncommitted `train_environment.cpp` and `shell_economy.h`, no inference diff. Sides 550/1000, 4 episodes, 32 tanks, economy 10/2/30, stochastic then deterministic. Probe does not respawn and does not apply the penalty.

| 550 / 1000 | shots/s | hit/sh | h<5° | touches/tank-s | ratio | p_al | p_off | 1st empty p50 | %empty | %disarmed | tour<5° |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 520 save_9 (D 650) | 1.93 / 1.19 | 12.4 / 7.1 % | 31 / 23 % | 0.240 / 0.084 | 1.4 / 1.6 | | 0.62 / 0.54 | 2.7 / 2.6 s | 88 / 99 % | 42 / 64 % | 21 / 14 % |
| 521 save_9 (D 600) | 1.32 / 1.06 | 12.4 / 7.3 % | 30 / 22 % | 0.164 / 0.078 | 2.8 / 2.8 | | 0.18 / 0.15 | 4.1 / 4.0 s | 89 / 95 % | 28 / 35 % | 17 / 13 % |
| **522 save_9 (D 650)** | 0.45 / 0.33 | 21.8 / 17.4 % | 34.5 / 32.4 % | 0.098 / 0.057 | 2.5 / 2.3 | 0.088 / 0.073 | 0.036 / 0.031 | 25.4 / 36.9 s | 71 / 73 % | 12 / 11 % | 25 / 20 % |
| 521 save_9 det | 0.42 / 0.25 | 19.1 / 14.9 % | 34 / 29 % | 0.079 / 0.038 | 2.2 / 2.6 | | | 5.6 / 5.5 s | 73 / 72 % | 7 / 7 % | 20 / 14 % |
| **522 save_9 det** | 0.014 / 0.011 | 21.4 / 12.6 % | 44 / 40 % | 0.003 / 0.001 | 1.4 / 2.5 | 0.070 / 0.093 | 0.050 / 0.037 | 78 / 50 s | 7 / 4 % | 0.5 / 1 % | 46 / 41 % |

## Reading

- Trigger: level dropped, not shape. Armed p_fire 0.055/0.039 (521: p_off alone 0.15-0.18). Aligned/off-axis ratio 2.3-2.5 as 521 (2.8): the penalty divided cadence by 3, did not make the trigger more selective. p_fire per angle: 0.09 at < 5°, 0.03-0.04 beyond 10°.
- Mute deterministic: p > 0.5 on 0.1-0.4 % of aligned states; 0.014/0.011 shot/s = 250-310 shots in 4 episodes. 521 save_9 still fired 0.42/0.25 deterministic.
- Shot: hit/shell ×1.75/×2.4 vs 521 at same index. 45/36 % of shots leave at < 5°; those hit 34.5/32.4 %. 23/32 % of shots still leave at ≥ 20° for 2 % touches.
- Reserve: initial dump gone. Median 1st empty 25/37 s (521: 4 s), p10 8/17 s. But 71-73 % of tanks eventually empty, over median lives of 143/180 s. Disarmed time 11-12 % (521: 28-35 %).
- Touch throughput falls: 0.098/0.057 per tank-second, −40 %/−27 % vs 521: the price paid so far.
- Aim: tour<5° 25/20 % (521: 17/13), 46/41 % deterministic: exploration noise (σ canon 0.26-0.31, no saturation) costs half the alignment. Mean μ dir_x −0.46 as earlier runs.
Established: penalty removes the start-of-episode dump and reduces cadence; per-shell precision at the level of 521 save_16 (22 %) with 190 k lead. Noise: ratio 2.3-2.5 vs 2.8 (aligned-and-close n ≈ 2 500); marker "ratio ≥ 2.8" neither reached nor clearly missed. Not settled: whether the level keeps falling to extinction (520 save_16: 0.13-0.29 shot/s) or selectivity takes off. CSV at 242 k: fire 0.47, kill 20.6-21.7.
Next: save_12-13 (D ≈ 800): stochastic shots/s must stay > 0.25 side 550, ratio > 3. save_16 (432 k, D ≈ 950-1000): compare 521 save_16 (hit/shell 22 %, 1st empty 9/20 s, ratio 5.7/9.5) and 520 save_16 (extinct, 0.29/0.13 shot/s). Deterministic: share of aligned states at p > 0.5 must rise, else the checkpoint does not fire in game.

# Trigger dynamics, save_1 → save_9 (side 550, stochastic)

| save | idx | D | p_fire 0-2° | 5-10° | 20-45° | ratio | shots/s | hit/sh | 1st empty p50 | disarmed | touches/tank-s | aligned states at p > 0.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 27 k | 300 | 0.634 | 0.632 | 0.629 | 1.0 | 1.59 | 5.4 % | 2.7 s | 57 % | 0.086 | 100 % |
| 3 | 81 k | 400 | 0.684 | 0.682 | 0.663 | 1.0 | 1.24 | 10.3 % | 2.6 s | 61 % | 0.127 | 99.5 % |
| 5 | 135 k | 450 | 0.543 | 0.487 | 0.300 | 1.8 | 0.90 | 13.5 % | 3.8 s | 55 % | 0.122 | 57 % |
| 7 | 189 k | 550 | 0.318 | 0.182 | 0.102 | 3.4 | 0.83 | 16.8 % | 8.1 s | 28 % | 0.140 | 22 % |
| 9 | 243 k | 650 | 0.094 | 0.056 | 0.040 | 2.5 | 0.45 | 21.8 % | 25.4 s | 12 % | 0.098 | 0.4 % |

- No rebound at save_9: on axis p_fire ÷1.7 from save_5 to 7 then ÷3.4 from save_7 to 9; off-axis ÷2.9 then ÷2.6. Ratio rose to save_7 (3.4) then fell (2.5): on the last interval the axis falls fastest.
- CSV after save_9: fire 0.48 → 0.25/s at 265 k, kill 21.7 → 19.4, r_hit 7.3 → 5.3·10⁻³.

Armed p_fire by shells remaining (all directions):

| shells left | 522 s5 | 522 s7 | 522 s9 | 521 s9 | 521 s16 |
|---|---|---|---|---|---|
| 1-2 | 0.356 | 0.131 | 0.054 | 0.229 | 0.132 |
| 3-5 | 0.368 | 0.140 | 0.056 | 0.229 | 0.127 |
| 6-10 | 0.365 | 0.137 | 0.050 | 0.229 | 0.106 |
| 11-20 | 0.514 | 0.302 | 0.078 | 0.404 | 0.293 |

- Trigger does not hold back when reserve nears zero: p_fire at 1-2 shells = at 6-10, in 522 and 521. Penalty not read as "the last shell costs", it acts as a uniform tax on firing.
- Precedent of late recovery: 520 (no penalty), p_fire 0-2° 0.056 at save_16 then 0.12/0.20/0.27/0.34 at save_20/24/28/32.

# save_10 and save_11 (side 550, stochastic): trough then on-axis rebound

| save | idx | D | p_fire 0-2° | 5-10° | 20-45° | 45-180° | ratio | shots/s | hit/sh | hit/sh < 5° | 1st empty p50 | %empty | disarmed | tour<5° | touches/tank-s | aligned at p > 0.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 9 | 243 k | 650 | 0.094 | 0.056 | 0.040 | 0.031 | 2.5 | 0.45 | 21.8 % | 34 % | 25.4 s | 71 % | 12 % | 25 % | 0.098 | 0.4 % |
| 10 | 270 k | 700 | 0.062 | 0.023 | 0.010 | 0.013 | 4.2 | 0.24 | 36.9 % | 52 % | 56.1 s | 34 % | 1 % | 31 % | 0.089 | 0.1 % |
| 11 | 297 k | 750 | 0.175 | 0.054 | 0.025 | 0.012 | 7.7 | 0.44 | 31.5 % | 48 % | 23.9 s | 41 % | 1 % | 22 % | 0.139 | 5.8 % |

- Trough at save_10; at save_11 the trigger recovers on axis (×2.8 at 0-2°) and stays at floor at 45-180°. Ratio 2.5 → 4.2 → 7.7.
- hit/shell 31.5-37 %: above anything 520 and 521 probed (520 save_16: 30 %). 53-59 % of shots at < 5°.
- Touches/tank-s back to 0.139 (save_7 level) with half the shots; disarmed 1 %.
- p_fire stays independent of shells left under 10 (save_11: 0.047/0.044/0.035 at 1-2/3-5/6-10).
- Single rebound point. CSV 275-335 k: fire oscillates 0.24/0.23/0.43/0.32/0.13, kill 16.5-18.9, r_hit 3.5-4.0·10⁻³ (521 at D 800: 7.4·10⁻³, kill 22.3).

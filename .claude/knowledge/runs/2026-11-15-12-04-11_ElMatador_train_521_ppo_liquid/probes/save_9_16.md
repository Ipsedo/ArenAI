# Probe 521 save_9 (243 k, D 600)

Run 521 = 520 config + `shells_per_hit` 3 → 2 (sole change, commit 9d843717 + uncommitted shell_economy.h). Binary `cmake-build-release` 12:03, no inference diff. Sides 550/1000, 4 episodes, economy 10/2/30 (520 probed in 10/3/30, its own economy).

| | shots/s | hit/sh | h<5° | touches/tank-s | ratio | p_off | 1st empty | %empty | %disarmed | tour<5° |
|---|---|---|---|---|---|---|---|---|---|---|
| 520 save_9 (D 650) | 1.93 / 1.19 | 12.4 / 7.1 % | 31 / 23 % | 0.240 / 0.084 | 1.4 / 1.6 | 0.62 / 0.54 | 2.7 / 2.6 s | 88 / 99 % | 42 / 64 % | 21 / 14 % |
| 521 save_9 (D 600) | 1.32 / 1.06 | 12.4 / 7.3 % | 30 / 22 % | 0.164 / 0.078 | 2.8 / 2.8 | 0.18 / 0.15 | 4.1 / 4.0 s | 89 / 95 % | 28 / 35 % | 17 / 13 % |
| 521 save_9 det | 0.42 / 0.25 | 19.1 / 14.9 % | 34 / 29 % | 0.079 / 0.038 | 2.2 / 2.6 | | 5.6 / 5.5 s | 73 / 72 % | 7 / 7 % | 20 / 14 % |

- No 520-style dump phase: p_off 0.15-0.18 vs 0.54-0.62, selectivity ×2 (ratio 2.8). CSV already showed: fire 1.5 → 0.9 between 180 and 240 k, without 520's runaway (2.6 at 225 k).
- Precision unchanged: hit/shell, h<5°, tour<5° identical to 520 at same index. N = 2 changed when the agent fires, not where it aims.
- Reserve still almost always emptied: 1st empty at 4 s (520: 2.7 s), 89-95 % of tanks empty. At hit/shell 7-12 % for a break-even of 50 %, the reserve is consumed without return: expected at this stage.
- Touches/tank-s lower at side 550 (0.16 vs 0.24): volume pays less, as the model predicted.
- To see: 520 extinguished between 250 and 450 k (fire 0.26/s, save_16), then raised cadence by lowering threshold to hit/shell 24 %. Question for 521: does nascent selectivity hold through this phase and does hit/shell exceed 520's 24-27 % at D 1000. Useful probes: save_16 (432 k), then save_24-28.

# save_16 (432 k, D 950)

| | shots/s | hit/sh | h<5° | touches/tank-s | ratio | p_off | 1st empty | %empty | %disarmed | tour<5° |
|---|---|---|---|---|---|---|---|---|---|---|
| 520 save_16 (432 k) | 0.29 / 0.13 | 30.2 / 23.2 % | 44 / 44 % | 0.087 / 0.031 | 8.4 / 12.8 | 0.015 / 0.010 | 60 / 81 s | 38 / 30 % | 1.4 / 1.2 % | 33 / 28 % |
| 520 save_32 (864 k, best) | 1.06 / 0.53 | 26.0 / 27.0 % | 40 / 40 % | 0.276 / 0.142 | | 0.061 / 0.025 | 22 / 28 s | 42 / 48 % | 3.1 / 2.3 % | 29 / 25 % |
| **521 save_16** | 1.06 / 0.50 | 22.2 / 22.3 % | 36 / 38 % | 0.235 / 0.112 | 5.7 / 9.5 | 0.071 / 0.035 | 8.9 / 19.7 s | 62 / 69 % | 8.4 / 6.9 % | 26 / 25 % |
| 521 save_16 det | 0.78 / 0.35 | 26.6 / 28.3 % | | 0.207 / 0.100 | | | 9.5 / 19.1 s | 56 / 47 % | 8.2 / 3.4 % | |
| 520 save_32 det | 0.81 / 0.36 | 27.9 / 29.5 % | | 0.227 / 0.107 | | | 20.7 / 35.5 s | 49 / 36 % | 2.7 / 1.0 % | |

- No extinction. At 432 k 520 hardly fired; 521 already has the cadence of 520's best save (save_32, 864 k). N = 2 removed the dump → extinction → recovery oscillation: same level in half the indices.
- Not more precise: hit/shell 22 % (det 27-28 %), h<5° 36-38 %, turret 25-26 %: slightly under 520 save_32. Reserve empties earlier (det 9.5/19 s vs 21/35 s): mechanical, 2 shells per touch instead of 3 at same precision.
- Run criterion (hit/shell above 24-27 % held at D 1000) not reached at this stage; judge at save_24-28.

# save_23 (621 k, D 1000): reserve empties earlier

| | shots/s | hit/sh | touches/tank-s | 1st empty p10/p50/p90 | %empty | %disarmed |
|---|---|---|---|---|---|---|
| 521 save_16 | 1.06 / 0.50 | 22.2 / 22.3 % | 0.235 / 0.112 | 2.8/8.9/36 · 3.1/19.7/59 s | 62 / 69 % | 8 / 7 % |
| 521 save_23 | 1.32 / 0.55 | 20.8 / 22.0 % | 0.274 / 0.121 | 2.1/5.6/20 · 2.7/11.1/47 s | 73 / 72 % | 12 / 6 % |
| 521 save_23 det | 1.24 / 0.50 | 23.6 / 21.2 % | 0.291 / 0.106 | 2.0/6.2/18 · 2.7/12.1/63 s | 68 / 66 % | 9 / 5 % |
| 520 save_24 (648 k) | 0.82 / 0.35 | 24.1 / 24.1 % | 0.198 / 0.084 | 3.1/15.2/39 · 11/45/125 s | 48 / 57 % | 3 / 2 % |
| 520 save_32 | 1.06 / 0.53 | 26.0 / 27.0 % | 0.276 / 0.142 | 5.3/22/46 · 10/28/94 s | 42 / 48 % | 3 / 2 % |

- Cadence rises (1.06 → 1.32 shots/s side 550), precision does not follow (hit/shell 21-22 %, det 21-24 % vs 27-28 % at save_16). 1st empty recedes 9-20 s → 6-11 s.
- Vs 520 at same index: reserve emptied 2-4× earlier, hit/shell 2-5 points lower. On both targeted measures 521 is behind 520.
- Same drift as 520 (reward pays touches/s, hence volume); N = 2 only makes it costlier in reserve.

# save_26 (702 k) and conclusion

| | shots/s | hit/sh | touches/tank-s | 1st empty p50 | %empty | %disarmed | ratio | p_off |
|---|---|---|---|---|---|---|---|---|
| 521 save_23 | 1.32 / 0.55 | 20.8 / 22.0 % | 0.274 / 0.121 | 5.6 / 11.1 s | 73 / 72 % | 12 / 6 % | 6.1 / 17.2 | 0.092 / 0.034 |
| 521 save_26 | 1.25 / 0.82 | 20.6 / 19.1 % | 0.258 / 0.157 | 5.3 / 8.6 s | 80 / 85 % | 19 / 15 % | 4.4 / 7.9 | 0.098 / 0.057 |
| 520 save_28 (756 k) | 0.95 / 0.38 | 24.9 / 24.5 % | 0.237 / 0.094 | 14.1 / 30.3 s | 60 / 58 % | 7 / 5 % | 4.8 / 19.4 | 0.067 / 0.019 |

CSV 600 → 728 k: hit 0.25 → 0.18-0.20, fire up to 1-2/s.
Conclusion: stop recommended. Drift toward volume continues (side-1000 selectivity halved in 80 k, reserve emptied by 80-85 % of tanks, 15-19 % of time disarmed) and 521 is behind 520 on both targets (hit/shell −4 to −5 points, 1st empty 3× earlier). Nothing in the mechanism reverses the trend: reward pays volume, N = 2 alone does not make emptying costly. Best 521 save on the targets: save_16 (det 27-28 %).

# Probe 543 save_13

Run 543 (541 + constant empty penalty 0.5), save_13, index 351 k, D 700 in CSV (374 k). Binary `build/arenai_agent/arenai_agent_probe` of 2026-11-20 18:39, no inference diff. Side 550, 4 episodes, 32 tanks, stochastic, economy 10 / 5 / 30.

| | 543 s8 | 543 s13 | 541 s13 | 541 s14 | 541 s16 | 539 s12 |
|---|---|---|---|---|---|---|
| index | 216 k | 351 k | 351 k | 378 k | 432 k | 324 k |
| p_fire aligned close | 0.80 | 0.148 | 0.71 | 0.35 | 0.08 | 0.56 |
| p_fire off-axis | 0.40 | 0.006 | 0.07 | 0.008 | 0.002 | 0.03 |
| shots/tank/s | 1.10 | 0.089 | 0.81 | 0.15 | 0.04 | 0.49 |
| disarmed steps | 49 % | 4 % | 18 % | 5 % | 0.2 % | 4 % |
| 1st empty reserve (median) | 3.6 s | 124 s | 16 s | 77 s | 151 s | 49 s |
| hit/shell | 0.114 | 0.157 | 0.184 | 0.166 | 0.236 | 0.209 |
| turret < 5° | 12.9 % | 4.4 % | 11.9 % | 2.7 % | 2.3 % | 17.2 % |

- Dump gone: 4 % of steps disarmed, reserve empty at 124 s median. It disappeared by closing the trigger, not by sorting: aligned close p_fire 0.15 (n 438), 0.047 on angle alone, 0.09 shot/tank/s. Deterministic the trigger is mute (0.4 % of aligned decisions).
- The rare shots are not better placed: 51 % leave at more than 20° (0.75 % touches), 23 % at under 5° (53 % touches). Aim followed: turret < 5° 4.4 % vs 12.9 % at save_8. CSV at 374 k: fire 0.19, kill 10.9, D 700 (541 was at D 800 at 351 k).
Conclusion: established: 543 reproduces 541's closing (save_14 to save_16), one to three saves earlier. The failure criterion set for save_16 (aligned close < 0.2, cadence < 0.15) is already met at save_13. Not settled: a passing reopening like 541 save_20-22 remains possible; it had not held (reclosed at save_24).

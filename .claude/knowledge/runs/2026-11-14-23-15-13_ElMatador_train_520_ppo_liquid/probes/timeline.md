# Probe chronology 520 (save_9 → save_37), compared with 514

Binary `build/` (23:58, `--force`, diff = probe `--zoom` option). Sides 550 / 1000, 4 episodes, stochastic, economy 10/3/30.
Site bias = signed median of the site error of aligned shots (< 5°) on the nearest target at 50-200 m. touches/tank-s = hit/shell × shots/s.

## 520

| save | idx | shots/s | hit/shell | h<5° | touches/tank-s | %disarmed | p_off | tour<5° | site bias |
|---|---|---|---|---|---|---|---|---|---|
| 9 | 243 k | 1.93 / 1.19 | 12.4 / 7.1 % | 31 / 23 % | 0.240 / 0.084 | **42 / 64 %** | **0.62 / 0.54** | 21 / 14 % | −0.2 / −0.1° |
| 16 | 432 k | **0.29 / 0.13** | 30.2 / 23.2 % | 44 / 44 % | **0.087 / 0.031** | 1.4 / 1.2 % | 0.015 / 0.010 | 33 / 28 % | −0.8 / −0.3° |
| 20 | 540 k | 0.54 / 0.32 | 27.6 / 24.5 % | 44 / 41 % | 0.149 / 0.079 | 1.3 / 1.2 % | 0.023 / 0.017 | 29 / 28 % | −0.1 / +0.0° |
| 24 | 648 k | 0.82 / 0.35 | 24.1 / 24.1 % | 40 / 38 % | 0.198 / 0.084 | 2.8 / 1.9 % | 0.057 / 0.021 | 29 / 28 % | −1.1 / −0.7° |
| 28 | 756 k | 0.95 / 0.38 | 24.9 / 24.5 % | 37 / 36 % | 0.237 / 0.094 | 6.6 / 4.5 % | 0.067 / 0.019 | 32 / 27 % | −0.8 / −1.1° |
| 32 | 864 k | 1.06 / 0.53 | 26.0 / 27.0 % | 40 / 40 % | 0.276 / 0.142 | 3.1 / 2.3 % | 0.061 / 0.025 | 29 / 25 % | −0.5 / −0.4° |
| 35 | 945 k | 1.36 / 0.72 | 21.4 / 19.9 % | 31 / 31 % | 0.292 / 0.143 | 11.3 / 10.3 % | 0.081 / 0.043 | 27 / 22 % | −0.7 / −0.8° |
| 36 | 972 k | 0.69 / 0.35 | 17.5 / 17.9 % | 24 / 25 % | 0.122 / 0.063 | 12.5 / 7.4 % | 0.037 / 0.017 | 24 / 23 % | **−2.7 / −2.7°** |
| 37 | 999 k | 1.31 / 0.60 | 24.5 / 23.1 % | 35 / 34 % | 0.320 / 0.140 | 5.3 / 3.4 % | 0.069 / 0.029 | 31 / 24 % | −0.7 / −0.8° |

## 514 (same economy, without target 0.01)

| save | shots/s | hit/shell | h<5° | touches/tank-s | %disarmed | p_off |
|---|---|---|---|---|---|---|
| 9 | 1.42 / 0.95 | 16.0 / 11.8 % | 39 / 34 % | 0.227 / 0.112 | 22 / 31 % | 0.19 / 0.13 |
| 14 | 1.46 / 0.73 | 18.4 / 15.9 % | 31 / 30 % | 0.268 / 0.117 | 13 / 11 % | 0.11 / 0.05 |
| 20-23 | 1.37-1.73 / 0.66-1.07 | 16-22 / 14-20 % | 28-34 % | 0.27-0.33 / 0.12-0.17 | 12-27 / 10-23 % | 0.10-0.21 / 0.04-0.11 |
| 34 | 1.40 / 0.82 | 20.0 / 18.8 % | 30 / 32 % | 0.280 / 0.154 | 13 / 11 % | 0.10 / 0.06 |

## Reading

1. Three phases for 520's trigger: save_9 total dump (p_off 0.6, 42-64 % disarmed), worse than 514 save_9; save_16 trigger nearly extinct (0.13-0.29 shot/s, touches/tank-s ÷3 vs 514); save_20 → save_37 monotone cadence recovery (×4.5 at 550). Without active regulator the trigger swings from one extreme to the other before settling. CSV confirms: fire 1.4 → 0.26/s between 250 and 450 k, then recovery.
2. Aim flat since save_16 (tour<5° 27-33 % / 22-28 %), as 510 and 514. No save moves the frontier.
3. h<5° drops as cadence rises (44 → 35 %): the turret does not degrade, the trigger lowers its threshold, firing more often inside the 5° cone at less perfect alignments. Hit/shell follows (30 → 24 %), disarmed steps rise (1 → 5-11 %).
4. 520's touches per tank-second rejoin 514's: 0.32/0.14 at save_37 vs 0.27-0.33/0.12-0.17; with fewer shots (1.31/0.60 vs 1.4/0.8) and fewer disarmed steps (5/3 % vs 11-13 %).
5. save_36 = isolated site relapse (−2.7°, rest of distribution unchanged), as 514 save_21 (−1.5°, disarmed 23-27 %).

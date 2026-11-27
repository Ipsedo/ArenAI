# Probe train_515_ppo_liquid / save_9

- Index 243 k. Run: 514's config with only gamma 0.998 (uncommitted), effective commit d1bb4020.
- Binary: `build/` 01:13, same as for save_4 and 514, launched in `--force` (see save_4.md).
- Sides 550 and 1000, 4 episodes, stochastic, economy 10/3/30. Landmarks 510/513/514 save_9 probed the same day with the same binary.

## Summary

| run | side | p_arm | ratio | n al&near | hit/sh | enrich | empty_s | %empty | %disarmed | shots/s | turret<5° | σ cx/cy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 510 | 550 | 0.10 | 5.95 | 593 | 17.0 % | 1.83 | 9.5 | 80 % | 36 % | 0.55 | 24.2 % | 0.55/0.49 |
| 513 | 550 | 0.33 | 3.10 | 543 | 14.9 % | 1.87 | 6.0 | 82 % | 28 % | 1.59 | 21.8 % | 0.48/0.40 |
| 514 | 550 | 0.26 | 2.88 | 716 | 16.0 % | 1.76 | 8.8 | 71 % | 22 % | 1.42 | 19.9 % | 0.50/0.41 |
| **515** | 550 | 0.18 | 3.19 | 767 | **17.3 %** | **2.14** | 7.6 | **63 %** | **17 %** | 1.14 | 18.3 % | 0.44/0.40 |
| 510 | 1000 | 0.05 | 10.9 | 227 | 8.8 % | 1.68 | 14.6 | 91 % | 30 % | 0.33 | 19.1 % | 0.60/0.53 |
| 513 | 1000 | 0.21 | 3.66 | 409 | 10.1 % | 1.93 | 7.6 | 87 % | 36 % | 1.03 | 15.0 % | 0.54/0.45 |
| 514 | 1000 | 0.19 | 4.38 | 387 | 11.8 % | 2.24 | 8.7 | 84 % | 31 % | 0.95 | 13.5 % | 0.55/0.45 |
| **515** | 1000 | 0.12 | 4.03 | 717 | **12.5 %** | 1.97 | **13.9** | 87 % | **26 %** | 0.74 | 14.0 % | 0.47/0.42 |

p_fire by angle (550): 0.44 / 0.32 / 0.20 / 0.13 / 0.13 / 0.12, from 0-2° to > 45°. Deterministic: trigger fires on 32 % of aligned decisions and 6 % off-axis (514: 45/9 %, 510: 16/3 %).

## Reading

- **Trigger**: selective, with a ratio of 3.2/4.0 at 513/514's level, far from 510 (6/11). It is however lower everywhere (p_arm 0.18/0.12 vs 0.26/0.19 for 514), placing 515 between 514 and 510 on level without gaining selectivity.
- **Dump**: least pronounced of the 10/3/30 runs at same index. At 1000, reserve empty at 13.9 s (514: 8.7 s, 510: 14.6 s) and the tank is disarmed 17/26 % of the time, lowest of the four runs. At 550 only 63 % of tank-episodes empty their reserve.
- **Fire**: hit/shell 17.3/12.5 %, best of the four at same index, and enrichment 2.1/2.0.
- **Turret**: 18/14 % ≡ 514. Lag seen at save_4 disappeared, but turret does not progress either.
- **CSV around 243-300 k**: hit 16-20 % and fire 0.7-0.9/s, on 510's trajectory (20-23 %, 0.7-0.8) and above 513/514 (14-16 %, 0.9-1.3).

## Established / noise

- Established on this save: 515 dumps less than 514 and disarms less, with the best hit/shell of the four runs at 243 k.
- Not established: an own effect of gamma. One seed only, and the gap is of the same order as 510 vs 513/514, which share the same trigger config. Selectivity (ratio) is not better.

## Useful next

- save_16 and save_23-28 vs 514 save_20-23 and 510 save_28 (D=1000): does the lead in reserve and hit/shell hold in cruise?
- If it holds, a 2nd seed at gamma 0.998 to separate it from noise.

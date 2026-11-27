# train_517_ppo_liquid

| | |
|---|---|
| machine | ElMatador |
| user | samuel |
| launch | 2026-11-14 17:27:09 |
| commit at launch | `e554371e` — update readme |
| commits ≤ 90 min after | none |
| diff at launch | unknown (identity recorded after the run ended) |
| raw data | `outputs/train_517_ppo_liquid` on ElMatador |

## Established facts

- 2026-11-14: stopped 18:45 at save_1 (37 k): hit 6.4 % at 25-37 k vs 8.2 (516) and 9.6-12.2 (510-515), kill 13 vs 16-21, Hd 0.24 (trigger ≈ 0.93), kl 0.109. Regression grows with network size, cause not isolated (4 changes vs 515). Config see 516 revision (tail block 128→256, LTC 256, sensors 128).

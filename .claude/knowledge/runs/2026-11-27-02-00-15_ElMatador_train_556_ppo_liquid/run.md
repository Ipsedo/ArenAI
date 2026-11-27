# train_556_ppo_liquid

| | |
|---|---|
| machine | ElMatador |
| user | samuel |
| launch | 2026-11-27 02:00:15 |
| commit at launch | `5367a45e` — docs(harness): skip commit hooks once final-check passed |
| commits ≤ 90 min after | none |
| diff at launch | none (clean tree, run in progress) |
| raw data | `outputs/train_556_ppo_liquid` on ElMatador |

## Established facts

- 2026-11-27, 568 k: D 1000 at 461 k (554: 950 max, 555: stuck 850), kill 21 vs 18 over 430-546 k, but hit 0.17 vs 0.27-0.29 and fire ×2.
- 2026-11-27, matches s17 and s21 (`matches/save_21.md`): below 554 s17 / 555 s17 in game — dry at 10 s 0.27 / 0.48 (0.03-0.05), hit/shell 0.18-0.26 (0.32-0.39), armed gun < −5° 70 % of the time; 553 s36 profile. One seed, confounded.
- `mask_loss` 0.28 → 0.18-0.21, no clear plateau.

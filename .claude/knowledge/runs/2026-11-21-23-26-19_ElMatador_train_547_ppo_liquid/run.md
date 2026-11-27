# train_547_ppo_liquid

| | |
|---|---|
| machine | ElMatador |
| user | samuel |
| launch | 2026-11-21 23:26:19 |
| commit at launch | `10211f76` — add trained model with new phi |
| commits ≤ 90 min after | none |
| diff at launch | unknown (identity recorded after the run ended) |
| raw data | `outputs/train_547_ppo_liquid` on ElMatador |

## Established facts

- 2026-11-21/22: run 547 (cmake-build-release, tree of 10211f76 + `shells_per_hit` 5 as default), twin of 546, ran in parallel with control 548 in the night 21→22/11. Up to D 900 (366 k): fire 1.4-1.6 vs 0.7-1.1, kill +1 to +2, hit lower (0.13-0.16 vs 0.16-0.19). Probe s13 side 550: off-axis fire ×2 (p_off 0.23 vs 0.11), dump identical (one emptying per 13 vs 15 tank-s), touches/tank-s 0.31 vs 0.23. D 1000: fire 0.99 (400-450 k) → 0.90 → 0.56 → 0.45 → 0.33 (650-700 k); 546 was at 0.42 from 450 k. kill 26 → 20, rejoins 546 (22). Probe s24 side 550: aligned p_fire < 2° 0.114 (546: 0.186), p > 0.5 on 2.6 % of aligned (546: 12 %), shots/tank-s 0.36 vs 0.59, touches/tank-s 0.12 vs 0.20, hit/shell 33 % in both, turret < 5° 31 % in both, one emptying per 210 tank-s. Why: 5 shells per touch delays the closing by ~150 k steps without preventing it; same final state. Apply: do not expect per-touch economy to settle the closing at D 1000. See run 546 and run 548 (cause-source-localisation notes). Turret σ canon 0.033-0.038 early in run (sigma-canon-effondre-debut-de-run: σ canon 0.02-0.04 until 300 k, KL brakes μ, fine GAE).

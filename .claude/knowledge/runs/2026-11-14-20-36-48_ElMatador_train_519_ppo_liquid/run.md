# train_519_ppo_liquid

| | |
|---|---|
| machine | ElMatador |
| user | samuel |
| launch | 2026-11-14 20:36:48 |
| commit at launch | `e554371e` — update readme |
| commits ≤ 90 min after | none |
| diff at launch | unknown (identity recorded after the run ended) |
| raw data | `outputs/train_519_ppo_liquid` on ElMatador |

## Established facts

- 2026-11-14 22:37 (at 102 k): config = 128×256, 3-block CNN (see 518). Throughput ~55 k/h (save every 29-30 min vs 19 for 515); misses the 1st curriculum stage (21 k, hit 7.7 %) ⟹ one D step late (350 vs 400 at 100 k; stages at 48/67 k instead of 21/48/67); CSV hit 11-12 % ≈ 515 −1 pt despite shorter D, kill ≡, ev 0.4-0.47 (corrected computation).
- 2026-11-14 evening: 519 abandoned, 515 config restored (PPO headers = d1bb4020, CNN 16→…→128 4 blocks, liquid micro-batch 512, gamma 0.998 ppo + liquid). Probes 513 s17 / 514 s34 stochastic vs deterministic: removing σ gains only 1-4 pts aligned hit ⟹ μ itself imprecise, the CNN is not implicated.

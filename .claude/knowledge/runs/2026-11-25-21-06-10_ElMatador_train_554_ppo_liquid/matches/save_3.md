# 554 save_3: game-rules match

Run 554 (panorama), save_3 ≈ 81 k, D 400. Binary `build/arenai_agent/arenai_agent_match` rebuilt at `b6376c46` (inference code = `b6f61755`). 8 matches, seeds 0-7, sides 1000 / 500. Reference at equal index: 550 save_3 (550's `match_bin`); current reference 552 s41.

| measure (side 1000 / 500) | 554 s3 | 550 s3 | 550 s10 | 552 s41 |
|---|---|---|---|---|
| living tanks dry at 10 s | 0.50 / 0.56 | 0.88 / 0.87 | **0.03** / **0.07** | 0.06 / 0.08 |
| living tanks dry at 30 s | 0.64 / 0.74 | 0.82 / 0.91 | **0.09** / **0.12** | 0.16 / 0.33 |
| armed within 5° of a target | 0.08 / 0.13 | **0.18** / **0.26** | 0.13 / 0.13 | 0.12 / 0.18 |
| touches / tank | 0.87 / 2.39 | 0.56 / 2.00 | 2.95 / 5.24 | **4.79** / **9.28** |
| kills / tank | 0.07 / 0.19 | 0.05 / 0.15 | 0.24 / 0.42 | **0.27** / **0.57** |
| touch / shell | 0.08 / 0.17 | 0.05 / 0.14 | **0.33** / 0.34 | **0.33** / **0.39** |
| player killed | 0.12 / 0.25 | 0.12 / 0.62 | 0.38 / 0.75 | **0.75** / **0.88** |
| armed, canon < −5° | 0.57 / 0.45 | 0.42 / **0.25** | **0.37** / 0.39 | 0.45 / 0.38 |

- Worse than the reference everywhere: reserve emptied (half the tanks dry at 10 s), 88 % of tanks with no touch on side 1000.
- Vs 550 s3 (same index, one seed): less dump (0.50 vs 0.88, gap > sd), touches/kills within noise, aligned < 5° time lower (0.08 vs 0.18, sd 0.05-0.09), canon more often low.
- Verdict prediction: bad in game at this stage, like 550 at same index. One save: no trajectory. Reference 553 s36 (bad, dry at 10 s 0.30/0.48): the instrument ranks 554 s3 even lower, coherent.

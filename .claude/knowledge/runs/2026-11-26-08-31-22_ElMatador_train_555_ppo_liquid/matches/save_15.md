# 555 save_15: game-rules match (2026-11-26)

Run 555 (panorama, mean aggregation, `a3e0b27d`). save_15 ≈ 405 k, D 850. Binary `build/arenai_agent/arenai_agent_match` (agent library of 11-26 08:25, mean). 8 matches, seeds 0-7, sides 1000 / 500. Reference at equal index: 554 save_15 (≈ 405 k, D 950, max aggregation, `outputs/train_554_ppo_liquid/match_bin`). One seed per run.

| measure (side 1000 / 500) | 555 s15 | 554 s15 | 554 s17 | 554 s10 |
|---|---|---|---|---|
| living tanks dry at 10 s | 0.14 / 0.18 | 0.05 / 0.07 | **0.04** / **0.05** | 0.14 / 0.20 |
| living tanks dry at 30 s | 0.25 / 0.34 | **0.13** / **0.26** | 0.19 / 0.32 | 0.24 / 0.40 |
| living time dry | 0.40 / 0.55 | **0.29** / **0.46** | 0.43 / 0.60 | **0.29** / **0.46** |
| armed within 5° of a target | 0.09 / 0.12 | 0.11 / 0.11 | **0.13** / **0.15** | **0.13** / 0.09 |
| touches / tank | 1.53 / 3.95 | 4.14 / 6.62 | **5.07** / **7.62** | 2.10 / 4.97 |
| kills / tank | 0.08 / 0.20 | 0.26 / 0.45 | **0.29** / **0.46** | 0.17 / 0.37 |
| shells fired / tank | 9.2 / 15.2 | 12.9 / 18.5 | 15.5 / 20.9 | 8.0 / 14.6 |
| touch / shell | 0.16 / 0.25 | **0.33** / **0.36** | 0.32 / **0.36** | 0.26 / 0.34 |
| player killed | 0.50 / 0.50 | 0.50 / **0.75** | 0.25 / **0.75** | **0.62** / 0.38 |
| tanks with no touch | 0.73 / 0.45 | 0.60 / 0.44 | **0.52** / **0.30** | 0.80 / 0.51 |
| armed, canon < −5° | 0.54 / 0.50 | 0.53 / 0.58 | **0.25** / **0.32** | 0.64 / 0.64 |

Worse than the reference:
- Stays dangerous: dry at 10 s 0.14 / 0.18 vs 0.05 / 0.07 (gap > sd on both columns at side 500, borderline side 1000); dry at 30 s 0.25 / 0.34 vs 0.13 / 0.26.
- Touches and kills: touches/tank 1.53 ± 0.72 / 3.95 ± 1.30 vs 4.14 ± 0.62 / 6.62 ± 1.41; kills/tank 0.08 / 0.20 vs 0.26 / 0.45; touch/shell 0.16 / 0.25 vs 0.33 / 0.36 — all beyond sd.
- Finds: ≡ (0.09 / 0.12 vs 0.11 / 0.11); 1st target < 5° later on side 1000 (3.1 s vs 2.2 s). Holding: canon low ≡ 554 s15 (0.54 / 0.50 vs 0.53 / 0.58).
555 s15 fires less and half as accurately as 554 s15: 554 s10's level (270 k) or below.
Verdict prediction: bad, clearly under 554 s15 and 554 s17 (themselves under reference 552 s41). One save, one seed: the 555 / 554 gap is not attributable to mean aggregation.
References: only reference 553 s36 "bad" (reserve emptied, dry at 10 s 0.30 / 0.48). 555 s15 empties less fast than 553 s36 but touches 2 to 3 times less: reading coherent with a bad verdict, instrument not contradicted.
CSV at match time: at 432 k: D 850, kill 13.9 (entered at 17.4). At 476 k: D 850 still, kill 17.9, hit 0.31 — recovery after save_15, to replay on save_17.

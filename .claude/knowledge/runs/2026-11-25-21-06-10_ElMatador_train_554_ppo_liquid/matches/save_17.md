# 554 save_10 / save_17: game-rules match (2026-11-26)

Run 554 (panorama). save_10 ≈ 270 k (D 700), save_17 ≈ 459 k (D 950). Binary `build/arenai_agent/arenai_agent_match` of 11-25 23:12 (`b6f61755` inference side). 8 matches, seeds 0-7. References: 550 s10 (270 k), 550 s26 (702 k, D 900), reference 552 s41. One seed per run.

| measure (side 1000 / 500) | 554 s10 | 554 s17 | 550 s10 | 550 s26 | 552 s41 |
|---|---|---|---|---|---|
| living tanks dry at 10 s | 0.14 / 0.20 | 0.04 / **0.05** | **0.03** / 0.07 | 0.05 / 0.06 | 0.06 / 0.08 |
| living tanks dry at 30 s | 0.24 / 0.40 | 0.19 / 0.32 | **0.09** / **0.12** | 0.14 / 0.28 | 0.16 / 0.33 |
| living time dry | 0.29 / 0.46 | 0.43 / 0.60 | **0.21** / **0.36** | 0.31 / 0.53 | 0.26 / 0.59 |
| armed within 5° of a target | 0.13 / 0.09 | 0.13 / 0.15 | 0.13 / 0.13 | 0.11 / 0.11 | 0.12 / **0.18** |
| touches / tank | 2.10 / 4.97 | 5.07 / 7.62 | 2.95 / 5.24 | **5.54** / 8.79 | 4.79 / **9.28** |
| kills / tank | 0.17 / 0.37 | **0.29** / 0.46 | 0.24 / 0.42 | **0.29** / 0.53 | 0.27 / **0.57** |
| touch / shell | 0.26 / 0.34 | 0.32 / 0.36 | 0.33 / 0.34 | **0.34** / 0.38 | 0.33 / **0.39** |
| player killed | 0.62 / 0.38 | 0.25 / 0.75 | 0.38 / 0.75 | 0.50 / **0.88** | **0.75** / **0.88** |
| tanks with no touch | 0.80 / 0.51 | 0.52 / 0.30 | 0.70 / 0.52 | **0.42** / 0.29 | 0.47 / **0.12** |
| armed, canon < −5° | 0.64 / 0.64 | **0.25** / **0.32** | 0.37 / 0.39 | 0.42 / 0.42 | 0.45 / 0.38 |

Contact on side 1000 (`entonnoir.py`, `approche.py`, `.venv/bin/python3`):

| | 554 s17 | 550 s26 | 550 s28 | 552 s41 |
|---|---|---|---|---|
| armed time within 100 m | 0.19 | 0.16 | 0.12 | 0.03 |
| cos velocity/target, all bands | −0.04 to +0.02 | −0.03 to +0.06 | −0.08 to +0.04 | ≈ 0 |
| closing at > 100 m (m/s) | +0.12 | −0.20 | −1.08 | +0.02 |
| touches/tank-s at 60-100 m | 0.019 | 0.064 | 0.068 | **0.454** |
| distance to center at 290 s (m) | 412 | 873 | 1580 | — |

- Worse first: s17 spends more time dry than 550 s26 and 552 s41 (0.43 vs 0.31 / 0.26, within sd 0.11), kills the player less on side 1000 (0.25), and stays under 552 s41 on side 500 (touches 7.6 vs 9.3, tanks with no touch 0.30 vs 0.12) and at 60-100 m (÷20 on touches/tank-s).
- 554 s17 ≡ 550 s26 on all criteria (gaps < sd), reached 240 k earlier.
- No approach: cos ≈ 0 in all bands; contact time < 100 m (0.19) is 550's. Only difference: tanks no longer flee the arena (stable distance to center).
- s10: under 550 s10 (canon low 64 %, almost immobile 37 % of armed time, touch/shell 0.26).
- Prediction: neither better nor worse than 550 in game, under 552 s41. Reference 553 s36 (bad, dry at 10 s 0.30/0.48) ranked under s17: coherent.

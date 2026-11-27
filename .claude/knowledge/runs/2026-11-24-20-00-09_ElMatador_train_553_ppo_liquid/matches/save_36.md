# Game-rules match — 553 save_36 vs 552 save_41 (2026-11-25)

553 save_36 = 972 k, D 1000; binary `build/` at commit `95f1812c`. 552 save_41 = 1.11 M, D 1000; binary `outputs/train_552_ppo_liquid/match_bin` compiled at `db9a825a`. Game rules: deterministic agent loaded as in the game, 16 tanks, game economy (10 shells, +2 per touch, max 30), no refill when dry, player tank immobile at its spawn, 300 s at most. 8 matches per side, seeds 0-7 (same spawns for both). Mean ± std across matches. Reference: Samuel judges 553 save_36 bad in game (reserve empty very fast). 552 not judged.

| side 1000 / 500 | 552 s41 | 553 s36 |
|---|---|---|
| living tanks dry at 10 s | **0.06 ± 0.07 / 0.08 ± 0.06** | 0.30 ± 0.11 / 0.48 ± 0.14 |
| living tanks dry at 30 s | **0.16 ± 0.10 / 0.33 ± 0.13** | 0.40 ± 0.10 / 0.72 ± 0.13 |
| living tanks dry at 60 s | **0.21 ± 0.12 / 0.51 ± 0.08** | 0.46 ± 0.11 / 0.70 ± 0.10 |
| living time spent dry | **0.26 ± 0.07 / 0.59 ± 0.05** | 0.48 ± 0.14 / 0.67 ± 0.12 |
| matches ended "all dry" | **0 / 1 of 8** | 0 / 5 of 8 |
| armed, canon within 5° of a target | 0.12 ± 0.02 / 0.18 ± 0.03 | **0.33 ± 0.09 / 0.32 ± 0.11** |
| player killed (share of matches) | 0.75 / **0.88** | 0.75 / 0.38 |
| impacts on player | 12.3 ± 8.4 / **14.4 ± 4.4** | **13.6 ± 9.8** / 10.4 ± 10.4 |
| touches / tank | 4.8 ± 2.0 / **9.3 ± 0.9** | **5.1 ± 1.5** / 8.3 ± 1.0 |
| kills / tank | 0.27 ± 0.13 / **0.57 ± 0.06** | **0.28 ± 0.07** / 0.47 ± 0.06 |
| touch / shell | **0.33 ± 0.08 / 0.39 ± 0.02** | 0.30 ± 0.05 / 0.35 ± 0.03 |
| tanks with no touch at all | **0.47 ± 0.13 / 0.12 ± 0.07** | 0.51 ± 0.15 / 0.27 ± 0.09 |
| canon elevation armed (°) | −4.7 ± 0.6 / −4.1 ± 0.4 | −0.0 ± 1.2 / −5.0 ± 11.0 |
| canon elevation dry (°) | −4.5 ± 0.9 / −4.6 ± 0.6 | −1.3 ± 1.7 / −1.0 ± 1.5 |

Reading:
- Worse than 552: 553 empties its reserve. At 10 s, 30 to 48 % of living tanks are dry, vs 6 to 8 %; on side 500, 5 matches of 8 stop because no tank has shells left. The gap far exceeds the dispersion between matches. This is what Samuel saw in game: the instrument ranks the reference in the right direction.
- Touches and kills: equal on side 1000, 552 ahead on side 500 (kills 0.57 vs 0.47, more than one std). 553's lead read in training (kill 22.7 vs 20.0) does not exist in game.
- 553 holds its canon on a target more often (0.33 vs 0.12-0.18) and has the canon horizontal; 552 keeps the canon low (−4.5°). This aim gain does not translate into touches.
- "Canon pointing down once dry" not reproduced: median dry elevation −1°.
- One training seed per run: the 552 → 553 gap cannot be attributed with certainty to spatial aim.
Prediction of the game verdict: 552 save_41 should appear better than 553 save_36 (it keeps its shells), without being good: on side 500, half of living tanks are dry at 60 s. To be confirmed by Samuel in game.

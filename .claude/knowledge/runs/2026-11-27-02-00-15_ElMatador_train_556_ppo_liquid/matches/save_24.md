# 556 save_24 — game-rule match, trajectory s17 → s21 → s24 (2026-11-27)

Run 556 (`5367a45e`), save_24 ≈ 648 k, D 1000. Binary `build/arenai_agent/arenai_agent_match` at `5367a45e`.
8 matches, seeds 0-7, sides 1000 / 500. References: 554 save_17 and 555 save_17 (≈ 459 k, not equal index).
One seed per run.

| measure (side 1000 / 500) | 556 s17 | 556 s21 | 556 s24 | 554 s17 | 555 s17 |
|---|---|---|---|---|---|
| alive dry at 10 s | 0.27 / 0.48 | 0.26 / 0.39 | 0.24 / 0.30 | 0.04 / 0.05 | **0.03** / **0.02** |
| alive dry at 30 s | 0.51 / 0.75 | 0.53 / 0.75 | 0.49 / 0.74 | 0.19 / 0.32 | **0.10** / **0.18** |
| time alive dry | 0.74 / 0.76 | 0.75 / 0.75 | 0.73 / 0.71 | 0.43 / 0.60 | **0.27** / **0.47** |
| armed within 5° of a target | 0.05 / 0.12 | 0.07 / 0.20 | 0.09 / **0.21** | 0.13 / 0.15 | **0.14** / 0.16 |
| touches / tank | 2.49 / 3.54 | 2.70 / 4.78 | 3.16 / 5.08 | 5.07 / 7.62 | **5.39** / **9.45** |
| kills / tank | 0.20 / 0.29 | 0.20 / 0.33 | 0.23 / 0.43 | **0.29** / 0.46 | 0.27 / **0.49** |
| hit / shell | 0.18 / 0.22 | 0.19 / 0.26 | 0.22 / 0.28 | 0.32 / 0.36 | **0.35** / **0.39** |
| tanks without a hit | 0.64 / 0.48 | 0.69 / 0.42 | 0.61 / 0.42 | **0.52** / 0.30 | 0.55 / **0.22** |
| player killed | 0.38 / 0.50 | 0.38 / 0.50 | 0.50 / 0.62 | 0.25 / **0.75** | **0.75** / 0.50 |
| armed gun < −5° | 0.71 / 0.72 | 0.70 / 0.52 | 0.49 / 0.43 | **0.25** / 0.32 | 0.33 / **0.26** |
| armed gun elevation, median (°) | −11.3 / −13.8 | −8.9 / −5.1 | −5.0 / −3.8 | −0.8 / −2.0 | −2.4 / −1.3 |

## Worse than the reference

- **Stays dangerous**: dry at 10 s 0.24 ± 0.10 / 0.30 ± 0.12 vs 0.02-0.05; dry at 30 s 0.49 / 0.74 vs 0.10-0.32.
  Unchanged from s17 to s24 at side 1000, far beyond the sd.
- **Hits**: hit/shell 0.22 / 0.28 vs 0.32-0.39; touches/tank 3.2 / 5.1 vs 5.1-9.5.
- **Hold**: armed gun < −5° 0.49 / 0.43 vs 0.25-0.33.

## Trajectory (three saves)

Monotone on the gun and on hits: armed gun < −5° 0.71 → 0.70 → 0.49 (side 1000) and 0.72 → 0.52 → 0.43 (500);
hit/shell 0.18 → 0.19 → 0.22 and 0.22 → 0.26 → 0.28; kills/tank at side 500 0.29 → 0.33 → 0.43 (554 0.46,
within sd). Steps between consecutive saves are about one sd. The early dump does not follow: dry at 10 s
0.27 → 0.26 → 0.24 at side 1000.

## Reading

The gun is rising in game as in the probe timeline, and hits follow. The reserve emptied in the first 10 s is
not carried by the gun elevation: it stays while the gun rises. The oracle-pitch result (first empty reserve
18 → 35 s, train rules, stochastic) does not transfer to the first 10 s of a game match.

## Predicted game verdict

Still bad in game, below 554 s17 and 555 s17: reserve emptied early. Improving slowly on hits.

## Labels

553 s36 « pourri en jeu » (dry at 10 s 0.30 / 0.48): 556 s24 at 0.24 / 0.30, same class at side 1000; consistent.

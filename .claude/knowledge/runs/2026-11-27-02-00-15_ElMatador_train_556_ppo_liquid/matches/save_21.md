# 556 save_17 and save_21 — match under game rules (2026-11-27)

Run 556 (panorama, shared encoder trained supervised on the enemy visibility mask, `5367a45e`, clean tree).
save_17 ≈ 459 k (D 950 → 1000), save_21 ≈ 567 k (D 1000). Binary `build/arenai_agent/arenai_agent_match` rebuilt
at `5367a45e` on 2026-11-27 08:53. 8 matches, seeds 0-7, sides 1000 / 500. Reference at equal index: 554 save_17
and 555 save_17 (≈ 459 k). One seed per run.

| measure (side 1000 / 500) | 556 s17 | 556 s21 | 554 s17 | 555 s17 |
|---|---|---|---|---|
| alive dry at 10 s | 0.27 / 0.48 | 0.26 / 0.39 | 0.04 / 0.05 | **0.03** / **0.02** |
| alive dry at 30 s | 0.51 / 0.75 | 0.53 / 0.75 | 0.19 / 0.32 | **0.10** / **0.18** |
| time alive dry | 0.74 / 0.76 | 0.75 / 0.75 | 0.43 / 0.60 | **0.27** / **0.47** |
| matches ended all dry | 0.62 / 0.75 | 0.25 / 0.88 | **0.00** / 0.12 | **0.00** / **0.00** |
| armed within < 5° of a target | 0.05 / 0.12 | 0.07 / **0.20** | 0.13 / 0.15 | **0.14** / 0.16 |
| 1st target < 5° (s) | 7.5 / 4.3 | 3.0 / 2.1 | **1.7** / 2.1 | 2.0 / **1.9** |
| hits / tank | 2.49 / 3.54 | 2.70 / 4.78 | 5.07 / 7.62 | **5.39** / **9.45** |
| kills / tank | 0.20 / 0.29 | 0.20 / 0.33 | **0.29** / 0.46 | 0.27 / **0.49** |
| shells fired / tank | 13.8 / 15.7 | 14.1 / 18.2 | 15.5 / 20.9 | 15.2 / 23.8 |
| hit / shell | 0.18 / 0.22 | 0.19 / 0.26 | 0.32 / 0.36 | **0.35** / **0.39** |
| tanks without a hit | 0.64 / 0.48 | 0.69 / 0.42 | **0.52** / 0.30 | 0.55 / **0.22** |
| player killed | 0.38 / 0.50 | 0.38 / 0.50 | 0.25 / **0.75** | **0.75** / 0.50 |
| armed, gun < −5° | 0.71 / 0.72 | 0.70 / 0.52 | **0.25** / 0.32 | 0.33 / **0.26** |

## Worse than reference

- **Stays dangerous**: dry at 10 s 0.27 ± 0.13 / 0.48 ± 0.14 vs 0.03-0.05; dry at 30 s 0.51 / 0.75 vs
  0.10-0.32; time alive dry 0.74-0.76 vs 0.27-0.60. Far beyond sd, both saves.
- **Hits**: hit/shell 0.18-0.26 vs 0.32-0.39; hits/tank ÷2; kills/tank 0.20 / 0.29-0.33 vs 0.27-0.29 /
  0.46-0.49 (≈ 1 sd).
- **Finds**: side 1000, armed within < 5° 0.05-0.07 vs 0.13-0.14.
- **Holds**: median armed gun −9 to −14°; armed gun < −5° 70 % of the time (ref 0.25-0.33).

## Better

Nothing beyond sd. save_21 side 500: armed within < 5° 0.20 ± 0.12 (554/555: 0.15-0.16), within sd.

## Trajectory

save_17 → save_21 (+108 k): stable state (dry, hit/shell), slightly better on side 500. Two saves only:
trajectory not established (rule: ≥ 3).

## Game verdict prediction

Bad in game, below 554 s17 and 555 s17: 553 s36 profile (reserve emptied within the first 10 s, gun low). Higher
training kill (21 vs 18 over 430-546 k) comes from refill when dry, as in 553. One seed: the gap cannot be
attributed to the mask without a control.

## Benchmarks

553 s36 « pourri en jeu » (dry at 10 s 0.30 / 0.48): 556 s17 at 0.27 / 0.48 ⟹ same class, consistent reading.

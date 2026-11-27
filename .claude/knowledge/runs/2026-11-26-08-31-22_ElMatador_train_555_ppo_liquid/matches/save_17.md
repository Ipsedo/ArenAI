# 555 save_17: game-rules match (2026-11-26)

Run 555 (panorama, mean aggregation, `a3e0b27d`). save_17 ≈ 459 k, D 850. Binary `build/arenai_agent/arenai_agent_match` (mean). 8 matches, seeds 0-7, sides 1000 / 500. Reference at equal index: 554 save_17 (≈ 459 k, D 950, max); reference 552 s41. One seed per run.

| measure (side 1000 / 500) | 555 s17 | 554 s17 | 555 s15 | 552 s41 |
|---|---|---|---|---|
| living tanks dry at 10 s | **0.03** / **0.02** | 0.04 / 0.05 | 0.14 / 0.18 | 0.06 / 0.08 |
| living tanks dry at 30 s | **0.10** / **0.18** | 0.19 / 0.32 | 0.25 / 0.34 | 0.16 / 0.33 |
| living time dry | 0.27 / **0.47** | 0.43 / 0.60 | 0.40 / 0.55 | **0.26** / 0.59 |
| armed within 5° of a target | **0.14** / 0.16 | 0.13 / 0.15 | 0.09 / 0.12 | 0.12 / **0.18** |
| touches / tank | **5.39** / **9.45** | 5.07 / 7.62 | 1.53 / 3.95 | 4.79 / 9.28 |
| kills / tank | 0.27 / 0.49 | **0.29** / 0.46 | 0.08 / 0.20 | 0.27 / **0.57** |
| shells fired / tank | 15.2 / 23.8 | 15.5 / 20.9 | 9.2 / 15.2 | — |
| touch / shell | **0.35** / **0.39** | 0.32 / 0.36 | 0.16 / 0.25 | 0.33 / **0.39** |
| player killed | **0.75** / 0.50 | 0.25 / 0.75 | 0.50 / 0.50 | **0.75** / **0.88** |
| tanks with no touch | 0.55 / 0.22 | 0.52 / 0.30 | 0.73 / 0.45 | **0.47** / **0.12** |
| armed, canon < −5° | 0.33 / **0.26** | **0.25** / 0.32 | 0.54 / 0.50 | 0.45 / 0.38 |

Worse than the reference: vs 554 s17 nothing beyond sd (canon < −5° 0.33 vs 0.25 on side 1000, within sd); vs 552 s41: tanks with no touch on side 500 (0.22 vs 0.12), kills on side 500 (0.49 vs 0.57, within sd).
Better: stays dangerous: living time dry 0.27 ± 0.09 vs 0.43 ± 0.11 (side 1000), dry at 30 s 0.18 ± 0.04 vs 0.32 ± 0.11 (side 500) — beyond one sd. Touches: side 500 touches/tank 9.45 ± 1.81 vs 7.62 ± 1.73, touch/shell 0.39 ± 0.02 vs 0.36 ± 0.02 (≈ 1 sd); side 1000 ≡. Vs 555 s15 (54 k earlier): touches ×3.5 / ×2.4, kills ×3.4 / ×2.5, dry at 10 s 0.14 → 0.03.
Verdict prediction: ≡ 554 s17 in combat, stays armed longer; reference 552 s41's level. None of these checkpoints is judged good by Samuel. The save_15 trough was transient. One seed: nothing attributable to mean aggregation. References: 553 s36 "bad" (dry at 10 s 0.30 / 0.48): 555 s17 at 0.03 / 0.02, reading not contradicted.

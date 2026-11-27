# Probe 535 save_5: aim no longer progresses

Run 535 (coloured noise ρ 0.9, actor lr 5e-5), commit b83da585. save_5, index 135 k, D 400. Binary `build/arenai_agent/arenai_agent_probe` (`--force`, same dating artefact as 534 s3; library recompiled 2026-11-19 with `kl_breakdown` added, no inference effect). Sides 250 and 550, 4 episodes, stochastic.

| side | ratio | p_al | p_off | hit/sh | h<5° | shots<5° | empty_s | %disarmed | shots/s | tour<5° | σx | \|μ\|cx | μsat |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 250 | 1.04 | 0.66 | 0.64 | 12.3 % | 30 % | 17 % | 4.1 | 45 % | 1.82 | 13.3 % | 0.19 | 0.29 | 17 % |
| 550 | 1.04 | 0.66 | 0.63 | 5.5 % | 15 % | 14 % | 2.6 | 55 % | 1.67 | 11.8 % | 0.19 | 0.24 | 17 % |

| aim (< 100 m, step ≥ 10) | 535 s5 (250 / 550) | 535 s3 (250 / 550) | 534 s3 (250 / 550) |
|---|---|---|---|
| corrective turret μ, 5-40° | 0.59 / 0.60 | 0.61 / 0.61 | 0.62 / 0.64 |
| gain (° command per ° yaw) | −0.017 / −0.017 | −0.018 / −0.018 | −0.031 / −0.027 |
| yaw < 5° | 6.0 / 5.3 % | 5.9 / 5.7 % | 7.0 / 6.4 % |
| yaw < 20° | 19.8 / 16.8 % | 20.1 / 18.3 % | 22.2 / 19.6 % |
| dwell < 5° (s) | 0.38 / 0.41 | 0.36 / 0.37 | 0.29 / 0.36 |

- No progression from s3 to s5: corrective share −1 to −2 points, gain unchanged, time on target unchanged at 250 and falling at 550.
- μ saturated (|μ| > 1.5) 17 % of decisions vs 1.6-1.8 % at s3. Trigger still flat (0.64-0.66), dump 2.6-4.1 s.
- CSV: falling back since 110 k (kill 21.2 → 16.3, hit 0.107 → 0.081 at 171 k, D 450; 528 at same D: 24.1 / 0.156), skip 0.66.

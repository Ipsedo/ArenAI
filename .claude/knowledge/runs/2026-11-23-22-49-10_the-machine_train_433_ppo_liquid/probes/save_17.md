# Probe train_433_ppo_liquid / save_17 — after the curriculum retreat (2026-11-24)

- Index 459 k, D 800 (fell back from 850 at 429 k). Same binary and same settings as `save_14.md` (sides 550 / 1000, 3 episodes, stochastic, oracles `--oracle_fov 79`), GPU shared with training.

| measure | s14 side 550 | s17 side 550 | s14 side 1000 | s17 side 1000 |
|---|---|---|---|---|
| turret<5° | 0.185 | 0.165 | 0.133 | 0.146 |
| armed p_fire, aligned < 5° (any distance) | 0.300 | 0.194 | 0.091 | 0.041 |
| armed p_fire, aligned < 5° and < 40 m | 0.741 | 0.744 | 0.758 | 0.722 |
| armed p_fire, off-axis ≥ 20° | 0.032 | 0.018 | 0.0098 | 0.0042 |
| shells/tank-s | 0.97 | 0.70 | 0.28 | 0.17 |
| hit/shell | 0.161 | 0.205 | 0.148 | 0.180 |
| hit/shell of shots < 5° | 0.263 | 0.359 | 0.246 | 0.352 |
| hits/tank-s | 0.156 | 0.144 | 0.041 | 0.031 |
| kills/episode | 20.3 | 19.3 | 9.0 | 6.7 |
| hits/tank-s under oracle aim (policy trigger) | 0.156 | 0.128 | 0.076 | 0.049 |
| shells/tank-s under oracle aim | 0.36 | 0.25 | 0.16 | 0.09 |
| hits/tank-s full oracle | 0.694 | 0.780 | 0.692 | 0.629 |
| median σ canon_x / canon_y | 0.39 / 0.35 | 0.18 / 0.29 | 0.50 / 0.48 | 0.22 / 0.36 |
| sweep, dominant sign of μ canon_x | 0.84 | 0.89 | 0.95 | 0.97 |

corr(μ canon_x, yaw), side 1000, s14 → s17: 100-200 m, 5-20° −0.26 → −0.32; 200-400 m, 5-20° +0.01 → −0.06 (n 872): within noise.

## Reading

- **Established**: aim neither regressed nor progressed (turret<5°, per-slice correlations).
- **Established**: the trigger closes everywhere except point-blank: aligned p_fire at all distances ÷1.5 (550) and ÷2.2 (1000), unchanged at < 40 m. Hit rises because long-range shots disappear; hits and kills fall in volume.
- **Established**: under perfect aim the trigger fires even less than at save_14 (÷1.4 to ÷1.8): it moves away from 552's behaviour (1.2-1.3 shells/tank-s).
- canon_x σ halved between the two saves.
- Cause of the closure not established.

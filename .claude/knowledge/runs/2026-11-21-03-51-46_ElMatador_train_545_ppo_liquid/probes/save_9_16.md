# Run 545 — probes save_9 and save_16

## Header

save_9: index 243 k, D 650. save_16: index 432 k, D 1000 (reached at 429 k). Binary `build/arenai_agent/arenai_agent_probe` compiled 2026-11-21 03:49. The script dates it on `b83da585` (mtime one minute before commit `d85cc90f`), but it was compiled from the working tree already carrying the run's code: sources modified 03:47, `libarenai_agent.so` 03:48 without coloured noise, `libarenai_model.so` with `get_aim_potential`. Run with `--force`, good compatibility. Sides 550 and 1000, 4 episodes, 32 tanks, stochastic, economy 10 / 2 / 30.

## Summary

| save | side | ratio | p_al | p_off | p_arm | shots/s | hit/sh | h<5° | shots<5° | shots ≥20° | touches/tank-s | 1st empty | disarmed | tour<5° | σx / σy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 9 | 550 | 2.9 | 0.865 | 0.294 | 0.39 | 1.16 | 15.0 % | 43.0 % | 30 % | 40 % | 0.17 | 5.4 s | 39 % | 16.9 % | 0.10 / 0.25 |
| 9 | 1000 | 3.5 | 0.847 | 0.239 | 0.30 | 0.84 | 11.4 % | 39.6 % | 24 % | 45 % | 0.10 | 6.1 s | 54 % | 12.4 % | 0.21 / 0.43 |
| 16 | 550 | 3.4 | 0.204 | 0.060 | 0.089 | 0.78 | 32.8 % | 59.6 % | 51 % | 30 % | 0.26 | 20.2 s | 1.3 % | 31.6 % | 0.13 / 0.16 |
| 16 | 1000 | 11.0 | 0.268 | 0.024 | 0.046 | 0.42 | 27.8 % | 55.2 % | 47 % | 28 % | 0.12 | 33.6 s | 2.7 % | 19.7 % | 0.37 / 0.44 |

(touches/tank-s = shots/s × hit/shell.)

save_9 (D 650): trigger late. Dump intact (empty in 5-6 s, 77-89 % of tank-episodes empty, 39-54 % of steps disarmed), trigger high everywhere (p_off 0.24-0.29), 40-45 % of shells fired at ≥ 20°. Aim is already there when the shot is aligned: 40-43 % touches.
save_16 (D 1000): dump resorbed, record precision of aligned shots.
- Trigger: whole curve fell (p_arm 0.39 → 0.09 side 550) with shape kept side 550 (ratio 2.9 → 3.4) and sharpened side 1000 (3.5 → 11). Disarmed 1-3 %, 1st empty 20-34 s.
- Shot: hit/shell ×2.2-2.4 (33 / 28 %), half the shells leave aligned (51 / 47 %), they hit 60 / 55 %.
- Turret: < 5° 31.6 / 19.7 % of time (16.9 / 12.4 % at save_9), median angle 25° → 15° side 550.
- Remaining waste: 28-30 % of shells leave at ≥ 20° and hit 1.3 %.
- Deterministic: p > 0.5 on 4.4-4.5 % of aligned states only ⟹ almost mute in deterministic game.
- Heads: σ canon bimodal (p10 0.04-0.07, p90 0.48-0.61), wider median side 1000; no saturated μ, no σ at ceiling. Zoom not abandoned (p_zoom 0.53-0.56), decoupled from the trigger (corr 0.99 → 0.40).

## Comparison to references (same side)

| | shots/s | hit/shell | h<5° | touches/tank-s | disarmed | tour<5° |
|---|---|---|---|---|---|---|
| **545 s16** (D 1000) | 0.78 / 0.42 | 33 / 28 % | 60 / 55 % | 0.26 / 0.12 | 1 / 3 % | 32 / 20 % |
| 510 s28 (D 1000) | — | 21 / 19.5 % | — | — | 14-16 % | 29 / 23 % |
| 520 s32 (D 1000) | — | 26 / 27 % | 40 % | 0.276 / 0.142 | — | 28 / 21 % |
| 522 s16 (D 1000) | 0.16 / 0.08 | 46 / 44 % | — | ≈ 0.07 / 0.04 | — | 45 / 37 % |
| 522 s54 (best throughput) | — | 34-40 % | 51-53 % | 0.211 / 0.101 | 0.4-1.6 % | 31 / 23-25 % |
| 522 s68 (precision) | — | 46 / 45 % | 60-62 % | 0.150 / 0.110 | — | — |

| | shots/s | hit/shell | 1st empty | disarmed | tour<5° |
|---|---|---|---|---|---|
| **545 s9** (D 650) | 1.16 / 0.84 | 15 / 11 % | 5.4 / 6.1 s | 39 / 54 % | 17 / 12 % |
| 510 s9 | — | 20.9 / 14.9 % | 9.1 / 19.6 s | — | 20.8 / 15.7 % |
| 522 s9 | 0.45 / 0.33 | 21.8 / 17.4 % | 25 / 37 s | 12 / 11 % | 25 / 20 % |

## Conclusions

Established: at save_9, 545 is behind 510 and 522 at the same D: dump not resorbed, turret less held. At save_16, 545 has 520 s32's touch throughput (0.26 / 0.12 vs 0.276 / 0.142) at higher hit/shell, and exceeds 522 s54 (0.211 / 0.101), which 522 reached only at 1.45 M. Aligned-shot precision (60 / 55 %) equals 522 s68's record (60-62 %), 1.4 M earlier. The trigger is not closed at save_16: 0.78 / 0.42 shots/s, when 522 s16 fired 0.16 / 0.08. Success criterion set 2026-11-20 (p_off < 0.05, p_al ≥ 0.5, ≥ 0.4 shot/s): cadence held, p_off held side 1000 only, p_al not (0.20-0.27).
Noise / not settled: turret 32 / 20 % ≈ 510 s28 and 520 s32, under 522 s16 (45 / 37 %, but closed trigger); the marker "≥ 30 % < 5° with closed trigger" cannot be read here: 545 has no closed phase. A single point at D 1000, 4 episodes, one seed; 513, 514 and 522 oscillated from save to save. The α_canon_x rise (×25) does not translate into σ at ceiling; its effect on precision is not isolable.
Next useful: save_20 and save_24 (540 k, 648 k): touches/tank-s and p_off side 550; save_16 in `--deterministic` (expected very low cadence); watch release: disarmed > 3 %, side-1000 p_off > 0.03, share of shots ≥ 20° rising.

# Probe train_433_ppo_liquid / save_14 — aim (2026-11-24)

- Index 378 k, D 850. Agent `ppo_liquid`, 4-block CNN + spatial softmax (x, y, amplitude per channel → 384) + presence head on the block-4 map.
- Binary: `build/arenai_agent/arenai_agent_probe` recompiled 24/11 13:44 on the working tree (f7435cc2 + staged uncommitted changes = run code, training binary of 23/11 22:49). The previous binary (23/11 20:52) predated the spatial softmax; `--check-only` did not flag it.
- Sides 550 and 1000, 3 episodes, stochastic, GPU shared with the running 433 training.
- Reference: 552 report (other PC, 7-block flattened CNN, save_41, 1.11 M, D 1000). **Not at equal D nor equal maturity**: 552 has 3× more steps.

## Summary

| measure | 433 s14 side 550 | 433 s14 side 1000 | 552 s41 marker |
|---|---|---|---|
| turret<5° | 0.185 | 0.133 | 0.17-0.19 |
| hit/shell | 0.161 | 0.148 | 0.34-0.37 |
| hits/tank-s, policy | 0.156 | 0.041 | 0.275 / 0.119 |
| hits/tank-s, oracle yaw+pitch in view, policy trigger | 0.156 (×1.0) | 0.076 (×1.9) | 0.527 (×1.9) / 0.365 (×3.1) |
| shells/tank-s under oracle aim | 0.36 | 0.16 | 1.2-1.3 |
| hits/tank-s, full oracle in view | 0.694 | 0.692 | 0.766 / 0.756 |
| kills/episode policy → oracle aim → full oracle | 20.3 → 21.7 → 30.7 | 9.0 → 17.7 → 35.0 | 25.3 → 32.0 → 30.7 / 17.7 → 28.3 → 43.7 |
| trigger: aligned&close / off-axis ratio | 22.9 (n 311) | 77.5 (n 72, noise) | — |
| reaction per pass, right pitch, < 60 m / 60-100 m | 0.76 / 0.34 | 0.83 / 0.27 | 0.58-0.91 / 0.56-0.57 |
| gesture credit ρ (GAE λ 0.95) | −0.009 to +0.015 | 0.000 to +0.009 | ≈ 0 |

## Aim by distance (side 1000, step ≥ 10)

corr(μ canon_x, target yaw); towards the target = negative.

| distance | 1-5° | 5-20° | 20-40° | ≥ 40° |
|---|---|---|---|---|
| < 60 m | −0.39 | −0.57 | −0.69 | +0.11 |
| 60-100 m | −0.42 | −0.49 | −0.16 | +0.03 |
| 100-200 m | −0.10 | −0.26 | −0.10 | 0.00 |
| 200-400 m | −0.04 | +0.01 | +0.06 | +0.05 |

Shots: hit/shell 0.43 (< 60 m), 0.18 (60-100 m), 0.075 (100-200 m), 0.08 (> 200 m); median pitch+drop error at fire −2.0 to −2.7° (|.| 2.5-4.4°). Sweep: mean tanh μ canon_x −0.39 / −0.53, dominant sign 84 % / 95 % of decisions.

## Position precision at the policy input (`banc/banc_position.txt`)

Images of save_14, side 1000, forced zoom, 134 k images of which 13.4 k training with enemy visible < 400 m, test 5.6 k (unseen episode), 4 passes, one seed. Yaw: left/right / median error / share < 1°; pitch: median error.

| configuration | 60-100 m | 100-200 m | 200-400 m |
|---|---|---|---|
| 433 frozen (softmax vector received by the LTC) | 0.96 / 0.87° / 55 %; 0.41° | 0.97 / 1.51° / 35 %; 0.68° | 0.84 / 4.18° / 16 %; 1.20° |
| 433 block-4 map frozen, read flattened | 0.92 / 1.79° / 29 %; 0.79° | 0.93 / 3.23° / 18 %; 0.93° | 0.78 / 7.82° / 7 %; 1.48° |
| 433 fully fine-tuned | 0.97 / 0.78° / 61 %; 0.37° | 0.98 / 0.94° / 52 %; 0.43° | 0.89 / 1.75° / 34 %; 0.63° |
| 4 blocks + softmax, fresh weights | 0.94 / 1.81°; 1.05° | 0.91 / 3.27°; 1.31° | 0.67 / 8.78°; 2.07° |
| 7 blocks flattened, fresh weights (reference) | 0.90 / 2.56°; 1.19° | 0.83 / 5.31°; 1.42° | 0.58 / 10.12°; 1.59° |
| always at centre | 6.20° | 12.36° | 11.59° |
| *552 frozen (report, other image set, 25.6 k)* | *1.26°* | *2.19°* | *5.34°* |

The training set is 2× smaller than the report's: the « 7 fresh blocks » reference gives 2.56 / 5.31 / 10.12° here vs 1.04 / 1.04 / 2.64° in the report ⟹ readouts are undertrained and absolute figures are not comparable to 552. Learned temperature ≈ 1 (0.93-1.03): it did not move.

## Reading

- **Established**: the representation conveys position (frozen < 100 m: 0.87°, better than all fresh readouts on the same set); the softmax does not degrade the block-4 map, it reads it better than a flatten at equal data.
- **Established**: the policy does not use it beyond 100 m (corr ≈ 0 at 200-400 m, −0.1 to −0.26 at 100-200 m), turret<5° equal to 552, gesture credit null as in 552.
- **Established**: the trigger is selective but does not follow perfect aim (0.16-0.36 shells/tank-s under oracle vs 1.2-1.3 in 552).
- **Undecidable**: 433 vs 552 at equal D (552 is not on this PC, 433 is not at D 1000).

## Next

- Re-probe at D 1000 (same commands, `oracle/` + `banc/`) for the equal-D comparison.
- `banc/s14_zoom.frames` (13 GB) kept to replay the bench; deletable.

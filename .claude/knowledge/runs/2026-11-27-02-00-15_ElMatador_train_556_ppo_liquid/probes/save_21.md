# 556 save_21 — probe (2026-11-27)

Run 556 (`5367a45e`), save_21, index 567 k, D 1000. Binary `build/arenai_agent/arenai_agent_probe` rebuilt at
`5367a45e` (09:30), no inference diff. Sides 550 and 1000, 4 episodes, 32 tanks, stochastic and deterministic,
run economy 10 / +2 per hit / max 30. GPU shared with the running training. Train rules (refill when dry).

Question: does the in-game dump (matches/save_21.md) come from the trigger or from the low gun?
Rule written before reading: trigger ⟹ most shells leave off-axis or out of range whatever the pitch error;
low gun ⟹ shells leave yaw-aligned with a large pitch error.

| side / mode | ratio | p_al | p_off | hit/sh | h<5° | shots<5° | empty_s | %emptied | turret<5° | σx | σy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 550 sto | 9.6 | 0.68 | 0.071 | 0.222 | 0.467 | 0.366 | 12.3 | 0.68 | 0.148 | 0.35 | 0.17 |
| 1000 sto | 29.2 | 0.70 | 0.024 | 0.185 | 0.373 | 0.366 | 17.9 | 0.68 | 0.095 | 0.39 | 0.18 |
| 550 det | 6.7 | 0.56 | 0.083 | 0.208 | 0.423 | 0.392 | 10.2 | 0.56 | 0.183 | 0.36 | 0.17 |
| 1000 det | 17.6 | 0.60 | 0.034 | 0.206 | 0.408 | 0.423 | 21.0 | 0.70 | 0.116 | 0.39 | 0.18 |

Counts aligned & close: 260-508 (ratio at the noise limit; the angle-only ratio, n 7-12 k, is 4-11).

## Reading (deterministic, side 550 / 1000)

- **Trigger**: selective in angle (p_fire 0.40-0.42 at < 2°, 0.03-0.09 at ≥ 20°), not flat, not mute. But the
  off-axis population is 4-5× the aligned one: 58-61 % of shells leave at ≥ 5°, hit/shell 0.09-0.12 at 5-20°
  and 0.005 at ≥ 20°.
- **Range**: 41 / 48 % of shells are fired with the best-aligned target > 200 m, hit/shell 0.032 / 0.023, only 18-19 % of
  them within 5°. Under 100 m (40 % of shells) hit/shell is 0.39-0.47. The reserve is spent on far targets.
- **Dump**: first empty reserve median 10.2 / 21.0 s, p10 4.1 / 4.8 s, 56-70 % of tank-episodes empty it.
  Shots are not concentrated at spawn (22 / 11 % in the first 10 s): steady leak, not a spawn burst.
- **Gun elevation**: armed median −6.9 / −8.7°, 60-68 % of armed time below −5° (matches the game matches).
  At fire time, among yaw-aligned shells (30-38 % of shells), pitch error median +2.0 / +1.5°;
  hit/shell 0.63 / 0.58 within ±2°, 0.38 / 0.32 at 2-5°, 0.25 / 0.21 beyond 5° (17-20 % of them).
- **Aim**: turret < 5° 0.12-0.18; canon_x mean μ −0.34 / −0.54 (constant sweep), σx 0.36-0.39.

## Verdict

Read alone, this save pointed at the trigger (range and off-axis volume). The comparison with 554 s17 and
555 s17 on the same decomposition (`comparisons/2026-11-27_shots-by-range-554-555-556.md`) revises it:

- The share of shells fired with the best-aligned target beyond 200 m is the same in the three runs
  (0.53 / 0.51 / 0.55): not specific to 556, not the cause of its dump.
- Specific to 556: gun elevation at fire time on close targets. Under 100 m, yaw-aligned shells (|yaw| < 3°)
  carry a pitch error > +2° in 57 % of cases (554 38 %, 555 24 %) and hit 0.36 there (554 0.69, 555 0.63).
  Close-range hit/shell 0.42 vs 0.58-0.56; this gap alone accounts for the overall hit/shell gap to 554.
- Also specific: shot volume 0.46 shots/tank-s (554 0.37, 555 0.21), with fewer refills per shell.

`fire_best_distance` is the distance of the best-aligned target, not of the nearest one.

## Next evaluators

- `probe-timeline` on 556 saves 9, 13, 17, 21: when the positive pitch error at fire time appears.
- Why the gun rests low: canon_y μ vs signed pitch error per error bucket (aim learning sign), 556 vs 555.

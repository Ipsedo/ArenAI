# 556 — where the first-10-seconds dump comes from in game matches (2026-11-27)

Match exports (`<run>/matches/save_K_side<S>_tanks.csv`, 8 matches, 16 tanks, deterministic agent), first 10 s
of each match; shells counted from the per-tank `fired` counter, context = best-aligned enemy at the shot.
556 s21 / s24 vs 554 s17 / 555 s17. One seed per run, unequal index.

| shells / tank in the first 10 s (side 1000 / 500) | 554 s17 | 555 s17 | 556 s21 | 556 s24 |
|---|---|---|---|---|
| total | **2.34** / 5.58 | 2.89 / **4.87** | 5.69 / 9.96 | 5.05 / 9.26 |
| aligned < 5°, target < 100 m | 0.69 / 1.90 | 0.51 / 1.54 | 1.03 / 3.22 | 0.87 / 3.00 |
| aligned < 5°, target ≥ 100 m | 0.20 / 0.48 | 0.61 / 0.70 | 0.94 / 0.82 | 0.62 / 1.09 |
| 5-20° off the target | 0.95 / 2.20 | 1.08 / 2.06 | 2.55 / 3.55 | 1.93 / 2.96 |
| ≥ 20° off any target | 0.51 / 1.01 | 0.70 / 0.56 | 1.17 / 2.37 | 1.64 / 2.20 |
| tanks firing ≥ 8 shells | 0.11 / 0.28 | 0.13 / 0.26 | 0.35 / 0.62 | 0.34 / 0.55 |
| hit / shell | 0.39 / 0.43 | 0.34 / 0.45 | 0.18 / 0.26 | 0.14 / 0.30 |
| spawn distance to nearest enemy, tanks firing ≥ 8 (m) | 56 / 35 | 55 / 48 | 107 / 68 | 141 / 68 |
| same, other tanks (m) | 153 / 82 | 158 / 82 | 151 / 82 | 145 / 75 |

## Reading

- 556 fires about twice as many shells in the first 10 s, in every context; the largest excess is off-target
  (5-20° and ≥ 20°: +2.2 to +2.7 shells/tank at side 1000 over 554).
- In 554 / 555 the tanks that empty their reserve early are those spawned next to an enemy (35-56 m) and they
  hit (0.45-0.54). In 556 they spawn at the same distance as the others (107-141 m at side 1000) and hit
  0.16-0.20: the early dump does not depend on having a target.
- Gun elevation at the shot is not low in the first 10 s of s24 (mean −0.9° / −2.1°): confirms that the
  early dump is a trigger matter.

## The trigger above the deterministic threshold (probe, side 1000, stochastic states)

| armed decisions | 554 s17 | 555 s17 | 556 s21 | 556 s24 |
|---|---|---|---|---|
| ≥ 20° off any target: share with p_fire > 0.5 | 0.015 | **0.006** | 0.020 | 0.028 |
| same, p_fire 99th percentile | 0.65 | **0.22** | 0.85 | 0.93 |
| aligned < 5°, target ≥ 100 m: share with p_fire > 0.5 | **0.013** | 0.020 | 0.128 | 0.071 |
| off-target p_fire > 0.5: median \|μ_canon_x\| (others) | 0.33 (1.07) | 0.26 (1.01) | 0.11 (0.87) | 0.06 (0.90) |
| off-target p_fire > 0.5: p_zoom (others) | 0.85 (0.83) | 0.69 (0.72) | 0.91 (0.71) | 0.93 (0.73) |
| off-target p_fire > 0.5 in runs of consecutive decisions | 0.63 | 0.30 | 0.72 | 0.79 |

Off-target fire in 556 is bimodal: p_fire is near zero on most decisions and close to 1 on 2-3 % of them, in
bursts, with the turret sweep stopped (|μ_canon_x| 0.06-0.11) and the zoom on (0.91-0.93) — the posture of an
engagement, with no living enemy within 20° of the gun axis. The deterministic agent turns each of these
decisions into a shell. Also 5-10× more decisions above threshold on aligned targets beyond 100 m.

## Verdict

Established on these saves: the early dump in game is the trigger, open (a) in engagement posture with no
target on the axis and (b) on aligned targets beyond 100 m. Not established: what the agent engages in (a) —
image content at those decisions is not exported here. Next evaluator: `arenai_agent_probe --frames` on 556
s24, frames of the off-target p_fire > 0.5 decisions, against the same decisions' panorama mask.

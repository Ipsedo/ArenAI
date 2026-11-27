# Probe run 420 (`train_20_sac`) — save_1

- Index 27k, D = 250 at the save. Run in progress (shared state encoder, aed53757).
- Binary `build/arenai_agent/arenai_agent_probe` recompiled at 15:49 from aed53757 (clean tree):
  no inference diff. The 14:20 `build/` binary already contained `StateEncoder`, but the script dated it
  to the previous commit; recompiled to remove the doubt.
- Sides 250 and 550, 4 episodes each, stochastic, CUDA (GPU shared with training).

## Summary

| side | ratio | p_al | p_off | p_arm | hit/shell | h<5° | shots<5° | enrich | empty_s | %empty | %disarmed | fire/s | turret<5° | σx | σy | \|μ\|cx | μsat dir |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 250 | 2.19 | 0.63 (n 299) | 0.29 | 0.36 | 0.087 | 0.163 | 0.113 | 1.49 | 3.6 | 0.99 | 0.73 | 0.67 | 0.089 | 0.48 | 0.43 | 0.65 | 0.25 / 0.29 |
| 550 | 2.44 | 0.54 (n 94) | 0.22 | 0.27 | 0.027 | 0.061 | 0.126 | 1.53 | 4.1 | 1.00 | 0.70 | 0.61 | 0.085 | 0.50 | 0.45 | 0.61 | 0.23 / 0.28 |

Armed p_fire by angle (side 250): 0.50 / 0.52 / 0.49 / 0.41 / 0.31 / 0.24 for
[0-2[, [2-5[, [5-10[, [10-20[, [20-45[, [45-180[ degrees.

## ∂Q/∂a_canon (qgrad, comparable to 417/419 save_1)

| side | P(right direction) armed yaw | P(right direction) armed pitch | P by yaw slice | armed \|∂Q/∂a\| | sign agreement Q1/Q2 (armed, \|yaw\| < 10°) |
|---|---|---|---|---|---|
| 250 | 0.52 | 0.51 | 0.50–0.51 | 3.3e-3 | 0.49 (x), 0.45 (y); corr 0.01 / −0.06 |
| 550 | 0.50 | 0.51 | — | 2.5e-3 | — |

corr(μ, useful command): +0.05 (yaw), +0.11 (pitch). P(μ right direction) 0.50–0.53.

## Reading

- **Selective trigger**: p_fire decreases monotonically with the target angle, on both sides
  (ratio 2.2 / 2.4). Shells fired under 5° hit 3.5× more than those fired beyond 20° (side 250).
  The discrete channel (Q(s, ·) with K outputs, exact expectation) learned when to fire.
- **Continuous aim absent**: ∂Q/∂a_canon remains without direction (0.50–0.52), the twins
  disagree in sign, canon μ is not correlated with the error. Turret under 5°: 8.5–9 % of decisions.
  Same picture as 417 and 419 at the same stage: the shared encoder did not change this lock.
- **Dump**: 99–100 % of tank-episodes empty their reserve, median 3.6–4.1 s, 70–73 % of decisions
  disarmed. The trigger's base level stays high (0.24 off-axis).
- **Heads**: canon σ 0.43–0.50, never at the ceiling, far above the 0.1 target; direction μ
  saturated 23–29 % of the time (this is where the CSV's Hc < −5 is realised).

## Established / noise

- Established: selective discrete channel; continuous ∂Q/∂a without direction; dump unchanged.
- Undecided: is the selectivity due to the shared encoder? No 417/419 probe with this script
  (no reference `ratio` at the same stage). « Aligned & close » counts small (299 / 94).

## Useful next steps

- Probe 419 save_1 side 250 with this script to get the trigger reference.
- 420 save_2/save_3: `ratio` and armed P(∂Q/∂a right direction).
- `--features` on save_1: is the bearing encoded by the shared encoder (419: R² ≤ 0.16)?

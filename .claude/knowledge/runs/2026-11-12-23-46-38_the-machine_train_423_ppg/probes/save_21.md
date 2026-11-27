# Probe run 423 (PPG) — save_21

- Index 567k, D = 1000 (final bound reached at 429k). Run in progress, code 802b1b32.
- Binary `cmake-build-release/arenai_agent/arenai_agent_probe` recompiled 2026-11-13 08:25 on 802b1b32
  + PPG support added to `probe.cpp` (`DistillationActor` branch, uncommitted): no inference diff.
  The old binary (11/11, aed53757) ignored gun laying and could not load a PPG actor.
- Sides 550 and 1000, 4 episodes each, stochastic, CUDA (GPU shared with training).
- No ∂Q/∂a nor features: the PPG actor has no Q critic and does not expose its encoding.

## Summary

| side | ratio | p_al (n) | p_off | p_arm | hit/shell | h<5° | shots<5° | enrich | empty_s | %empty | %disarmed | fire/s | turret<5° | σx | σy | \|μ\|cx | μsat dir_x | p_zoom |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 550 | 4.86 | 0.359 (836) | 0.074 | 0.139 | 0.229 | 0.369 | 0.500 | 2.04 | 9.7 | 0.70 | 0.095 | 1.08 | 0.277 | 0.37 | 0.37 | 0.23 | 0.018 | 0.47 |
| 1000 | 7.53 | 0.293 (592) | 0.039 | 0.061 | 0.191 | 0.327 | 0.467 | 2.15 | 20.0 | 0.73 | 0.060 | 0.53 | 0.235 | 0.58 | 0.56 | 0.23 | 0.010 | 0.39 |

Armed p_fire by angle of the best target ([0-2[, [2-5[, [5-10[, [10-20[, [20-45[, [45-180[ °):
- side 550: 0.34 / 0.25 / 0.14 / 0.10 / 0.10 / 0.06
- side 1000: 0.18 / 0.11 / 0.05 / 0.04 / 0.04 / 0.04

Hit/shell by angle at fire: < 5° 0.37 / 0.33, 5-20° 0.14 / 0.11, > 20° 0.015 / 0.012 (sides 550 / 1000).

corr(μ canon, target angular error), decisions step ≥ 10: yaw −0.016 / −0.009, pitch
+0.011 / +0.004; under 20° of yaw −0.026 / −0.072.

## Reading

- **Selective trigger**: p_fire decreases monotonically with the angle, on both sides. Ratio 4.9 / 7.5
  (aligned & close vs off-axis), 3.7 / 3.2 on angle alone. Half the shells leave under 5°.
  Low base level (p_off 0.07 / 0.04).
- **Shots hit**: hit/shell 0.23 / 0.19, consistent with the CSV hit (0.19–0.22 at D=1000).
  Shells fired under 5° hit 25× more than those fired beyond 20°.
- **No more dump**: 6–10 % of decisions disarmed, median 1st empty reserve 9.7 s / 20 s. 70 % of
  tank-episodes still empty their reserve at some point, but late.
- **Aim**: turret within 5° of a target 24–28 % of the time, median angle 11–12°. But canon μ
  is not correlated with the yaw or pitch error (|corr| < 0.08). Alignment therefore does not come from a
  canon command proportional to the error: it may come from the chassis (dir), from gun-laying
  stabilisation, or from a non-linear dependence. The probe cannot decide.
- **Heads**: canon σ 0.37 (550) and 0.57 (1000), never at the ceiling; canon μ small (|μ| 0.23) and
  unsaturated; dir μ barely saturated (≤ 2 %).
- **Deterministic mode** (p > 0.5): the trigger fires in only 24 % (550) / 9 % (1000) of aligned
  states. In deterministic game, the agent will fire very little.

## Against the only reference probe (SAC 420 save_1, D=250, sides 250/550)

| | 420 (550) | 423 (550) |
|---|---|---|
| ratio | 2.44 | 4.86 |
| hit/shell | 0.027 | 0.229 |
| %disarmed | 0.70 | 0.095 |
| median empty reserve | 4.1 s | 9.7 s |
| turret<5° | 0.085 | 0.277 |

No PPG reference with this script: no probe of 421 or 422.

## Established / undecided

- Established: selective and strong trigger (ratio 5–7.5), hit/shell ≈ CSV hit, dump removed, turret
  aligned 3× more often than in 420.
- Established: canon μ has no linear relation with the target error.
- Undecided: where the turret alignment comes from (chassis, gun laying, non-linearity); at which
  point of the run selectivity appeared.

## Useful next steps

- Probe save_5 (≈ 140k, D≈450) and save_12 (≈ 330k, D≈850) to date the selectivity and the fire
  drop (skill `probe-timeline` afterwards).
- Probe a 421 save on the same side to separate the gun-laying + φ effect from the PPG effect.
- For aim: corr(μ dir, yaw) or a probe with frozen chassis, to know who aligns the turret.

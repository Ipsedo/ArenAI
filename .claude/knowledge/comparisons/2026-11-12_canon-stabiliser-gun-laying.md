# Canon stabiliser — GunLaying, world aim setpoint (yaw + elevation), hinges at 20 Hz

Runs: 507 (reference), 508 (first run)

2026-11-12 — validated and coded (uncommitted): GunLaying, aim setpoint in the world (yaw + elevation), always active for player + agent; `aim_target` removed; turret/canon hinges at 20 Hz (AIM_HINGE_FREQUENCY).

Samuel's choice (2026-11-12, « choix recommandés »): 2-axis stabilisation, always active without option, `aim_target` removed from `inputs.h` (separate change, to commit apart), then hinge stiffness 2 → 20 Hz validated (« Oui, 20 Hz »).

Code: `arenai_model/src/tank/parts/gun_laying.{h,cpp}` (world yaw/elevation; `turret_angle`/`canon_angle` from the parent rotation; `anchor_*` when the hinge does not reach its target: speed or stop ⟹ no runaway; `reset` from the real canon direction), shared by TurretItem/CanonItem, created by JoltTank, re-anchored at construction and after `teleport` (respawn). `right_joystick` keeps its velocity meaning.

**Why:** the agent does no better than a frozen canon in elevation; relief dominates the need (comparisons/2026-11-12_oracle-aim-ceiling-elevation-parallax.md).

**Test plan:** run 508 from scratch vs 507 at equal D: |elevation gap| < 2° (17-24 % before), hit/shell < 50 m (41 %), hits/tank/min (5.4 / 1.9). Measured delay (critical spring): 2 Hz = 30° at full-speed rotation, 1-4° in chassis turn; 20 Hz = 6-9° and < 1°; delay ≈ speed / (π·f), inherent; test TurretFollowsItsSetPointWhileTheChassisTurns (< 1.5°). Do not probe saves ≤ 507 with this binary to compare with their old figures. (see docs/decisions.md 2026-11-12: gun laying validated)

# Absolute-angle yaw control ruins aim — reverted twice (runs 411, 413)

Runs: 409, 411, 412, 413

Absolute turret angle control (`target = a·π`, commit 986550f5, run 411) was **reverted 2026-09-23** (commit 0eb125bd): back to delta control + Jolt motor 8 Hz.

Measurements: at equal curriculum (D=400), run 409 (delta) kill=11.7 / hit=0.095, run 411 (absolute) kill=6.3 / hit=0.071. `fire` identical (~1.0/s) in both ⟹ the pointing is at fault, not the trigger (see comparisons/2026-08-17_agent-does-not-aim.md).

**Why:** unimodal Beta law gives, for a centred mode, `σ_action = 1/√(κ+1)` with `κ ≤ CONCENTRATION_MAX = 2000` (`arenai_agent/src/networks/constants.h`).
- delta: 1 action unit = 0.105 rad ⟹ jitter `6°/√(κ+1)` per frame, harmless random walk (≈1.9°/frame at the observed κ≈10).
- absolute: 1 action unit = π rad ⟹ **absolute pointing error** `180°/√(κ+1)` every frame: ≈33° 1σ at κ≈28, and **4.0° irreducible even at the κ=2000 ceiling**.

Lowering entropy targets fixes nothing: in `networks/entropy.cpp`, `error = target − H` and α is clamped ≥ 0, so **α is a floor, never a ceiling** — nothing forces the policy to tighten. The −3.0 nats target was below the law's floor (−2.38 at κ=2000), hence infeasible (same trap as in comparisons/2026-08-17_ppo-entropy-regulation-known-gaps.md).

**Recurrence — run 413 (2026-09-24).** Commit `1cd80e0d` reintroduced absolute *and* replaced Beta by a truncated normal, betting the `CONCENTRATION_MAX` ceiling was the problem. It was not: `SIGMA_MIN = 1e-4` makes the −3.0 nats target reachable, yet σ did **not move from its init** in 3 M steps — `initial_sigma = 0.5`, final σ `[0.51, 0.50, 0.39, 0.51]` (dims dir.x, dir.y, turret, canon). σ = 0.39 in absolute = **±70° pointing 1σ**. Because α is clamped ≥ 0, a lower target only disables the bonus: it never pulls σ down. Only the `max_rad_per_frame` clamp (6°/frame) masks the disaster as a low-pass — the turret dithers and never settles. A/B at equal curriculum (D = 500): 412 (delta) hit 0.083 / kill 9.56 vs 413 (absolute) hit 0.072 / kill 6.35, degrading continuously (end of run 0.059 / 3.44). Same gap as 409 vs 411.

Reverted 2026-09-24, this time **semantics only**: `TurretItem` / `CanonItem` keep `max_rad_per_frame` but read it as a rate (`angle -= x * max_rad_per_frame`), `EnemyControllerHandler` stays deleted, the player goes from `UNLIMITED_RADIAL_VELOCITY` to `PLAYER_TURRET_RADIAL_VELOCITY = 12π` rad/s (= former mouse gain 0.4π/frame at 30 Hz). Agent gain identical to run 412: 0.1047 rad/unit turret, 0.0419 canon. Entropy targets back to `{-0.2, -0.2, -0.88, -0.88}` (`-3.0/-1.4` only make sense in absolute).

**Rule:** before changing the scale of a continuous action, check `σ_action × (action→physical gain)` against the required precision. To retry absolute: either raise `CONCENTRATION_MAX` by ~3 orders of magnitude *and* add a mechanism that really pushes tightening, or output yaw as (cos θ, sin θ) — which also removes the ±π discontinuity (a=+1 and a=−1 = same physical direction, not representable as a unimodal mode).

Side pitfall of the same commit: `MotorSettings(8 Hz, 1)` removed from `turret.cpp` and `canon.cpp`, restoring the Jolt default 2 Hz (≈29° slew lag at π rad/s vs ~7° at 8 Hz) — although 44ed1f79 « fix: turret oscillating » had added it 4 h earlier. The added `tests_aim.cpp` saw nothing: they only assert the *commanded* angle (`get_angle()`) and never step the physics.

Later: absolute angle retried as cos/sin in run 505 and in run 421 (2026-11-11); see docs/decisions.md 2026-09-24 (refused, aim is a delta command).

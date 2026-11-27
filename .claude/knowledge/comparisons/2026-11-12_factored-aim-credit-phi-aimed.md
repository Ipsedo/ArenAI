# Factored aim credit + aimed Φ (A_vis) — validated and coded 2026-11-12

Runs: 508, 509 (first run with it)

2026-11-12 evening — validated and coded (uncommitted): factored advantage (canon on A_vis = λ-return of Φ without critic, `aim_gae_lambda` 0; rest on GAE 0.95) in ppo/ppg/ppo_liquid via `agents/ppo_objective`; Φ = Cauchy(miss/3 m) on visible enemies (frustum + line of sight).

Samuel's choices (2026-11-12): factored design OK, Φ on the aimed enemy (« j'adore »), no hits/kills in A_vis, the three trainers at once, measure before code.

Code: `arenai_agent/src/agents/ppo_objective.{h,cpp}` (generalized_advantages, normalized, aim_advantages, PolicyLogProbs::split with FIRST_AIM_ACTION = 2, clipped_surrogate per group, joint ratio for KL/clip); continuous log-probs stored per action; `aim_reward` in the 3 rollout buffers, `on_transition(rewards, aim_rewards, done, truncated)` (SAC ignores it); `TrainTankEnvironment::last_step_aim_rewards()`; hyperparameter `--aim_gae_lambda` (0). Φ: `get_aim_potential(tanks, aspect_ratio)` (public contract), `visible_enemies` (camera frustum with `CanonItem::get_fov()` + muzzle→centre ray_test excluding the target), predicted trajectory crossing targets, `AIM_TARGET_RADIUS = 3 m` (chassis 6.4 × 9.4 × 3.2 m). Tool aim_finetune and `train_actor` removed after the causal test.

Measurements: new Φ ρ(λ 0) 0.096/0.089 turret, 0.103/0.114 canon (pre-registered criterion 0.08); end to end the trainer's A_vis gives ρ 0.089/0.106 vs 0 for the usual advantage. Φ facing at 30 m = 0.58 (muzzle 2 m above centre). Tests: 94 model, 2440 agent, ctest 5/5.

**Why:** comparisons/2026-11-12_aim-diagnosis-mu-white-noise.md — the λ 0.95 horizon destroys aim credit, causally demonstrated (finetune 13 → 18-19 %).

**Test plan:** first run = 508 + this change; compare at equal D: turret correction gain (−0.05/10° before), nearest < 5° (9 %), tour<5°, elevation < 2°, hit/shell. Rebuild cmake-build-release before launching.

**Hull correction (2026-11-12, validated and kept by Samuel despite turret 1000 under the threshold):** miss counted from the hull, max(0, miss − 3 m)/3 m ⟹ straight ahead at 30 m Φ = 1 (before 0.58). ρ re-measured (trainer's aim A): turret 0.085/0.067, canon 0.121/0.095 (550/1000); turret side 1000 falls below the 0.08 threshold (−0.02 to −0.03, ~3 standard errors): lost signal = push toward the exact centre when the shot already hits.

**Φ plateau removed (2026-11-14, validated by Samuel):** Φ was 1 for any pass < AIM_TARGET_RADIUS (3 m) ⟹ no aim gradient under ~1.7° at 100 m, exactly the measured dispersion (1.3-1.9° in yaw and elevation, in μ). Now ratio = miss / 3 m (Φ = 0.5 at 3 m). Tests PeaksOnTheEnemy/DecaysWithTheMiss adapted; uncommitted. Judge on median aim error at 50-200 m (< 1°?) and h<5°.

**a·|a| curve turret/canon (2026-11-14, validated by Samuel):** `aim_action_to_rate` in torch_types.cpp (agent only, all algos, player intact); the probe oracle goes through the inverse `aim_rate_to_action`. Confounded with the Φ plateau removal in the same run. Old checkpoints play differently (turret response changed).
(see docs/decisions.md 2026-11-22: Φ removed, tried ≥ 8 times; 2026-11-24 refused)

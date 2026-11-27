# train_505_ppo_liquid

| | |
|---|---|
| machine | ElMatador |
| user | samuel |
| launch | 2026-11-11 18:52:28 |
| commit at launch | `aed53757` — shared encoder (CNN + proprioception) between actor and critics for SAC (learned by critics and used by actor) |
| commits ≤ 90 min after | none |
| diff at launch | unknown (identity recorded after the run ended) |
| raw data | `outputs/train_505_ppo_liquid` on ElMatador |

## Established facts

- 2026-11-11: Decision by Samuel (11/11, after probes 504; see run 504 facts): absolute setpoint for turret AND canon, reached at the current max speed (π and 0.4π rad/s, unchanged); yaw as (cos θ, sin θ) ⟹ 5 continuous actions; pitch = a·MAX_ANGLE; σ as is (SoftClamp, init 0.3), « on l'optimise normal »; `MIN_ALPHA` 1e-3 → 1e-4 global (continuous + discrete, all trainers), `initial_alpha` unchanged 1e-3; player always in velocity; setpoint via `std::optional` in `controller::user_input`; old 4-action checkpoints = dedicated binary, no compat. 3rd attempt of absolute angle after 411/413 (see memory note on absolute angle = ruined aim, reverted 2× — 411 on 23/09, 413 on 24/09). Why: σ-gradient measurement on 504 save_16 (canon_x no precision signal): velocity yaw has no one-step credit, absolute gives it near alignment. Context differs from 411/413: event reward, SoftClamp, cos/sin (no discontinuity). Known risk = σ not descending (413: σ frozen ⟹ ±70°); judge effective σ_yaw and turret <5° at probe from save_2-3; probe scripts read mu_canon_x/canon_y ⟹ adapt to new columns.
- 2026-11-11: Implemented 11/11 evening (not committed): `controller::aim_target` + `std::optional aim` in `user_input`; `TurretItem`/`CanonItem` follow the setpoint at `max_rad_per_frame` (yaw by shortest path, pitch = pitch·MAX_ANGLE bounded), else right_joystick in velocity; `ENEMY_NB_CONTINUOUS_ACTION` 5, names dir_x, dir_y, yaw_cos, yaw_sin, pitch; `MIN_ALPHA` 1e-4; probe (columns by name) + `probe_metrics` scripts (dims read from header, σyaw° = angular dispersion of the setpoint); 6 `AimTest` tests added; 5 ctest suites ok in `build/`. `cmake-build-release` not rebuilt (504 running).
- 2026-11-11: Reading at 20 k (19:10, 23 rows, D 250) vs 500-504 at same index: hit 9.4 % vs 7.1-8.1, r_hit 0.0059 vs 0.0030-0.0036 (partly fire 1.7 vs 1.1-1.45/s), ev 0.54 vs 0.10-0.36, kill 6.8 ≡ 504 (7.3), Hc −0.05 (not comparable: 5 dims). Promising, not settled.
- 2026-11-11: Probe save_2 (54 k, D 350, ~20:00): trigger flat at init (p_fire 0.50-0.55 everywhere), dump 3.5 s in 98-100 %, 1.8 shot/s (2× 504 save_5) ⟹ CSV lead comes from fire rate; hit/shell 4.2 / 1.5 %; turret <5° 16 % (504 save_5: 13.8); |μ_yaw| 0.13/0.08 < σ 0.28/0.25 ⟹ setpoint dispersed at 77-80° (slew low-pass as 413). Judge at save_5: |μ_yaw| > σ, σyaw° falling. Report `probes/save_2.md`.
- 2026-11-11: Probe save_5 (135 k, D 400, evening) vs 504 save_5: hit/shell 7.1/2.6 % (504: 8.4/3.9), turret <5° 14 % (≡ 504, < save_2 16 %), reserve 11-12 s (504: 6 s), p_arm 0.13, angle-only ratio 1.25/1.06; σ_yaw 0.28 → 0.44/0.35 (alpha_yaw_sin/pitch up to 0.004 near 65-90 k, target H 0), |μ_yaw| 0.21/0.16 < σ ⟹ setpoint dispersed at 73°. Unmeasured hypothesis: target H 0 ⟹ σ ≳ 0.25 ⟹ floor ~14° even at |μ| = 1. D 400 at 67 k (504: 114 k) but D 450 at same index. Report `probes/save_5.md`.
- 2026-11-11: Computed floor (evening, Monte-Carlo tanh-Gaussian, H per dim ≥ target, optimal μ and σ): minimum RMS dispersion of yaw setpoint = 28-40° at target 0, 12-17° at −0.5, 6-7° at −1 (σ ≈ 0.25 on the μ-zero dim); save_5 state = 78°. Target 0 of 502 makes absolute (cos, sin) structurally imprecise; turret (slew π rad/s) low-pass filters so real error is smaller than setpoint's. Levers submitted to Samuel: per-dim target (yaw/pitch lower, dir at 0) (led to run 506).

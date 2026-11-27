# Canon σ collapsed early in run (0.02-0.04 up to 300 k); « collapsed σ = brake » hypothesis withdrawn

Runs: 510, 514, 522, 546, 547, 548 (save_7/11)

2026-11-22: σ canon_x 0.02-0.04 from 81 to 300 k then 0.3; hypothesis « collapsed σ = brake » WITHDRAWN (closure 0.3-1°/decision identical in 510-547 whatever σ); end-of-episode GAE + target −1 committed `dcca274e`.

Measurement 2026-11-22 after the refutation of « perception = source » (comparisons/2026-11-21_target-localisation-source-cause.md), `sigma_gradients` on control 548 save_7 (`outputs/comparisons/2026-11-22_temoin_gisement/grad.py`, `grad_548_s7.csv`) + probes 546/547/548.

- Φ₂ credit correct and monotone: action toward target > 0.2 unit ⟹ advantage +1.0; opposite −0.3; ρ(A, gesture) 0.10 at < 60 m.
- Entropy bonus traction on μ: 0.0009 vs 3.8 for credit.
- Turret delay minor: 8-10°/unit in the same decision, 1.4-5.7 at the next.
- Median σ canon_x: 0.033-0.038 (547 s3-s11), 0.024-0.028 (548 s7/s11), 0.14-0.16 at s13, 0.26-0.31 at s24-s27 (546/547). μ displacement allowed by KL threshold 0.075: σ·√0.15 ≈ 0.01 unit per rollout at start; kl 0.07-0.09 and skip_canon 13-20 % at 100-200 k; α_canon_x 0.004 at 189 k (regulator too slow at start, overshooting at the end).
- Control's yaw-column weight: +0.4 to +0.5 per 100 k, linear.
- Not demonstrated: that this brake alone explains the weak command (≈ 8 % of full authority).

Samuel's decisions 2026-11-22 (one question per piece): GAE bootstrapped on the last state of the episode at time limit; entropy target −1 on the four continuous actions, **without** hard σ floor nor faster regulator (proposed, set aside); empty end of game not handled in this run.

Coded uncommitted in the repo (build/ compiled, 2 448 tests ok, only failure = pre-existing unstable `TruncatedNormalEdgeTest.LogPdfAndPdfConsistent`; real 4-min trial with 20 s episodes without error): `episode_ends` + `closing_states` in the 3 rollout buffers, `closed_next_values` / `with_episode_ends` in `ppo_objective`, the 3 trainers (GAE recursion and ReturnScaler cut at episode ends, bootstrap kept), default `continuous_target_entropy` −1 in the 3 hyperparams. Shells per hit back to 2 (like 546) 2026-11-22; `main.cpp` now reads the default from GAME_SHELL_ECONOMY. Samuel runs himself, on the other PC, the twin run without Φ in full single advantage (one ratio, one KL stop).

**Rule:** read the next run on median σ canon_x at saves 3-11 and the directed command (`compare.py`, `gain.py`); if σ falls under 0.09 at the start, re-propose the floor (but see withdrawal below). Never `pkill -f` a pattern present in the current shell's command line (redone 2026-11-22): kill by PID.

**Hypothesis « collapsed σ = brake » withdrawn 2026-11-22 (inter-run comparison, probes side 550):** real closure on visible target < 100 m at 5-30° = 0.3-1.0°/decision and « < 5° in 1 s » = 19-33 % in ALL runs, whether turret σ is 0.03-0.05 (546 s9, 547 s11) or 0.2-0.55 (510 s9-s28, 514, 522 s10-s68); std of Δyaw 3-5°/decision everywhere. Invariant to σ, a·|a| (514/522), Φ/Φ₂, 2/5 shells, target position as input (548). Never propose a σ floor again (no patches; see docs/decisions.md 2026-11-22: hard σ floor refused). Unmeasured lead: Claude's « toward target » aims at the nearest visible while Φ₂ credits the argmax over all enemies.

**Φ₂ target measured 2026-11-22 (`cible_phi.py`, `variance_credit.py`, horizontal replica of Φ₂ on `grad_548_s7.csv`, corr 0.61 with run credit):** when the nearest is visible < 100 m at 5-30°, Φ₂ designates that same enemy 94 % of the time ⟹ hypothesis « my measurement looks at the wrong target » rejected. ρ(credit, gesture toward Φ₂ target) 0.08-0.10 at any gap, out-of-field included; directed command 0.04 unit in view, 0.01 out of field. Variance of ΔΦ: 41 % own turret rotation, 65 % exogenous; exploration noise explains only 4-5 % of own rotation (corr −0.2). Control's directed command flat at s3/s7/s11 (0.87 / 1.10 / 0.96°/decision) while the input weight grows, and flat in 522 from s10 to s68 ⟹ plateau, not slow growth. Cause of the plateau not found; proposed test: replay a per-group update (canon only / rest only) from a save and measure displacement of the directed command.

**Per-group update replay (2026-11-22, 548 save_11, 12 rollouts, `rejeu.py`, `rejeu_548_s11.csv`; tool `sigma_gradients --update_replay` + `LiquidPpoTrainer::update_replay` in worktree `arenai_wt_temoin` only):** quadratic displacement of the turret command per update 0.014-0.018 unit; the « rest » group update alone moves it as much or more than canon alone (0.018-0.020 vs 0.016) and moves log σ turret by +0.029 (canon alone +0.017) ⟹ interference through the shared trunk, established. Directed share ≤ 10 % of displacement: visible target < 60 m +0.0012 ± 0.0029 (real), +0.0016 ± 0.0024 (canon alone), −0.0040 ± 0.0036 (rest alone), not significant at 12 rollouts. Directed gain « in sample » also out of field (+0.003 ± 0.001), where direction is unobservable ⟹ suspicion of rollout fitting, to confirm out of sample. Trigger: aligned p_fire falls in all three modes, canon alone included (−0.0015).

**Extended replay (2026-11-22, 548 save_11, 60 rollouts, `rejeu2.py`, `rejeu2_548_s11.csv`; per-group modes with Adam momentum zeroed; out-of-sample rows = previous rollout). Replaces the 12-rollout replay, whose « rest alone » mode was polluted by momentum.** Δ directed command per update, ×1e-3:
- canon alone, in sample: all rows +3.55 ± 0.66, out of field +3.47 ± 0.80 (direction unobservable), visible < 100 m +3.45 ± 1.01;
- canon alone, out of sample: all rows −0.52 ± 0.40, out of field +0.04 ± 0.54, visible < 100 m −0.33 ± 0.78, visible < 60 m +2.18 ± 1.15;
- quadratic displacement out of sample 0.0139 (canon alone), 0.0100 (rest alone), 0.0146 (real).
⟹ **the canon update fits its rollout and does not generalise**: the policy moves ~0.014 unit per update on unseen states, without direction; that is the plateau. Interference of the « rest » group through the trunk confirmed without momentum: 70 % of canon's own amplitude, undirected.
- Trigger, Δp ×1e-3 out of sample: real (momentum kept) −3.36 ± 0.42 aligned < 2°, −6.16 ± 0.78 off-axis; without momentum ≈ 0; rest alone +0.58 ± 0.43 (+1.51 ± 0.50 in sample). Confounded: the tool's environment lacks the empty-reserve penalty (0 without curriculum) and plays side 550, while momentum comes from real training; do not conclude « it is the momentum ».

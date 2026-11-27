# Aim diagnosis: turret jitters through μ itself (state white noise), not σ

Runs: 507 (save_5/9/12), 508 (save_5/9)

2026-11-12: the turret agitates through μ itself (state white noise, std 0.44 ≈ 8°/decision) and not through σ (5°); 4-14 % of var(μx) explained by yaw, growing over saves; single scalar advantage for 6 dims (ppo/ppo_liquid/ppg identical).

Measured on probes side 550 (507 save_5/9/12, 508 save_5/9):
- clean plant under stabiliser: action t → +9.3° at t and +9.0° at t+1 (18°/unit); 507 (2 Hz) spread 0.8/5.2/6.0° over 3 decisions ⟹ the Φ shaping of 507 paid action t for gestures of t−1/t−2.
- std tanh(μ_canon_x) 0.21 → 0.46 (507), 0.30 → 0.44 (508); autocorr μx(t,t+1) 0.09 at 508 save_9; share explained by yaw 0.04-0.14; μ_y same (R² 0.01-0.14).
- wanted correction k = 0.044 of the gap/decision; off-command perturbation 5.9-7.4°/decision.

Code: single normalised GAE advantage, joint ratio of the 6 dims, common clip, in the 3 trainers; Φ shaping (0.3) in r_t.

**Why:** the aim lock after run 508 (stabiliser); same mechanism as comparisons/2026-11-11_peak-does-not-hold-critic-leads-actor-follows.md (random walk of the actor).

Levers proposed 2026-11-12 (nothing decided): temporal smoothing of μ (CAPS), separate advantage for aim dims (critic with heads per reward component), re-test stronger Φ now that credit falls on the right step, absolute command to reopen cautiously.

**Measurement 2026-11-12 evening (outputs/train_508_ppo_liquid/gradients/save_9.md, sigma_gradients tool enriched with u, reward decomposition, geometry):** ρ = corr(A, ε·d) of the trainer's advantage = 0.003-0.02 (t/minibatch < 0.5); Φ at the same step responds at 0.10-0.13; ρ decreases monotonically with λ (Φ alone turret 0.116 → 0.005 from λ 0 to 0.95); separate aim advantage at λ 0.95 = also null ⟹ lever B as is refuted, the source is the credit horizon, not the reward mix. Pitfall: elevation sensitivity ∝ cos(yaw), filter |yaw| < 10°; probe columns already in degrees; yaw to fold ±180°.

**Causal test 2026-11-12 19:20-20:05 (outputs/train_508_ppo_liquid/finetune/verdict.md, tool arenai_agent_aim_finetune + LiquidPpoTrainer::train_actor, uncommitted):** 40 updates from save_9, critic frozen, 2 trials/arm: GAE λ 0.95 advantage ⟹ tour<5° 8.3/13.6 % (start 13.1); immediate Φ ⟹ 18.8/18.1 %, best_angle 22 → 12-14°, μx bias removed. Horizon = causal lock demonstrated. But nearest < 5° unchanged (Φ rewards any enemy) and hit/shell not better (confounded: Φ also applied to fire/driving, p_fire 0.27 → 0.16).

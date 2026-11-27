# Aim credit in the single advantage — 524 save_9 (243 k, D 600) and save_16 (432 k, D 950)

Tool `build/arenai_agent/arenai_agent_sigma_gradients` (2026-11-16 23:07, run code), 2 episodes per side, λ 0.95 (the run's) and λ 0 by config copy. ρ = corr(x, ε·d), protocol of `train_522_ppo_liquid/gradients/save_68.md`; scripts `rho.py` and `echelle_phi.py`. Standard error 0.007 (turret, n ≈ 21 k) and 0.012 (canon, n ≈ 6-8 k). Threshold 2026-11-12: ρ ≥ 0.08.

## ρ with the corrective gesture (550 / 1000)

| signal | s9 turret | s9 canon | s16 turret | s16 canon |
|---|---|---|---|---|
| trainer advantage, λ 0.95 | +0.009 / +0.008 | +0.023 / +0.003 | +0.014 / +0.007 | +0.022 / +0.015 |
| advantage, λ 0 | +0.042 / +0.018 | +0.038 / +0.028 | +0.036 / +0.016 | +0.030 / +0.021 |
| step reward, no critic | +0.048 / +0.046 | +0.047 / +0.008 | +0.041 / +0.018 | +0.036 / +0.016 |
| step Φ (step without event) | +0.099 / +0.080 | +0.096 / +0.082 | +0.082 / +0.039 | +0.077 / +0.050 |
| Φ λ-return alone, λ 0.95, no critic | +0.020 / +0.019 | +0.039 / +0.017 | +0.025 / +0.005 | +0.025 / +0.029 |

Enemy out of field (≥ 40°): ρ between −0.004 and +0.010 for all signals.
Simulated Φ scale (A + (k − 1) × Φ λ-return, critic unchanged), turret side 550 at s9: ×1 +0.009, ×3 +0.010, ×10 +0.013, ×30 +0.018, ×100 +0.020. Ceiling = the "Φ λ-return alone" row: 0.005-0.04 everywhere.

## Trigger: normalised advantage gap shot − no-shot, λ 0.95 (550 / 1000)

| state | s9 | s16 |
|---|---|---|
| < 2° and < 100 m | +2.36 / +3.59 | +1.93 / +3.62 |
| 5-20° | +1.31 / +2.39 | +1.47 / +2.14 |
| ≥ 20° | −0.21 / −0.14 | −0.11 / −0.41 |

## Established

1. The run's advantage does not carry the aiming gesture: ρ 0.003-0.023, at standard-error level.
2. The signal exists at the step itself (Φ: 0.04-0.10); lost in the sum over the λ 0.95 horizon (÷3 to ÷5 for Φ alone), then in event noise.
3. Raising Φ scale at λ 0.95 plateaus at ρ 0.02-0.04, under threshold.
4. The trigger receives a strong well-oriented signal (+2 to +3.6 std aligned and close), stronger than 522 s68 (+1.1/+1.4).
Limits: simulated scale keeps the run's critic (a critic learning Φ would cancel the term); `advantage` column normalised by the tool: ×k mixture approximate, ceiling not; selection bias on trigger gap (cf. 522).

## Addendum: what events alone credit (no Φ, no critic)

Turret 1-40°, rollouts > 100 steps from end (n ≈ 15-17 k, s.e. 0.008), s9 550 / s9 1000 / s16 550 / s16 1000:

| signal | ρ with corrective gesture |
|---|---|
| step events | +0.009 / +0.019 / +0.002 / −0.001 |
| events over 2 s (λ 0.95) | +0.013 / +0.005 / +0.014 / +0.005 |
| events over 10 s (λ 1) | +0.013 / +0.005 / +0.005 / +0.001 |
| step Φ | +0.095 / +0.085 / +0.079 / +0.042 |
| Φ over 2 s (λ 0.95) | +0.020 / +0.017 / +0.021 / +0.005 |

Std: events over 2 s 0.23-0.45, step Φ 0.025-0.040. Trigger, aligned < 2° and < 100 m: 2-s event return after a shot +0.66 to +1.45, without shot +0.09 to +0.22. Noise is the same for all actions; it is the effect of a decision that differs: an aligned shot shifts the return by 2 to 3 std, a turret gesture by less than a hundredth.

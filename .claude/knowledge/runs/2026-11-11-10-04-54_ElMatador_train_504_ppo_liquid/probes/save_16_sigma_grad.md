# Pushes on log σ — 504 save_16 (432 k, D 950)

Tool: `arenai_agent_sigma_gradients` (`build/`, uncommitted code). Reloads the save (actor, critic, alphas, return_scaler), collects with the real collector and `TrainTankEnvironment` at fixed side, computes the trainer's GAE, then per living row:
- `g_pg` = ∂surrogate/∂log σ = Â·(ε² − 1) at epoch 1 (ratio = 1);
- `g_h` = ∂(α·H)/∂log σ, same entropy estimator as the trainer.

SoftClamp's Jacobian is common to both terms of a row and not included. 2 episodes per side, 12 rollouts of 300 steps, 82 k rows (side 550) and 99 k rows (side 1000).

## Results (mean per row ± standard error)

| dim | σ p50 | entropy | PPO, all rows (550 / 1000) | PPO, armed <5° (550 / 1000) | PPO, fire decisions (550 / 1000) |
|---|---|---|---|---|---|
| dir_x | 0.65-0.66 | +0.0004 | +0.008 ± 0.005 / −0.002 ± 0.005 | +0.053 ± 0.027 / −0.012 ± 0.021 | +0.063 ± 0.037 / −0.020 ± 0.042 |
| dir_y | 0.65-0.66 | +0.0005 | −0.002 ± 0.005 / +0.006 ± 0.004 | −0.011 ± 0.021 / −0.011 ± 0.020 | +0.006 ± 0.030 / +0.040 ± 0.036 |
| canon_x | 0.68-0.69 | +0.0005 | −0.003 ± 0.005 / +0.004 ± 0.005 | −0.009 ± 0.023 / −0.004 ± 0.019 | −0.032 ± 0.034 / −0.045 ± 0.037 |
| canon_y | 0.60-0.63 | +0.0006 | −0.002 ± 0.005 / −0.003 ± 0.005 | **−0.048 ± 0.021 / −0.046 ± 0.020** | −0.023 ± 0.029 / +0.009 ± 0.035 |

PPO mean per rollout: ±0.01 to ±0.05, random sign (5 to 9 positive rollouts out of 12). Rows of one chunk are GAE-correlated, so standard errors are underestimated.

## Reading

- **Entropy bonus is small but the only constant component.** Per row +0.0005 on the 4 dims: 30 to 100× below one rollout's PPO noise, 10× below the standard error of the PPO mean over 80-100 k rows. Adam accumulates ~10⁵ steps (≈ 1 440 rollouts × 4 epochs × ~19 minibatches). A constant drift grows as N, zero-mean noise as √N. The steady σ rise (0.48 → 0.62 → 0.69 from save_5 to save_16) is compatible with this mechanism. **Not demonstrated**: needs a drift measured over several saves.
- **canon_x: no detectable precision signal.** Neither over all rows nor over armed aligned states (|mean| < 0.01, compatible with 0). On fire decisions, mean −0.03 / −0.045 ± 0.035: same sign on both sides but only 1σ. PPO gradient does not retain the turret's σ against entropy.
- **canon_y: a precision signal exists.** On armed aligned states, mean −0.048 / −0.046 ± 0.02: ~2.3σ, same sign on both sides, 90× entropy. Consistent with σ_canon_y never touching the ceiling (0 % vs 27-44 % for canon_x).
- dir_x and dir_y: nothing stable across sides.

## Consequence for the α floor

Lowering the floor would remove the only constant push. For canon_y, PPO gradient would probably tighten σ on aligned states. For canon_x nothing pulls it down: σ would random-walk (Adam) instead of rising, with no reason to become precise. The real gap is the absence of credit on turret precision (canon_x is a yaw velocity; its noise only has effect integrated over several decisions).

## Possible follow-ups

- Measure the same on save_5 and save_9, and on a 500/502 save, to know whether canon_x signal was always null or disappeared.
- More rows (4-6 episodes per side) to bring standard error under entropy.

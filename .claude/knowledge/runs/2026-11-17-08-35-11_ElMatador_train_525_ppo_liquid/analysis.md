# Learning audit: why aim is not learned (run 525)

2026-11-17. Code read: `liquid_ppo_trainer.cpp`, `ppo_objective.cpp`, `update_utils.cpp`, `return_scaler.cpp`, `entropy_alphas.cpp`, `gaussian_tanh.cpp`, `liquid_actor.cpp`, `liquid_cell.cpp`, `liquid_critic.cpp`, `vision.cpp`, `misc.cpp`. Measures: side-550 probes of 525 (s1-s12), 524, 522 and the separate-aim-credit control (430 of the other PC), `actor.pt` checkpoints.

## 1. What the turret really does

Command direction "toward target" = mean of −tanh(μ canon_x)·sign(yaw of nearest).

| state | 525 s9 | 525 s12 | 524 s9 | control s9 | control s16 |
|---|---|---|---|---|---|
| target in view < 20° and < 100 m: toward target | +0.09 | +0.14 | +0.11 | +0.08 | +0.09 |
| same state: mean bias | +0.17 | +0.17 | +0.11 | −0.01 | 0.00 |
| same state: σ | 0.48 | 0.57 | 0.55 | 0.22 | 0.21 |
| target in view at 100-300 m: toward target | 0.00 | +0.01 | +0.02 | +0.07 | +0.04 |
| out of field (> 40°, 67-72 % of time): mean bias | +0.29 | +0.24 | +0.29 | −0.01 | 0.00 |
| mean \|μ\| | 0.45 | 0.60 | 0.47 | 0.25 | 0.24 |

- Correction toward a near target is learned, at the control's level.
- Two differences: a constant rotation bias (+0.24 to +0.29) that persists with target in view and exceeds the correction; no correction beyond 100 m.
- Dwell on target: as many entries at < 5° as control (16-25 vs 21-24 per tank-minute), duration 0.20-0.28 s vs 0.97-0.99 s. The turret finds the target and overshoots it.
- Deterministic replay of s12: tour<5° 6.7 % vs 6.4 %, dwell 0.20 s in both. Exploration noise is not the cause; the mean command sweeps.

## 2. Verified, no anomaly

- Init: σ = 0.30 (bias 1.5 in SoftClamp), μ = 0, output weights 0.01.
- tanh-Gaussian density, GAE with done/truncated cuts, advantage normalisation on valid steps, scaling by return std, 30-step chunks restarting from recorded hidden state.
- Critic: ev 0.6-0.8, network separate from actor.
- Liquid cell: τ p10/p50/p90 = 0.12 / 0.42 / 1.43 s, identical in 525, 524 and control (init 0.13 / 0.45 / 1.48: almost not learned). Vision / proprio / recurrent weight norms comparable (0.33 / 0.15 / 0.37 vs 0.35 / 0.17 / 0.48).
- Same network, same hyperparameters and same CNN hold the target in the control: no architecture piece forbids aiming.

## 3. Weak points found

1. Aggressive update regime in all runs: clip 0.27-0.39, kl 0.04-0.07 per minibatch, skip 15-33 % (525, 524, control; 522: clip 0.33-0.39, kl 0.04-0.08). 76 Adam steps per rollout (4 epochs × 19 minibatches of 512 rows = 17 chunks). A third of rows clipped. Common to good and bad runs: does not separate, but amplifies point 2.
2. Aim signal under noise in the single advantage. Measured on 524: ρ(advantage, corrective gesture) 0.003-0.023. At ρ = 0.02 ~2 500 rows are needed for SNR 1; a minibatch has 512, a rollout 9 600, reused 4 times. Canon dimensions mostly follow minibatch noise.
3. No pressure on canon σ. α_canon_x stays at floor 1e-4 in 524/525; in 522 and control it is at 0.006-0.015 from the start, sign PPO tightened σ below target there. Here σ rises 0.30 → 0.60.
4. SoftClamp: high σ = attenuated gradient. Upper bound ln 0.7; sigmoid 0.82 at σ 0.30 and 0.97 at σ 0.60, i.e. log σ gradient ÷4.7 at 0.60 and ÷10 at 0.65. Descending from 0.6 is slow by construction. Not causal for target holding (deterministic test) but delays any return.
5. Saturating command. Median |μ| canon_x 0.29 → 0.49 → 0.82 from s5 to s12, 17 % beyond 1.5; mean dir_x −0.59. With a·|a| at output the turret is driven all-or-nothing.
6. Field of view. Nearest enemy (median 53 m side 550) out of field 67-72 % of time in all runs. Searching is necessary; sweeping is the answer events reward.
7. Perception range. No correction beyond 100 m (control +0.04 to +0.07). Supervised bench 2026-11-16: left/right 0.65-0.68 with this CNN (final map 1×2).

## 4. Reading

No training bug identified. What the run learns is coherent with what the advantage carries: the trigger (signal +1.9 / +3.6 σ) becomes selective, correction toward a near target exists, and the retained turret strategy is "sweep and fire in passing". Holding the target needs several coordinated gestures whose benefit arrives 1 to 3 s later via a rare event; at ρ ≈ 0.02 and 512 rows per step this credit does not emerge from noise. The control holds the target because its aim credit was immediate (λ 0) and separate.

## 5. Options (no decision)

- Hyperparameters alone, not touching credit: reduce per-optimizer-step noise (larger actor minibatch, fewer epochs, lower actor lr) to leave the clip 0.3 / kl 0.05 regime. Reservation: 503 (batch = rollout, lr 5e-4) did not conclude, and 480 (epochs 2) was under 478; expected gain a factor 2-4 on SNR, for a threshold needing ~5.
- Auxiliary head on actor (reopened by Samuel 2026-11-17): acts on point 7, not point 2.
- Separate aim credit: the only measured mechanism that makes target holding work (control, 510-522), rejected 2026-11-16/17.

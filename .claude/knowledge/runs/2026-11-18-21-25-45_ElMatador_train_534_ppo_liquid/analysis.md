# Run 534 — ppo_liquid, coloured noise alone (start)

Launched 2026-11-18 21:25, running, read at 69 k (D 350), 90 k steps/h. Effective commit b83da585 (launched uncommitted, committed 22:01; good confidence, config ≡ header defaults).

## What the run tests

533 without the short value head (`short_gamma` removed): coloured AR(1) noise ρ 0.9 alone. One change vs 533.

## CSV

D milestones: 300 at 48 k, 350 at 67 k (533: 300 at ~48 k, 350 at ~72 k; 528: same ±5 k).

| index | run | kl | clip | skip | kill | hit |
|---|---|---|---|---|---|---|
| 1.8 k | 528 / 529 (white noise) | 0.009 / 0.008 | 0.15 / 0.13 | 0 | – | – |
| 1.8 k | 533 / 534 (coloured) | 0.030 / 0.026 | 0.34 / 0.31 | 0 / 0.10 | – | – |
| 66 k | 528 / 529 | 0.044 / 0.058 | 0.33 / 0.36 | 0.005 / 0.05 | 17.7 / 12.1 | 0.131 / 0.088 |
| 66 k | 526 (batch 2400) | 0.009 | 0.12 | 0 | 19.6 | 0.104 |
| 66 k | 533 / 534 | 0.067 / 0.068 | 0.37 / 0.39 | 0.49 / 0.54 | 21.0 / 19.0 | 0.103 / 0.104 |

Skip 534: 0.11 at 10 k, 0.36 at 26 k, 0.48 at 43 k, 0.54 at 69 k (+1 pt / 4 k, decelerating). Hc −0.26, α at floor everywhere, ev 0.5-0.7.

## Diagnosis

- Established: the KL regime comes from coloured noise, not the short head (534 ≡ 533 on kl, clip, skip at any index) nor from learning (kl ×3 from the first rollout, untrained policy).
- Mechanism (reading): in a 30-step chunk ε is almost constant (ρ 0.9 ⟹ ~1.6 independent draws per chunk instead of 30). The μ gradient Σ A·ε/σ adds coherently per chunk: a 512-row minibatch = 17 chunks ≈ 27 independent draws. The Adam step (lr unchanged) thus follows a direction ~20× noisier, which the liquid (low-pass) network fits ⟹ ratio diverges fast, clip 0.39, stop at 1.5 × target_kl = 0.15.
- Scope: 76 planned steps per rollout (19 per epoch × 4), ~35 executed: epoch 1 whole, epoch 2 partial, epochs 3-4 never. All data still seen once; no longer true beyond 75 % skip.
- Not settled: that this regime hurts. 533 s3 (skip 55 %) gave the first positive sign on the turret; 526 (kl ÷5, skip 0) gained nothing on aim and slowed the trigger.
Options (no decision): (1) let it run: keeps the clean comparison with 533 s3 (only change = short head); (2) `--actor_learning_rate 5e-5`: kl expected ÷3-4, same 76 steps per rollout; confounded with 533; (3) `--actor_batch_size 1024` (or 2400, measured in 526): mean over more chunks, fewer steps; (4) `--noise_correlation 0.7`: 5.7 correlated draws instead of 19; changes the tested lever.
Markers: skip > 0.75 = data not seen: stop threshold; save_3 (81 k): share of turret μ opposed to yaw vs 533 s3 (0.63-0.65) and 529/532 s3 (0.55).

# PPO liquid train 418 — plateau, actor moves on noisy advantages (2026-09-30, updated 2026-10-04)

Host: ElMatador (memory notes); raw data no longer on disk.
Source note date: 2026-09-30 (weights measurement corrected 2026-10-04). `train_418_ppo-liquid_skip_larger-cnn_epochs-2`, 5.5 M steps: config identical to 417 (relaunched after an accidental stop). Same code (`87c508e8` only touches the `epochs` default).

- **Early peak reproduced**: hit 0.092-0.095 / kill 9.6-10.5 between 0.4 and 0.9 M (D 266-345), as 417 ⟹ real, not luck.
- **Plateau from ~1.1 M** (D≈363): hit 0.07-0.085, kill 6.4-9.4 up to 5.5 M. At equal D back to 414's level (D=500: 0.085 / 9.4 vs 0.084 / 9.0; D=600-700 slightly under 414). Curriculum stuck at D=700 from 3.9 to 5.5 M.
- **Actor not frozen**: kl 0.011-0.023, clip 0.11-0.22, skip ≈ 0, π loss stable −0.014; α_c ~0.003 and α_d ≤ 0.0026 negligible in the loss. It moves without progress.
- `ev` median 0.12-0.29 and **partly measured on already-seen data**: `train_critic` computes ev before the step but from epoch 2 the minibatch has been used (`liquid_ppo_trainer.cpp`, ev added after `critic_optim->step()` on pre-step values). True out-of-sample ev ≤ 0.2. So 416's ev 0.4-0.5 (4 epochs) is mostly in-sample, not a better critic.
- `Hd` falls steadily 0.62 → 0.34 (target 0.277), `fire` ~1.08 (still bound by ammunition).

**Decisive measurement (2026-09-30, weights only, reliable).** Relative movement per save (||ΔW||/||W||) constant from save_15 to save_50: deep CNN ~10-11 %/save, heads 20-35 % (corrected 2026-10-04: the first measure at ~4.5 % divided a 5-save displacement by 5, underestimating by √5). Random-walk test (net displacement / path over 15 saves; directed ≈ 1, random ≈ 0.26): **whole CNN, skip, μ, σ ≈ 0.26 in ALL phases, even during the peak**; only `liquid.to_output` (0.73), `a/τ` (0.55) and `recurrent_weights` (0.41) were directed early, then fall to ~0.25-0.33 after ~save_20. ⟹ the actor **diffuses**: Adam (lr 1e-4 constant, no decay) turns a noise-dominated gradient into steps of size ~lr; the CNN is rewritten at random (~40 % of its norm on the plateau) and downstream layers chase it. Probable noise source: minibatch 1024 rows = **34 chunks** of 30 correlated frames.

Samuel rejected (rightly) the « unobserved shells in flight » explanation: firing is visible on the frame (cooldown reset to 0, shell in image) and the LTC can remember it; other runs converged with the same proprioception.

**Critic (same measure, lr 3e-4)**: moves ~1.4-3× more than the actor per save — deep CNN 34 → 14 %/save, LTC weights 17-55 %, `to_value` **27-43 %/save**. Net/path ratio 0.20-0.25 everywhere, and **`to_value` at 0.13-0.14 (< 0.26): not anti-correlated, the value head oscillates** — signature of too large an lr. Hypothesis « critic makes the actor diffuse » **refuted by 419**.

**CNN (same checkpoints, weights only).** Actor, save_50: γ of the 6 GroupNorms always ≈ 1.0 (p10 0.93-0.98, max ≤ 1.13 — init), the 2048 output features read with the same column norm (p10/p50/p90 = 1.20/1.32/1.47, none dead, none privileged), no favoured spatial cell (20-22 each), vision and sensors weighted alike per feature (ratio 1.0×) from save_0 to save_50. ⟹ **the actor CNN learned nothing identifiable in 5.5 M steps.** The critic specialises: γ 0.58-1.85, sensors weighted 1.6-1.9× more than each vision feature (and growing), lower half of image read more. No dead channel anywhere. Geometry 417/418: 128×256 → 2×4×256 (2048 features), receptive field 127 px, cells spaced 64 px ≈ 20°, 1 stride-2 conv per scale without residual, 1st block 8 channels, ~393 k CNN params read by ~2.2 M dense params (liquid.f + skip).

- Why: hit ceiling ~7-9 % holds on vanilla/liquid, 3 CNNs, LTC 128/512 with and without skip, Beta/normal, kl 0.003/0.015/0.06 ⟹ neither capacity, nor lag, nor throughput is THE lock. The 417/418 arch learns faster the short-range solution, which no longer holds beyond D≈350 (miss never better, even at the peak).
- Proposed order (not all run): `--critic_learning_rate 1e-4` (only variable) = run 419; then `--actor_learning_rate 3e-5` if the actor still diffuses; then larger `--minibatch_size`. Verify with the random-walk test (ratio must rise above 0.26 and kl fall). No resume-from-checkpoint in main.cpp: each test starts from zero.
- Related: architecture-lock candidates note; run 417.

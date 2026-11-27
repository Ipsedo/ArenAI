---
name: ppo-train-368-diagnosis
description: "Run 368 ppo_impala rollout 30 decisions (480 samples/update), 1 Adam step/epoch lr 3e-4 — KL explodes (skip 0.7), constant critic (ev = 0.000), σ and Hd collapse, hit 0.02 → 0.007"
metadata:
  type: project
---

Run train_368_ppo-impala (2026-10-28, 160k decisions in 6h45, ~2.5× slower than 364) — commits dd0487b6/6218d9f0: rollout_size 900 → 30 (×16 tanks = 480 rows), actor epochs 4 / critic 8, lr 3e-4, entropy c 0.001 / d 0, state-dependent σ.

- 0-20k: healthy (clip 0.05, kl 0.004, ev 0.4-0.55, Hc 0.3).
- from ~20k: Hc → −1.0, Hd 0.69 → 0.12; kl 0.3-0.9 and skip 0.5-0.73 (theoretical max 0.75 with 4 epochs, epoch 0 never cut) → a single Adam step suffices to exceed 1.5×target_kl.
- ev stuck at ±0.000 over long ranges = var(R−V) = var(R) ⇒ V nearly constant over the minibatch: dead critic.
- hit 0.02 (start) → 0.007 at end, kill → 0, curriculum frozen D=250.

Retained cause: 480 samples + 1 Adam step at lr 3e-4 (step ∝ lr independent of noise) + advantage normalisation over a 3 s rollout where the sparse reward is often absent → policy gradient ≈ noise amplified to variance 1; drift of σ (state-dependent, entropy 0.001) → KL hypersensitive ((Δμ/σ)²). GAE horizon 30 vs 1/(1−γλ)≈62 with constant critic ⇒ no useful bootstrap.

See ppo-full-batch-update, [ppo-train-364-diagnosis](ppo-train-364-diagnosis.md), [ppo-train-345-diagnosis](ppo-train-345-diagnosis.md).

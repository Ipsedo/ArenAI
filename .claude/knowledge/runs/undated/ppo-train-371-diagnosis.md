---
name: ppo-train-371-diagnosis
description: "Runs 370/371 ppo_impala (optim batch 512) — critic repaired (ev 0.75-0.85) but actor immobile in substance: σ frozen at init (Hc ≈ 0.5, entropy 0.01), Hd = ln2, fire 0.85/s, KL budget burned on noise (clip 0.4, skip 0.6), hit ≈ init 0.02"
metadata:
  type: project
---

Runs train_370 (actor_epochs 4) and train_371 (actor_epochs 8), 2026-10-28, rollout 300, batch 512/128, lr 3e-4, entropy c 0.01 / d 0.001.

- ev 0.72-0.88 from 3.6k → the optim-batch split repaired the critic (ppo-full-batch-update).
- Actor: kl ≈ 0.04 (mean incl. 1st step at 0), skip 0.33 (370) / 0.62 (371), clip 0.35-0.44 → whole KL budget consumed every rollout; Hc 0.42-0.53 = init value (σ₀ 0.5), Hd 0.61-0.69 ≈ ln2, fire 0.85/s flat, hit 0.015-0.025 ≈ initial random level, D=250. 370 and 371 identical → epochs not the cause.
- Reward: r ≈ −0.0036/decision ⇒ ~1 death −1 per life; s ≈ 23.5 s = 20 shells / 0.85 fire/s → almost all lives end in out_of_shells suicide (before the 30 s starvation). High ev = mostly the shell countdown (observable), not tactical value.
- Aim signal per rollout: ~408 shots, ~8 hits (+0.2), aim ≈ 0.1·exp(−27 m / ~6 m) ≈ 0 for a typical shot.
- σ 0.5 on a velocity aim (180°/s full) ≈ ±3°/frame, ~±9°/decision of noise vs tolerance 2° → precision impossible; entropy 0.01 (which Claude had recommended) keeps σ at init for lack of a coherent policy gradient.
- Superseded by [ppo-train-378-diagnosis](ppo-train-378-diagnosis.md): s≈23.5 was not 20 shells / 0.85 fire/s but a fast dump + ~20 s wait.

See [ppo-train-369-diagnosis](ppo-train-369-diagnosis.md), [ppo-train-345-diagnosis](ppo-train-345-diagnosis.md), ppo-beta-entropy-init.

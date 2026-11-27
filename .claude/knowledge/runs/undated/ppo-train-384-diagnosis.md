---
name: ppo-train-384-diagnosis
description: "Run 384 (383 + free state-independent σ, actor lr 1e-4) — σ holds at 0.05 (Hc −1.58 → −1.70), skip ≤ 0.05, but marker missed: miss 23 m at 200k (∝ D), hit 0.06, kill 0.2, μ moves uniformly on the 4 axes; exploration noise was not the lock"
metadata:
  type: project
---

Run `outputs/train_384_ppo` (2026-10-31, launched 15h08, ~880 steps/min) = 383 + free log_sigma
per axis + actor_learning_rate 1e-4. Read at 76k steps (expected marker: miss < 15 m before 200k).

- 383's pathology is gone: Hc −1.577 → −1.607 (σ 0.05 → 0.048), skip = 0 until 70k
  (0.01 after), Hd 0.693 → 0.686 (Bernoulli head not saturated, unlike 382/383).
- But clip rises 0.13 → 0.30 and kl 0.008 → 0.023 in 40k: μ moves a lot per update
  (0.02 of μ = 0.4 σ) with no useful direction.
- miss 21.8 → 19.3 m at D=250 (24k) then 24.8 m as soon as the curriculum goes to 300 (35k, false
  positive hit 0.07 > 0.06): miss ∝ D = geometric floor, see ppo-actor-audit-2026-10-29.
- hit 0.07 → 0.053, kill 1.75 → 0.45, s 6-12 s (reserve emptied in ~10 s, fire 0.5-0.9/s).

**Why:** free σ neutralises the σ(state) lever well; the aim signal remains too weak for μ.

**How to apply:** if at 200k miss is not < 15 m, noise was not the main blocker;
go back to the dense oracle test proposed in ppo-actor-audit-2026-10-29.

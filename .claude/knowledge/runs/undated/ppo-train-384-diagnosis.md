---
name: ppo-train-384-diagnosis
description: "Run 384 (383 + σ libre indépendant de l état, lr acteur 1e-4) — σ tient à 0.05 (Hc −1.58 → −1.70), skip ≤ 0.05, mais marqueur raté : miss 23 m à 200k (∝ D), hit 0.06, kill 0.2, μ bouge uniformément sur les 4 axes ; le bruit d exploration n était pas le verrou"
metadata:
  type: project
---

Run `outputs/train_384_ppo` (2026-10-31, lancé 15h08, ~880 steps/min) = 383 + log_sigma libre
par axe + actor_learning_rate 1e-4. Lu à 76k steps (marqueur attendu : miss < 15 m avant 200k).

- La pathologie de 383 est partie : Hc −1.577 → −1.607 (σ 0.05 → 0.048), skip = 0 jusqu'à 70k
  (0.01 ensuite), Hd 0.693 → 0.686 (tête Bernoulli non saturée, contrairement à 382/383).
- Mais clip monte 0.13 → 0.30 et kl 0.008 → 0.023 en 40k : μ bouge beaucoup par update
  (0.02 de μ = 0.4 σ) sans direction utile.
- miss 21.8 → 19.3 m à D=250 (24k) puis 24.8 m dès que le curriculum passe à 300 (35k, faux
  positif hit 0.07 > 0.06) : miss ∝ D = plancher géométrique, cf. ppo-actor-audit-2026-10-29.
- hit 0.07 → 0.053, kill 1.75 → 0.45, s 6-12 s (réserve vidée en ~10 s, fire 0.5-0.9/s).

**Why:** σ libre neutralise bien le levier σ(état) ; le signal de visée reste trop faible pour μ.

**How to apply:** si à 200k miss n'est pas < 15 m, le bruit n'était pas le blocage principal ;
revenir au test d'oracle dense proposé dans ppo-actor-audit-2026-10-29.

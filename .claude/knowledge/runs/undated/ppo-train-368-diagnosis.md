---
name: ppo-train-368-diagnosis
description: "Run 368 ppo_impala rollout 30 décisions (480 échantillons/update), 1 pas Adam/epoch lr 3e-4 — KL explose (skip 0.7), critique constant (ev = 0.000), σ et Hd s'effondrent, hit 0.02 → 0.007"
metadata:
  type: project
---

Run train_368_ppo-impala (2026-10-28, 160k décisions en 6h45, ~2.5× plus lent que 364) — commits dd0487b6/6218d9f0 : rollout_size 900 → 30 (×16 tanks = 480 lignes), epochs actor 4 / critic 8, lr 3e-4, entropie c 0.001 / d 0, σ dépendant de l'état.

- 0-20k : sain (clip 0.05, kl 0.004, ev 0.4-0.55, Hc 0.3).
- dès ~20k : Hc → −1.0, Hd 0.69 → 0.12 ; kl 0.3-0.9 et skip 0.5-0.73 (max théorique 0.75 avec 4 epochs, epoch 0 jamais coupée) → un seul pas Adam suffit à dépasser 1.5×target_kl.
- ev collé à ±0.000 sur de longues plages = var(R−V) = var(R) ⇒ V quasi constant sur le minibatch : critique mort.
- hit 0.02 (début) → 0.007 en fin, kill → 0, curriculum figé D=250.

Cause retenue : 480 échantillons + 1 pas Adam à lr 3e-4 (pas ∝ lr indépendant du bruit) + normalisation des avantages sur un rollout de 3 s où la récompense éparse est souvent absente → gradient de politique ≈ bruit amplifié à variance 1 ; dérive de σ (dépendant de l'état, entropie 0.001) → KL hypersensible ((Δμ/σ)²). Horizon GAE 30 vs 1/(1−γλ)≈62 avec critique constant ⇒ pas de bootstrap utile.

Voir ppo-full-batch-update, [[ppo-train-364-diagnosis]], [[ppo-train-345-diagnosis]].

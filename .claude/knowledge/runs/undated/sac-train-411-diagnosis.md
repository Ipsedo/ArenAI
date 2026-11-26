---
name: sac-train-411-diagnosis
description: "Run SAC 411 (2026-11-08, 6 α, lr α 3e-5) à 33k pas : Q = valeur d'entropie pure (≈12, suit α), acteur au maximum d'entropie (Hc 2.73 = σ≈0.9 sur 4 axes, cible 0.51), visée au plancher aléatoire, fire 0.6→0.3/s à cause du cut sans bootstrap sur réserve vide"
metadata:
  node_type: memory
  type: project
  originSessionId: 51b10db1-fb50-4f0f-81d1-b1b5b214142f
  modified: 2026-11-08T17:31:10.689Z
---

Run `outputs/train_411_sac` (config : α₀ 0.1, lr α 3e-5, target_sigma 0.3, γ 0.99, batch 256, gradient_steps 1,
32 tanks, buffer 800k, image 128×64, D fixé à 250). Lu le 2026-11-08 à l'index 33k (≈1 M transitions, 33k updates, 75 min).

Constats :
- **Q ≈ 1 → 12.1 (pic à 17k) → 10.0 tandis que α 0.10 → 0.037** : Q n'est que le bonus d'entropie
  (α·H ≈ 0.1 × 3.5 nat/pas, asymptote 35) ; le reward moyen est −0.0003/pas. Q suit α, pas la tâche.
- **Hc = 2.73 plat** = maximum exact de 4 tanh-gaussiennes (0.683/axe à σ ≈ 0.9) ; cible 4 × 0.129 = 0.51.
  L'acteur ignore ∂Q/∂a ; hit 0.039 (< plancher aléatoire 0.05), miss 22.7 m plats depuis le début.
- **fire 0.6 → 0.3 tir/s monotone, kill 1.2 → 0.44, s 60 → 40 s** : la réserve vide est un cut sans bootstrap
  (commit 3a1cb788), donc tirer coûte ≈ γ·V ≈ 10 d'entropie future contre +1 par hit → le soft value
  agit comme bonus de survie anti-tir. Prédiction : fire → 0 comme au run 392 avant que α soit assez petit.
- Vitesse d'α : ÷e tous les ~33k pas (Adam, lr 3e-5) ; pour α·H ≈ échelle du reward (α ~ 3e-4) il faut ~190k pas (~7 h).
- UTD = 1 update / 32 transitions (gradient_steps 1 pour 32 tanks) ; buffer déjà rebouclé.

**Why:** Samuel veut juger si SAC débloque la visée après les plateaux PPO ; ici le verrou est l'échelle
entropie/reward, pas la perception.

**How to apply:** α₀ passé à 0.01 le 2026-11-08 (validé). Refusé : bootstrap sur réserve vide (PPO cut aussi, `ppo_trainer.cpp` cuts = dones + truncateds). Reste à valider :
gradient_steps 8 + batch 512 validés le 2026-11-08 soir (défauts, run 414) ; le bootstrap SAC seul reviendrait sur la table si fire → 0 se confirme.
Voir [[sac-agent-2026-11]], [[ppg-train-392-diagnosis]] (fire → 0), [[sac-train-309-diagnosis]] (α effondré, l'inverse).

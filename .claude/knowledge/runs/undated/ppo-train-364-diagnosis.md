---
name: ppo-train-364-diagnosis
description: "Run 364 ppo_impala (IMPALA CNN + SpatialSoftmax, minibatch 128) — ev 0.7 mais σ s'effondre sans gain de hit, 2.2× plus lent que vanilla 362"
metadata:
  node_type: memory
  type: project
  originSessionId: 890e3bc3-16ec-4388-85c1-e9912581e991
  modified: 2026-10-27T18:39:38.066Z
---

Run train_364_ppo-impala (2026-10-27, 328k steps en 5.5 h, arrêté) vs vanilla train_362_ppo aux mêmes steps :
- **ev ≈ 0.70 stable** (362 : 0.30-0.38) → le critique IMPALA est nettement meilleur.
- **Hc 0.55 → −0.25** dès 150k (362 : 0.36 à 330k) ; la chute coïncide avec un pic de hit 0.031-0.035 (160-260k) puis hit retombe à 0.021 en fin → σ se resserre sans progrès (engagement prématuré).
- hit/kill en retard sur 362 (0.030 vs 0.038-0.048 à 300-360k) ; curriculum jamais déclenché (D=250 ; 362 décolle ~580k).
- clip 0.03→0.07, kl ≤ 0.006 (≪ target 0.05) : pas d'instabilité d'optimisation.
- Débit : ~1h50 / 108k steps vs ~50 min pour 362 (convs stride 1 à pleine résolution 128×256).
- minibatch 128 (vs 512 en vanilla) = 4× plus de pas de gradient par rollout — suspect n°1 de la chute de σ.

Voir [[ppo-train-345-diagnosis]], ppo-vanilla-liquid-alignment, ppo-beta-entropy-init.

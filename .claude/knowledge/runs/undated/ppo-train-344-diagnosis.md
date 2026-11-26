---
name: ppo-train-344-diagnosis
description: "train_344 (ppo_liquid, facteurs d'entropie fixes 0.001, 256x128, 27.7M steps, 2026-10-11→16) — pic à 0.3-0.7M puis régression lente, curriculum redescend 550→350, Hc s'effondre sans plancher, Hd ≈ max"
metadata:
  node_type: memory
  type: project
  originSessionId: cf6947bf-0442-45af-815d-943234f1d08a
  modified: 2026-10-16T16:44:01.196Z
---

Run analysé le 2026-10-16 (terminé à 27.7M steps, 256 saves, save_k ≈ k×108k steps).
Config : ppo_liquid, truncated normal, `continuous/discrete_entropy_factor = 0.001` fixes
(PID Lagrangien retiré dans 00866e3d), aim reward, epochs 2, minibatch 2048, neuron_number 128.

- Pic hit (précision/épisode) 0.081, kill 4.3 à 0.3-0.7M (D 250-300) → save_3..6 meilleurs.
  Moins bon que le pic de [[ppo-train-339-diagnosis]] (0.095).
- Curriculum monte à D=550 (5.1M), y reste 11M, puis **redescend** 500 (16.2M) → 350 (26.8M).
  À D=350 en fin : kill 0.76 vs 3.9 au même D à 0.83M → vraie régression, pas un effet de D.
- Hc décroît sans arrêt 0.51 → −1.9 (σ moyen ≈ 0.04) : 0.001 ne retient rien. Hd monte
  0.62 → 0.686 (≈ ln2) : fire/zoom restent ~pile-ou-face, la tête discrète n'apprend pas.
- clip 0.2-0.27, kl ~0.02, skip à pics jusqu'à 0.7 ; ev 0.3-0.4 (pic isolé −137 à 5M).
- fire ≈ 1.07 obus/s constant, s ≈ 33 constant ; miss 50 → 29 (en partie effet D).

Lecture : facteurs fixes = pire que le plancher PID (Hc incontrôlé). Voir [[ppo-beta-entropy-init]].

Suite validée par Samuel (2026-10-16), **ppo_liquid uniquement** : σ indépendant de l'état
(paramètre `sigma_logit` [nb_continuous] → `SigmaOutput` log-sigmoïde [SIGMA_MIN, SIGMA_MAX],
init = `initial_sigma` 0.5, tête μ orthogonale ×0.01 inchangée) façon CleanRL/Andrychowicz 2021,
et facteurs d'entropie fixes 0.01 (standard Atari/Procgen, avantages normalisés par rollout).
Les anciens checkpoints ppo_liquid (tête `sigma.0.*`) ne se rechargent plus. Options écartées
pour l'instant : retour au PID, plancher σ par dim (DreamerV3), pool d'adversaires passés.

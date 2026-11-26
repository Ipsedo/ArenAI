---
name: ppo-train-369-diagnosis
description: "Run 369 ppo_impala rollout 300 + entropie 0.01/0.001 — régime inverse de 368 : politique figée (kl 3e-4, clip ≈ 0), Hc remonte, ev ≤ 0 ; aucun run full-batch (365-369) n'a de critique qui apprend, contrairement à 364"
metadata:
  type: project
---

Run train_369_ppo-impala (2026-10-28, 100k décisions en 3h36, commit aad2ab41) : rollout 300 (4 800 lignes), lr 3e-4, epochs 4/8, full-batch (1 pas Adam/epoch).

- clip 0.000-0.02, kl 0.0003-0.001 (cible 0.05), skip 0 → acteur quasi immobile ; Hc 0.56 → 0.66 (le bonus d'entropie 0.01 l'emporte), Hd 0.69 → 0.45.
- ev oscille entre −0.7 et +0.05 → critique pire qu'une constante ; hit plat 0.02, kill ~0.1, D=250.

Comparaison clé (ev final) : 364 (vrais minibatchs 128, 224 pas/rollout, lr 1e-4) **0.69** ; 365 full-batch lr 1e-3 −0.37 ; 367 critic 16 epochs −0.06 ; 368 0.000 ; 369 < 0. À 100k décisions : 364 ≈ 25k pas critique, 369 ≈ 2.7k (≈10× moins).

Cause retenue : le full-batch (ppo-full-batch-update) donne trop peu de pas d'optimiseur pour que le critique apprenne ; sans critique, les avantages = bruit MC → politique qui ne bouge pas (369) ou qui part en vrille (368). Proposition à Samuel : découpler taille du batch d'optimisation (ex. 1024 lignes, accumulées par blocs de 128) de la taille mémoire.

Voir [[ppo-train-368-diagnosis]], [[ppo-train-364-diagnosis]].

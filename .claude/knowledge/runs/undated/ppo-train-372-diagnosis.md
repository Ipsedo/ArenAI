---
name: ppo-train-372-diagnosis
description: run 372 impala (entropie c 0.001 / d 0, target_kl 0.02) — acteur calmé, σ baisse sans gain de hit, tir toujours aléatoire
metadata:
  type: project
---
Run 372 (2026-10-29, début ~39k décisions/tank ≈ 130 rollouts) = 371 + continuous_entropy 0.001, discrete 0, target_kl 0.02 (commits cd60f19b, 588b35c6 de Samuel).
- clip 0.21-0.28 (vs 0.4 en 371), kl ≈ 0.018 ≈ cible, skip 0.70 (≈ 2.4 epochs utiles sur 8).
- Hc 0.57 → 0.20-0.30 : σ diminue cette fois, mais hit 0.047 → 0.012-0.02 (niveau aléatoire), miss 27 m inchangé.
- Hd ≈ ln 2 : tir/zoom toujours pile ou face ; fire ~0.85/s, s ≈ 24-25 s = toujours suicide out_of_shells.
- ev 0.70-0.86.
Lecture : les hyperparams ont réglé l'agitation de l'acteur mais pas le manque de signal (visée récompensée ≈ 0, −1 de suicide dominant) ; risque de collapse de σ sans compétence comme en 368. Voir [[ppo-train-371-diagnosis]].

Suite décidée 2026-10-29 (non committée au moment de l'écriture) : visée A3 = 0.5·exp(−m/tol) + 0.5·exp(−m/tol_large), tol_large = 10 m + 6°·d ; out_of_shells reste une terminaison mais coûte −0.3 (B2), retourné/combat restent −1. Option refusée/non retenue : coût par tir (B1), shaping dense (A4).

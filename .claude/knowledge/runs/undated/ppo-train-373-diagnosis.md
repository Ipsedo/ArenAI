---
name: ppo-train-373-diagnosis
description: run 373 impala = 372 + visée A3 + out_of_shells −0.3 — miss toujours 28-30 m pendant que σ s'effondre (Hc → −0.3) : la moyenne de la visée n'apprend rien
metadata:
  type: project
---
Run 373 (2026-10-29, ~194k décisions/tank, binaire rebâti après A3+B2) : r_aim ×2 (0.0012/décision, ≈ 0.014 par tir, comme prévu), r ≈ −0.0003.
- miss 27 → 30 m, hit 0.006-0.02 (aléatoire), fire 0.85/s, s ≈ 23 s (toujours vidage des obus), D=250, ev 0.7-0.88.
- Hc 0.38 → −0.3 (σ ≈ 0.5 → 0.18) sans baisse du miss : 3e fois que le σ dépendant de l'état d'impala s'effondre sans compétence (364, 368, 373).
- Hd 0.69 → 0.55-0.6, sans effet.
Lecture : l'écart de récompense entre un bon et un mauvais tir (≈ 0.03) reste ≈ 10× plus petit que les événements aléatoires ±0.3 (touche reçue, mort) ; avantage normalisé ≈ 0.1 σ, dilué sur ~20 décisions de vol d'obus. Pistes : hit_reward_scale 0.1 → ~1, run de contrôle ppo/ppo_liquid (σ indépendant) à config identique. Voir [[ppo-train-372-diagnosis]].

2026-10-29 : vérifié dans le code — ppo, ppo_liquid, ppo_liquid_impala ont tous un σ dépendant de l'état (Linear + SigmaOutput). Samuel a fait passer ppo_impala seul à un `log_sigma` paramètre libre par action (clampé [1e-4, 1]), pour tester l'hypothèse de l'effondrement.

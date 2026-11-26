---
name: ppo-actor-audit-2026-10-29
description: audit complet de la chaîne acteur PPO impala (run 375) — aucun bug trouvé ; le signal de visée est sous le bruit (plancher géométrique du miss ≈ 27 m = demi-espacement des tanks)
metadata:
  type: project
---
Audit 2026-10-29 (run 375 = 373 + log_sigma libre impala ; identique à 373 sur 43k décisions) :
vérifiés OK — act(sample=true) à l'entraînement, log-pdf/sample/entropie de la normale tronquée cohérents, masque fire identique à la collecte et à l'update, GAE (truncation bootstrap sur sa propre valeur), ReturnScaler, normalisation des avantages sur valids, buffer (valids/bootstrap/finish_episode), mapping actions → right_joystick (delta par frame, joystick tenu 3 frames), tir (will_fire → obus 250 m/s), caméra = 20 m derrière le canon, SpatialSoftmax (x, y, présence par canal).
Constat clé : 16 tanks dans 250×250 → ennemi le plus proche ≈ 30 m ; miss ≈ 27-30 m = plancher géométrique (l'ennemi le plus proche de n'importe quel point d'impact est à ~27 m) : r_aim mesure la géométrie, pas la visée. Écart bon/mauvais tir ≈ 0.03 vs événements ±0.3 → ≈ 0.1 σ d'avantage, dilué sur les décisions de visée précédentes. clip 0.25 / KL à la cible = l'acteur suit le bruit de chaque rollout.
Proposition : test d'oracle temporaire (récompense dense −erreur angulaire canon→ennemi, non committé) pour trancher machinerie vs signal ; sinon hit_reward_scale ×10 et/ou plus de données par update. Voir [[ppo-train-373-diagnosis]].

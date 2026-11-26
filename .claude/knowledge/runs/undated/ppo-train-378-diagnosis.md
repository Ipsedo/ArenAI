---
name: ppo-train-378-diagnosis
description: "run 378 ppo (32 tanks, 10 obus, regen 5) — hit = niveau aléatoire ; les obus traversent le terrain (Jolt discret à 250 m/s) et volent 20 s, le tank vit donc ~18 s sans obus ; P(fire) appris 0.98, metrics fire/Hd aveugles"
metadata:
  node_type: memory
  type: project
  originSessionId: 8911f436-4c10-46c0-b776-0ffffe091aef
  modified: 2026-10-30T07:39:25.011Z
---

Mesuré le 2026-10-30 sur save_0 / save_3 (6 épisodes chacun, harnais temporaire hors dépôt) :
- hit/fire 0.049 (init) vs 0.045 (final), miss 16.7 vs 17 m : aucune compétence ; D figé à 250 car la baseline aléatoire (~0.05) tombe entre curriculum_ratio_low et high.
- Vie ≈ 20-22 s dont dernier tir à 2.5-3.7 s ; 83-87 % des décisions avec réserve vide ; médiane du temps après le dernier tir = 20.0 s = durée de vie de ShellItem.
- ~82 % de la récompense de visée arrive après 19 s (hors de portée de GAE).
- Sonde terrain : à 250 m/s, contact 39 % à 0°, 18 % à −5°, 2 % à −45° (MotionQuality Discrete par défaut, mesh à une face, 8.3 m/frame).
- P(fire|can_fire) 0.5 → 0.98 : chaque tir rapporte ≈ +0.035 sans coût ; initial_fire_proba n'est pas la cause.
- Metrics trompeuses : fire plafonné par la réserve, Hd dominé par le zoom sur les lignes réserve vide, Hc moyenne des σ (canon_x 0.03, canon_y 0.92).
- La mort sur réserve vide attend `!shell_in_flight` depuis ca4583a5 (2026-10-25) : tous les runs depuis sont concernés. Corrige [[ppo-train-371-diagnosis]] : s≈23.5 n'était pas 20 obus / 0.85 tir/s mais un vidage rapide + ~20 s d'attente.
Voir [[ppo-actor-audit-2026-10-29]], [[jolt-migration-fidelity]].

2026-10-30 : LinearCast sur ShellItem commité (7ed8a411). Mesure à la politique d'init : terrain 100 % de contact en sonde, mais temps après dernier tir encore 11.8 s (médiane 13.6), 22 % de la récompense de visée après 19 s, réserve vide 73 %, hit 0.068. Durée de vie 4 s non appliquée. Canon baissé à fond = obus créé sous le sol plat (spawn 10 m devant le canon). Le lecteur de ressources des tests model ne lit pas les PNG ; sur cube, pas de tunneling même en Discrete.

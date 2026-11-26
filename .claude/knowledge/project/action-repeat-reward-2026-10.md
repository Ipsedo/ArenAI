---
name: action-repeat-reward-2026-10
description: 2026-10-21 — action repeat (décision 10 Hz), récompense de visée 3D exponentielle, normalisation des récompenses par std glissante du retour ; décisions de conception validées par Samuel
metadata:
  type: project
---

Implémenté le 2026-10-21 (non commité à la fin de la session), validé par Samuel point par point :

- **Action repeat** `--action_repeat` (défaut 3 → décision 10 Hz, physique 30 Hz). `BaseTanksEnvironment`
  découpé en briques protégées `advance_frame` / `render_frame(with_visions)` / `tick_frame` /
  `read_states` (ordre de `step()` inchangé). Tir = impulsion sur la 1re frame seulement
  (`core::release_fire`), cohérent avec le fire_mask. Événements has_hit/kill/fire consommés à
  chaque frame (sinon tick() rajoute du temps de famine en boucle). `max_episode_steps` en frames.
  Desktop : `DesktopGameEnvironment::play_frame(dt, actions, observe)`, `game.cpp` décide toutes
  les k frames ; k lu par `AgentFactory::get_action_repeat()` depuis `config.environment`,
  **défaut 1 pour les vieux configs** (choix de Samuel).
- **Défauts rééchelonnés** (vanilla + liquid) : γ 0.991 (=0.997³), λ 0.97, liquid `delta_t` 1/10,
  chunk_size gardé à 30 décisions (3 s), rollout 900 décisions.
- **Visée** : `compute_aim_reward` = exp(−‖shell−enemy‖ / (3.25 + tan-free 2°·d)) en 3D
  (sphères iso-récompense) ; Samuel a refusé une version 2D (repère plan).
- **ReturnScaler** (`agents/return_scaler.*`) : r / std glissante du retour actualisé, sans
  soustraction de moyenne, dans compute_gae des deux trainers ; sauvé en `return_scaler.pt`.

**How to apply :** la métrique `r` est désormais par décision (≈3× l'ancienne échelle) ; les runs
≤ 348 ne sont pas comparables en r/γ. Pas de test unitaire dédié à action_repeat > 1 ni au scaler.

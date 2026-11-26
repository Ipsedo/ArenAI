---
name: hit-direction-proprioception
description: "2026-11-21 direction du tireur ajoutée à la proprioception (3 flottants, décroissance 2 s) ; checkpoints antérieurs incompatibles"
metadata:
  node_type: memory
  type: project
  originSessionId: 6339cbc0-bfa6-437b-9c0d-21957afa72da
  modified: 2026-11-21T11:54:59.336Z
---

Validé par Samuel le 2026-11-21 : la proprioception se termine par la direction du dernier tireur
(position de tir de l'obus − châssis, repère châssis, vecteur unitaire × intensité linéaire 1 → 0
en `HIT_DIRECTION_SECONDS` = 2 s, nul sans touche récente). `ENEMY_PROPRIOCEPTION_SIZE` 105 → 108.

**Why:** l'agent ne se retournait pas quand on lui tirait dans le dos ; l'observation ne contenait
que des ratios de vie, sans événement ni direction. Équivalent de l'arc de dégâts du HUD joueur,
donc pas un signal training-only ([[proprioception-archi-boundary]]).

**How to apply:** tout checkpoint antérieur au 2026-11-21 (dont le modèle du jeu du commit
4b0a2bde) ne se charge plus ; choix assumé « casser et réentraîner », pas de notebook de
conversion. La durée de 2 s est un a priori non mesuré.

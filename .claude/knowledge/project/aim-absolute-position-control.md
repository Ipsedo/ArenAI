---
name: aim-absolute-position-control
description: "2026-09-23 — la visée passe de commande en vitesse à consigne absolue normalisée portée par TurretItem/CanonItem ; cause racine de l'oscillation du canon"
metadata:
  node_type: memory
  type: project
  originSessionId: 8bc61c6a-313f-4e04-9392-f796e36e3a71
  modified: 2026-09-22T22:41:20.337Z
---

Symptôme (2026-09-22, checkpoint ppo_liquid_409_save_53) : en jeu l'agent visait et tirait, mais le canon oscillait lentement (période ~0.5–1 s) sans jamais se stabiliser, même sur une cible immobile.

**Cause racine** : `controller::user_input::right_joystick` transportait un **delta angulaire par frame**, intégré par les parts (`angle += ...`). Une commande en vitesse ne peut tenir une visée qu'en émettant un zéro exact — impossible pour une tête `Linear → Sigmoid`. S'y ajoutaient : aucune récompense pour l'immobilité (reward purement événementielle, cf. [[ppo-reward-redesign-2026-08]]), et un retard de boucle (1 frame + moteur Jolt).

Ce n'était **pas** le bruit d'échantillonnage : en jeu `act()` utilise le mode, pas un échantillon.

Deux correctifs tentés, dans cet ordre :

1. **2026-09-22 (commit `44ed1f79`), puis REVERTÉ le 2026-09-23** — moteur de tourelle/canon passé à des `MotorSettings` explicites 8 Hz / damping 1.0 au lieu du défaut Jolt (ressort 2 Hz, ~10 frames pour rejoindre la consigne). Palliatif sur le plant, jamais un fix. Une fois le correctif 2 en place la justification tombe : avec une consigne absolue le point fixe est l'angle visé quel que soit le retard, le retard ne fait plus dépasser. **Décision Samuel : retour au défaut Jolt, ne pas re-proposer de raidir le moteur.** Le seul bénéfice résiduel était l'erreur de poursuite (un 2e ordre amorti critique suivant une rampe traîne de `2v/ω` : 28.6° à 2 Hz contre 7.2° à 8 Hz à 180°/s, et l'obus part le long du canon physique) — jugé non prioritaire tant que l'agent n'a pas réappris à viser.
2. **2026-09-23** — `right_joystick` devient une **consigne absolue normalisée dans [-1,1]** : `.x` → lacet sur [-π, π] (enroulement, plus court chemin via `std::remainder`), `.y` → site sur [-0.2π, 0.2π] (butée). `TurretItem`/`CanonItem` portent le mapping, le limiteur de vitesse et la butée. `EnemyControllerHandler` supprimé. Les handlers joueur intègrent leur delta en unités normalisées (souris 0.4/0.8, manette 0.02/0.04) — le stick reste une vitesse côté joueur.

**Pourquoi la position plutôt que la vitesse :** une erreur de politique sur une vitesse s'intègre sans borne (biais 0.01 → 1.8°/s pour toujours) ; sur une position elle est bornée par elle-même (1.8° d'erreur statique). Ça transforme un problème de stabilité, que l'objectif ne mesure pas, en problème de précision, que l'apprentissage sait résoudre. Bonus : le crédit devient quasi local (l'angle au tir dépend de l'action du pas, pas de toute la séquence) et le bruit d'exploration cesse de dériver.

**Décisions de conception (Samuel) :** limiteur de vitesse en paramètre par tank (ennemi bridé, joueur `UNLIMITED_RADIAL_VELOCITY` pour garder le flick souris) ; lacet en absolu ±π avec plus court chemin plutôt que sin/cos ; conversion côté agent puis descendue dans les parts.

**How to apply :** deux pièges à ne jamais réintroduire. (a) Un handler qui émet `0` sur un événement non-visée commande désormais « recentre plein axe » — les branches bouton et curseur-relâché doivent réémettre la consigne tenue (testé dans `PlayerAimTest.ButtonEventKeepsTheHeldAim`). (b) Le `0.4f` du canon faisait deux métiers, gain joystick **et** rapport de vitesse 0.4× face à la tourelle ; d'où `ENEMY_CANON_RADIAL_VELOCITY`. Conséquence directe sur les cibles d'entropie : voir [[ppo-beta-entropy-init]]. Les checkpoints antérieurs sont inexploitables sur les dims de visée.

**Mise à jour 2026-09-24 : REVERTÉ** par Samuel (commit bb66d985 « backtrack: delta for turret
control ») — la visée est de nouveau une commande en delta ; la truncated normal est revenue
pour PPO vanilla et ppo_liquid (1cd80e0d/6402913c). Les cibles d'entropie de ppo_liquid sont
revenues à `{-0.2, -0.2, -0.88, -0.88}`.

**2026-11-11 : réintroduite** (c02137c6) sous une autre forme : action `aim` séparée (yaw_cos, yaw_sin, pitch) côté agent seulement, joueur en delta ; 1er run = [[ppg-train-421-diagnosis]].

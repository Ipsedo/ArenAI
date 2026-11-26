---
name: controls-submenu-design
description: "Décisions validées du sous-menu Contrôles (2026-08-27) — 1 slot de tir, manette tout remappable, conflit = désassignation, B/Start réservés"
metadata:
  node_type: memory
  type: project
  originSessionId: e89cc127-e6a3-4a37-a28b-4b186cf25264
  modified: 2026-08-26T22:20:07.148Z
---

Sous-menu Contrôles (`controls.rml`, validé par Samuel le 2026-08-27, maquettes OK) :

- **1 seul slot de tir** clavier et manette (le clic gauche ne tire plus par défaut, il ne garde que la recapture du curseur ; défaut = Space / RB).
- **Manette tout remappable** : fire (bouton) + 5 slots d'axes (steer, aim x/y, accelerate, reverse). Accel/reverse capturent une direction (signe) ; un axe doit repasser au repos (<0.3) avant d'être capturable (>0.6).
- **Conflit = désassignation** de l'ancienne action (état UNBOUND ambre #F5B356), jamais de swap ni de popup.
- **Annulation de capture : Escape seul, ou timeout 15 s** (B est bindable — refusé comme cancel le 2026-08-27) ; **Start réservé** (toggle pause du GameInputRouter) ; Escape/visée souris fixes, affichés grisés.
- Choix du device par GUID (persisté), fallback première manette ; `list_gamepads`/`select_gamepad`/`key_label` ajoutés à `AbstractWindow` (validé, pas de leak GLFW).
- `controller::Key` étendu (~80 touches, plages A..Z/0..9/F1..F12 contiguës pour conversion arithmétique) ; noms canoniques dans `arenai_controller/input_names.h`. Labels layout-aware via `glfwGetKeyName` (Samuel est en AZERTY).
- Effet de bord assumé : deadzone 0.05 désormais appliquée aussi aux gâchettes (avant : brutes).

Voir [[menu-styling-validation]] pour la charte.

---
name: zoom-action-bernoulli-head
description: "2026-09-18 — action zoom (FOV ×4, 0.25 s) + head discrète sigmoid/Bernoulli (fire, zoom) ; SAC supprimé du code"
metadata:
  node_type: memory
  type: project
  originSessionId: 9df2f44c-87d4-4707-8362-728decdf55df
  modified: 2026-09-18T12:48:40.771Z
---

2026-09-18 : ajout de l'action discrète **zoom** (joueur : clic droit / LB manette, IA : neurone 2).

- Head discrète des actors = **Sigmoid** (Bernoulli indépendantes par action), plus de softmax ;
  encodage binaire `[fire, zoom]`, seuil strict > 0.5 en mode max_action.
  Distribution dans `arenai_agent/src/distributions/bernoulli.{h,cpp}` ; clamp haut à `1 - 1e-6`
  (en float32, `1 - 1e-8` s'arrondit à 1 → log(1-p) = -inf quand la sigmoid sature).
- Alpha discret (PID Lagrangien) passé à **1 dim par action**, cible = `factor × ln 2` par action.
- FOV : `AbstractCamera::fov()` (défaut π/4 dans `arenai_view/include/arenai_view/camera.h`),
  lissé dans `CanonItem::fov()` (constantes `ZOOM_MAGNIFICATION = 4`, `ZOOM_TRANSITION_SECONDS
  = 0.25` dans `arenai_model/include/arenai_model/constants.h`) ; le renderer lit `camera_->fov()`
  au lieu du π/4 codé en dur. FOV de base inchangé, mais le look-at (ci-dessous) a changé la vue
  → golden images régénérées.
- Hyperparam `--initial_zoom_proba` (défaut 0.25) à côté de `--initial_fire_proba`.
- **Anciens checkpoints incompatibles sémantiquement** : le neurone 2 signifiait « ne pas tirer »,
  il signifie « zoomer » — les modèles commités dans `resources/trained_models` sont à ré-entraîner.
- **SAC supprimé du code** à la demande de Samuel (2026-09-18) : dossier `agents/sac/`, `QFunction`,
  `multinomial`, classes `AlphaParameters`/`ConstantTargetEntropy`, entrée menu IA + enum. Les
  diagnostics [[sac-train-293-diagnosis]] / [[sac-train-309-diagnosis]] restent de l'historique.
- Visée : `CanonItem::look()` pointe le point à `CANON_AIM_DISTANCE` (100 m) sur l'axe du
  canon — le même que le réticule HUD (constante partagée) — pour que le zoom converge sur
  l'impact ; `AbstractCamera::pivot()` (défaut = look()) garde le pivot du bras-ressort de
  `CollisionCamera` près du canon. Goldens régénérés (attention : premier run docker
  incrémental avait testé un binaire pas relinké — relancer le script en cas de goldens
  inchangés suspects).

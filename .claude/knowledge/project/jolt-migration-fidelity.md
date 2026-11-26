---
name: jolt-migration-fidelity
description: "Bullet→Jolt (2026-07-30) — goldens sensibles à la physique; technique harnais de mesures + worktree pour valider une migration à l'identique"
metadata:
  node_type: memory
  type: project
  originSessionId: 451a5e50-98f5-4d0d-9d28-a043f8739060
---

Migration Bullet → Jolt v5.3.0 faite le 2026-07-30 (interface publique arenai_model intacte).

**Ce qui n'est pas évident depuis le code :**
- Les golden images (agent/core/desktop) encodent les **poses physiques** au pixel près, pas
  seulement le rendu : tout changement de moteur/solveur les casse (suite view = rendu pur, seule
  à survivre). Régénération via `sg docker -c "./scripts/goldens_docker.sh --regenerate"`.
- Technique validée pour "reproduire à l'identique" : harnais de mesures gtest temporaire
  (API publique uniquement, sortie JSON via env var) exécuté AVANT migration, puis dans un
  `git worktree` à HEAD pour re-mesurer l'ancien moteur après coup. A permis de découvrir que
  Bullet ne plafonnait pas la vitesse (les roues patinent, traction limitée par friction) et le
  mode de combinaison du frottement (multiplication).
- Écarts résiduels irréductibles Bullet→Jolt : marge de collision 4 cm (objets ~3-10 cm plus
  hauts sous Bullet), drift ~0.8° de la tourelle libre pendant le settle. [[dev-environment]]

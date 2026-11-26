---
name: ppo-train-389-diagnosis
description: "Run 389 (388 + recharge +10 obus/coup, commit dba4e002) — lu à 202k steps : kill 1.5-2.1 contre 0.8-1.7 à D égal, tirs par vie ~14 contre ~10, hit tenu mais miss inchangé ; gain de volume, pas de visée ; creux de 388 à 237k pas encore atteints"
metadata:
  node_type: memory
  type: project
  originSessionId: f4f60a9b-baf5-487c-aea1-2d1d583ff538
  modified: 2026-11-01T12:59:10.303Z
---

Run `outputs/train_389_ppo` (lancé 2026-11-01 10h15, lu à 202k steps) = [[ppo-train-388-diagnosis]]
avec `shells_recharged_per_hit` 5 → 10. Samuel a décidé de le laisser continuer.

- Par D (389 / 388) : kill 1.88/1.29 à D=300, 2.06/1.63 à 350, 1.78/1.68 à 400, 1.79/0.74 à 450,
  1.52/0.78 à 500. 388 est moyenné sur ses 300 premiers k steps, creux de 237k inclus à 450-500.
- Tirs par vie ~14 contre ~10, s 12-14 contre 10-12, fire 1.1-1.2 contre 0.9-1.0.
- hit 0.09 contre 0.10 à D=300-400, pas de retombée vers le plancher aléatoire. miss non amélioré.
- Hc identique à 388 (σ baisse pareil), skip 0.045 à 194k contre 0.058, clip 0.34, ev ~0.6.

**Why:** la recharge fait monter les kills par le volume de tir ; la visée n'a pas progressé.

**How to apply:** le test décisif est le passage de 237k-490k (creux de `fire`/`hit` sur 388) et un
éventuel hit > 0.10 à D ≥ 500, le seuil d'autonomie de la réserve.

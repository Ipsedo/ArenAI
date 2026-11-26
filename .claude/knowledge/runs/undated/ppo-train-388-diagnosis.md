---
name: ppo-train-388-diagnosis
description: "Run 388 (γ 0.99, λ 0.95, σ₀ 0.3, comparé à 385) — gain réel : hit 0.10 à D=400, D monte à 550, ev 0.57 ; kill plat car le curriculum régule le hit et kill ∝ hit³ (5 coups sur la même pièce pour tuer) ; creux périodiques fire/hit ; σ baisse et skip monte à 0.33"
metadata:
  node_type: memory
  type: project
  originSessionId: f4f60a9b-baf5-487c-aea1-2d1d583ff538
  modified: 2026-11-01T09:11:47.194Z
---

Run `outputs/train_388_ppo` (lancé 2026-11-01 00h07, lu à 502k steps, toujours en cours) =
385 avec γ 0.994 → 0.99, λ 0.99 → 0.95 et aussi σ₀ 0.05 → 0.3 (donc gain non attribuable à γ/λ seuls).

- Gain : hit 0.10 à D=350-400 contre 0.055 à D=350 pour 385, D atteint 550, ev 0.25 → 0.57.
- `kill` = total des 32 tanks par épisode (moyenne de 16 épisodes), ≈ 1.5/31 au mieux. Ne se lit pas
  seul : le curriculum relève D dès que hit sur les épisodes probe > 0.06, ce qui ramène le hit.
- kill ∝ hit^3 environ : 0.07 à hit 0.04, 1.5 à hit 0.10 (corrélation 0.89), à cause des PV : obus
  = 1 dégât, roues/tourelle/canon 5 PV, châssis 10, une pièce à 0 tue. Il faut ~24 coups par kill à hit 0.10.
- Recharge +5 obus/coup : réserve autonome seulement à hit > 0.20 ; vies ≈ 10-14 s, ~10 tirs.
- Creux à 237k, 356k, 490k : fire → 0.5, hit 0.03-0.045, kill ≈ 0. Cause non établie.
- σ 0.3 → 0.15 (Hc −0.49), skip 0 → 0.33, clip 0.36 : pente vue avant les régressions de [[ppo-train-383-diagnosis]].

- Comparé à [[ppo-train-345-diagnosis]] (regen 1 obus/1.5 s jusqu'au 2026-10-23, 16 tanks) : même
  hit ~0.08, mais ~38 tirs par vie contre ~9, donc ~2.9 coups par tank contre ~0.7, et kill 4.4 contre
  0.77. La regen faisait monter les kills par le volume de tir, pas par la visée (hit 345 plat).
- Options proposées le 2026-11-01, non validées : recharge +10/coup (autonomie à hit > 0.10, au-dessus
  du plancher aléatoire ; +15 = 6.7 % ≈ plancher), PV des pièces réduits, retour de la regen.

**Why:** kill plat ne prouve pas l'absence d'apprentissage, mais le gain à D fixe n'est pas mesuré.

**How to apply:** juger sur un run témoin à D fixe (final_spawn = initial_spawn désactive le
curriculum) et non sur `kill` à D mobile. Tout changement de PV/curriculum : à valider par Samuel.

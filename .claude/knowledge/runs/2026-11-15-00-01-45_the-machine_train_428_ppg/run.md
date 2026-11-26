# train_428_ppg

| | |
|---|---|
| machine | the-machine |
| utilisateur | samuel |
| lancement | 2026-11-15 00:01:45 |
| commit au lancement | `0e5093a1` — backtrack higher resolution : real problem is in discrete target entropy value |
| commits ≤ 90 min après | aucun |
| diff au lancement | inconnu (identité relevée après la fin du run) |
| données brutes | `outputs/train_428_ppg` sur the-machine |

## Faits établis

_Run 428 PPG (0e5093a1, vs 427 : discrete_target_proba 0.05→0.01 + γ 0.998, confondus) — α_fire/zoom au plancher tout le run ; bascule 560-620k : fire 0.9→0.2-0.4/s, hit 0.17→0.26, kill 22→14-17 ; sonde save_27 : ratio 8.6/21.9 mais p_al divisé par 2, déterministe muet_

Run 428 PPG lancé 2026-11-15 00:01 sur 0e5093a1 (confiance haute), analysé à 739k (en cours).
Vs 427 (d1bb4020) : discrete_target_proba 0.05 → 0.01 et gamma 0.995 → 0.998, **confondus** ; ev désormais par batch d'optim.
Jalons D identiques à l'index près de 424 à 428 (curriculum à vitesse max) → comparer à D=1000 à index égal.

- α_fire/α_zoom au plancher 1e-4 dès 64k (427 : α_fire jusqu'à 0.017) → contrainte discrète inerte.
- 640-740k vs 423/427 : hit 0.232 (vs 0.20/0.15), miss 54 m (vs 67/89), fire 0.44 (vs 0.83/1.09), kill 16.9 (vs 23).
- Sonde save_27 (729k) : ratio 8.6 / 21.9 (423 : 4.9 / 7.5), p_off 0.017 / 0.007, p_al 0.15 (423 : 0.36 / 0.29),
  hit/obus 0.27, h<5° inchangé 0.35, tirs alignés 65 % (vs 47-50 %), %vident 0.49 / 0.39 (vs 0.70), désarmé 1-3 %,
  déterministe : tire dans 4-7 % des états alignés.

**Why:** gain de hit = sélection des tirs (hors axe éteint, dump supprimé), pas balistique ; gâchette globalement basse → moins de kills.
**How to apply:** ne pas lire la hausse de hit comme une meilleure visée ; surveiller p_al et fire (< 0.15/s = gâchette éteinte) ; attribution γ vs cible discrète nécessite un run séparé. Voir run 423.

**Sonde save_49 (1323k, 2026-11-15)** : h<5° 0.43 / 0.47 (1er gain de précision par tir aligné de la lignée, ≤ 0.37 avant), tour<5° 0.375 / 0.332,
hit/obus 0.29 ; zoom abandonné (p_zoom 0.02-0.03, Hd CSV 0.07 dès 1.18M, α_zoom ≈ plancher) ; châssis braqué d'un côté (μ dir_x +1.05)
→ tourne en rond ; ratio 7.0 / 11.1 (fond hors axe plat 0.02-0.04 au-delà de 10°), déterministe toujours muet (3-7 %).

# Sonde run 420 (`train_20_sac`) — save_1

- Index 27k, D = 250 à la save. Run en cours (encodeur d'état partagé, aed53757).
- Binaire `build/arenai_agent/arenai_agent_probe` recompilé à 15:49 depuis aed53757 (arbre propre) :
  aucun diff d'inférence. Le binaire `build/` de 14:20 contenait déjà `StateEncoder`, mais le script le
  datait au commit précédent ; j'ai recompilé pour lever le doute.
- Côtés 250 et 550, 4 épisodes chacun, stochastique, CUDA (GPU partagé avec l'entraînement).

## Résumé

| côté | ratio | p_al | p_off | p_arm | hit/ob | h<5° | tirs<5° | enrich | vide_s | %vident | %désarmé | tir/s | tour<5° | σx | σy | |μ|cx | μsat dir |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 250 | 2.19 | 0.63 (n 299) | 0.29 | 0.36 | 0.087 | 0.163 | 0.113 | 1.49 | 3.6 | 0.99 | 0.73 | 0.67 | 0.089 | 0.48 | 0.43 | 0.65 | 0.25 / 0.29 |
| 550 | 2.44 | 0.54 (n 94) | 0.22 | 0.27 | 0.027 | 0.061 | 0.126 | 1.53 | 4.1 | 1.00 | 0.70 | 0.61 | 0.085 | 0.50 | 0.45 | 0.61 | 0.23 / 0.28 |

p_fire armé par angle (côté 250) : 0.50 / 0.52 / 0.49 / 0.41 / 0.31 / 0.24 pour
[0-2[, [2-5[, [5-10[, [10-20[, [20-45[, [45-180[ degrés.

## ∂Q/∂a_canon (qgrad, comparable à 417/419 save_1)

| côté | P(bon sens) lacet armé | P(bon sens) site armé | P par tranche de lacet | |∂Q/∂a| armé | accord signe Q1/Q2 (armé, \|yaw\| < 10°) |
|---|---|---|---|---|---|
| 250 | 0.52 | 0.51 | 0.50–0.51 | 3.3e-3 | 0.49 (x), 0.45 (y) ; corr 0.01 / −0.06 |
| 550 | 0.50 | 0.51 | — | 2.5e-3 | — |

corr(μ, consigne utile) : +0.05 (lacet), +0.11 (site). P(μ bon sens) 0.50–0.53.

## Lecture

- **Gâchette sélective** : p_fire décroît de façon monotone avec l'angle de la cible, aux deux côtés
  (ratio 2.2 / 2.4). Les obus tirés sous 5° touchent 3,5× plus que ceux tirés au-delà de 20° (côté 250).
  Le canal discret (Q(s, ·) à K sorties, espérance exacte) a appris à quel moment tirer.
- **Visée continue absente** : ∂Q/∂a_canon reste sans direction (0.50–0.52), les jumeaux sont en
  désaccord de signe, μ canon n'est pas corrélé à l'erreur. Tourelle sous 5° : 8.5–9 % des décisions.
  C'est le même tableau qu'en 417 et 419 au même stade : l'encodeur partagé n'a pas changé ce verrou.
- **Dump** : 99–100 % des tank-épisodes vident leur réserve, médiane 3.6–4.1 s, 70–73 % des décisions
  désarmées. Le niveau de base de la gâchette reste élevé (0.24 hors axe).
- **Têtes** : σ canon 0.43–0.50, jamais au plafond, très au-dessus de la cible 0.1 ; μ direction
  saturé 23–29 % du temps (c'est là que se réalise le Hc < −5 du CSV).

## Établi / bruit

- Établi : canal discret sélectif ; ∂Q/∂a continu sans direction ; dump inchangé.
- Non tranché : la sélectivité est-elle due à l'encodeur partagé ? Aucune sonde 417/419 avec ce script
  (pas de `ratio` de référence au même stade). Effectifs « aligné & proche » faibles (299 / 94).

## Suite utile

- Sonder 419 save_1 côté 250 avec ce script pour avoir la référence de la gâchette.
- save_2/save_3 de 420 : `ratio` et P(∂Q/∂a bon sens) armé.
- `--features` sur save_1 : le gisement est-il codé par l'encodeur partagé (419 : R² ≤ 0.16) ?

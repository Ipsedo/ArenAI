# Sonde run 423 (PPG) — save_21

- Index 567k, D = 1000 (borne finale atteinte à 429k). Run en cours, code 802b1b32.
- Binaire `cmake-build-release/arenai_agent/arenai_agent_probe` recompilé 2026-11-13 08:25 sur 802b1b32
  + support PPG ajouté à `probe.cpp` (branche `DistillationActor`, non commité) : aucun diff d'inférence.
  L'ancien binaire (11/11, aed53757) ignorait le gun laying et ne savait pas charger un acteur PPG.
- Côtés 550 et 1000, 4 épisodes chacun, stochastique, CUDA (GPU partagé avec l'entraînement).
- Pas de ∂Q/∂a ni de features : l'acteur PPG n'a pas de critique Q et n'expose pas son encodage.

## Résumé

| côté | ratio | p_al (n) | p_off | p_arm | hit/ob | h<5° | tirs<5° | enrich | vide_s | %vident | %désarmé | tir/s | tour<5° | σx | σy | \|μ\|cx | μsat dir_x | p_zoom |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 550 | 4.86 | 0.359 (836) | 0.074 | 0.139 | 0.229 | 0.369 | 0.500 | 2.04 | 9.7 | 0.70 | 0.095 | 1.08 | 0.277 | 0.37 | 0.37 | 0.23 | 0.018 | 0.47 |
| 1000 | 7.53 | 0.293 (592) | 0.039 | 0.061 | 0.191 | 0.327 | 0.467 | 2.15 | 20.0 | 0.73 | 0.060 | 0.53 | 0.235 | 0.58 | 0.56 | 0.23 | 0.010 | 0.39 |

p_fire armé par angle de la meilleure cible ([0-2[, [2-5[, [5-10[, [10-20[, [20-45[, [45-180[ °) :
- côté 550 : 0.34 / 0.25 / 0.14 / 0.10 / 0.10 / 0.06
- côté 1000 : 0.18 / 0.11 / 0.05 / 0.04 / 0.04 / 0.04

Hit/obus par angle au tir : < 5° 0.37 / 0.33, 5-20° 0.14 / 0.11, > 20° 0.015 / 0.012 (côtés 550 / 1000).

corr(μ canon, erreur angulaire de la cible), décisions step ≥ 10 : lacet −0.016 / −0.009, site
+0.011 / +0.004 ; sous 20° de lacet −0.026 / −0.072.

## Lecture

- **Gâchette sélective** : p_fire décroît de façon monotone avec l'angle, aux deux côtés. Ratio 4.9 / 7.5
  (aligné & proche vs hors axe), 3.7 / 3.2 sur l'angle seul. La moitié des obus partent sous 5°.
  Niveau de base bas (p_off 0.07 / 0.04).
- **Les tirs touchent** : hit/obus 0.23 / 0.19, cohérent avec le hit du CSV (0.19–0.22 à D=1000).
  Les obus tirés sous 5° touchent 25× plus que ceux tirés au-delà de 20°.
- **Plus de dump** : 6–10 % de décisions désarmées, 1re réserve vide médiane 9.7 s / 20 s. 70 % des
  tank-épisodes vident quand même leur réserve à un moment, mais tard.
- **Visée** : tourelle à moins de 5° d'une cible 24–28 % du temps, angle médian 11–12°. Mais μ canon
  n'est pas corrélé à l'erreur de lacet ni de site (|corr| < 0.08). L'alignement ne vient donc pas d'une
  commande canon proportionnelle à l'erreur : il peut venir du châssis (dir), de la stabilisation du gun
  laying, ou d'une dépendance non linéaire. La sonde ne permet pas de trancher.
- **Têtes** : σ canon 0.37 (550) et 0.57 (1000), jamais au plafond ; μ canon petit (|μ| 0.23) et
  non saturé ; μ dir peu saturé (≤ 2 %).
- **Mode déterministe** (p > 0.5) : la gâchette ne tire que dans 24 % (550) / 9 % (1000) des états
  alignés. En jeu déterministe, l'agent tirera très peu.

## Face à la seule sonde de référence (SAC 420 save_1, D=250, côtés 250/550)

| | 420 (550) | 423 (550) |
|---|---|---|
| ratio | 2.44 | 4.86 |
| hit/obus | 0.027 | 0.229 |
| %désarmé | 0.70 | 0.095 |
| réserve vide médiane | 4.1 s | 9.7 s |
| tour<5° | 0.085 | 0.277 |

Pas de référence PPG avec ce script : aucune sonde de 421 ou 422.

## Établi / non tranché

- Établi : gâchette sélective et forte (ratio 5–7.5), hit/obus ≈ hit du CSV, dump supprimé, tourelle
  alignée 3× plus souvent qu'en 420.
- Établi : μ canon sans relation linéaire avec l'erreur de la cible.
- Non tranché : d'où vient l'alignement de la tourelle (châssis, gun laying, non-linéarité) ; à quel
  moment du run la sélectivité est apparue.

## Suite utile

- Sonder save_5 (≈ 140k, D≈450) et save_12 (≈ 330k, D≈850) pour dater la sélectivité et la chute de
  fire (skill `probe-timeline` ensuite).
- Sonder une save de 421 au même côté pour séparer l'effet gun laying + φ de l'effet PPG.
- Pour la visée : corr(μ dir, lacet) ou une sonde en figeant le châssis, pour savoir qui aligne la tourelle.

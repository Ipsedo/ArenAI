# Sonde train_428_ppg / save_49

- Checkpoint : index ≈ 1323k, D = 1000. C'est la dernière save. Le run est en cours, CSV à 1343k.
- Binaire : `cmake-build-release/arenai_agent/arenai_agent_probe`, compilé à 9d843717, **aucun diff d'inférence** avec le code du run (0e5093a1).
- Protocole : côtés 550 et 1000, 4 épisodes chacun, mode stochastique, économie 10 / +3 / 30.

## Résumé, comparé à save_27 (729k) et à 423 save_21 (567k)

| côté | save | ratio | p_al (n) | p_off | p_arm | hit/ob | h<5° | tirs<5° | enrich | vide_s | %vident | %désarmé | tir/s | tour<5° | σx | σy | p_zoom |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 550 | 423 s21 | 4.86 | 0.359 (836) | 0.074 | 0.139 | 0.229 | 0.369 | 0.500 | 2.04 | 9.7 | 0.70 | 0.095 | 1.08 | 0.277 | 0.37 | 0.37 | 0.47 |
| 550 | s27 | 8.60 | 0.146 (1441) | 0.017 | 0.054 | 0.268 | 0.345 | 0.649 | 2.00 | 31.9 | 0.49 | 0.026 | 0.50 | 0.340 | 0.35 | 0.34 | 0.40 |
| 550 | **s49** | 6.97 | 0.194 (1070) | 0.028 | 0.056 | **0.292** | **0.426** | 0.633 | 1.76 | 47.0 | 0.40 | 0.004 | 0.53 | **0.375** | 0.29 | 0.29 | **0.03** |
| 1000 | 423 s21 | 7.53 | 0.293 (592) | 0.039 | 0.061 | 0.191 | 0.327 | 0.467 | 2.15 | 20.0 | 0.73 | 0.060 | 0.53 | 0.235 | 0.58 | 0.56 | 0.39 |
| 1000 | s27 | 21.9 | 0.149 (888) | 0.007 | 0.025 | 0.270 | 0.352 | 0.656 | 2.60 | 46.1 | 0.39 | 0.011 | 0.23 | 0.264 | 0.44 | 0.43 | 0.34 |
| 1000 | **s49** | 11.1 | 0.196 (775) | 0.018 | 0.026 | **0.295** | **0.466** | 0.573 | 1.75 | 69.9 | 0.32 | 0.008 | 0.25 | **0.332** | 0.43 | 0.40 | **0.02** |

p_fire armé selon l'angle à la meilleure cible ([0-2[, [2-5[, [5-10[, [10-20[, [20-45[, [45-180[ °) :
- côté 550 : 0.138 / 0.080 / 0.038 / 0.034 / 0.038 / 0.021
- côté 1000 : 0.071 / 0.034 / 0.017 / 0.014 / 0.017 / 0.018

## Lecture

- **Zoom abandonné.**
  - p_zoom tombe de 0.34–0.40 à 0.02–0.03, et sa corrélation avec la gâchette disparaît (0.43–0.48 → ≈ 0).
  - C'est la chute de Hd dans le CSV (0.35 → 0.07–0.09 dès 1.18M) : `alpha_zoom` décolle à peine du plancher (≤ 1.8e-4). La cible discrète à 0.01 laisse le zoom devenir déterministe.
- **Précision par tir aligné : nouveau fait.** h<5° (touches par obus tiré à moins de 5°) passe de 0.35 à **0.43 / 0.47**. C'est le premier gain de balistique à angle égal de la lignée : 423 et save_27 plafonnaient à 0.33–0.37. Ce gain arrive **sans zoom**. Le hit/obus monte à 0.29.
- **Visée.** La tourelle est à moins de 5° d'une cible dans 0.375 / 0.332 des décisions, contre 0.340 / 0.264 à save_27, en hausse des deux côtés. σ canon reste à 0.29–0.43, sans plafond ni saturation.
- **Gâchette moins nette en angle.**
  - La probabilité de tir quand le canon est aligné et la cible proche remonte (p_al 0.15 → 0.19).
  - Mais le fond hors axe remonte aussi (p_off 0.007–0.017 → 0.018–0.028) et devient **plat au-delà de 10°** (0.014–0.038).
  - Le ratio baisse donc (8.6 → 7.0 et 21.9 → 11.1). Sur l'angle seul, il passe de 6.4 / 9.4 à 3.5 / 2.6.
  - 15–20 % des tirs partent à plus de 20° (8–9 % à save_27), avec un hit/obus de 0.007.
- **En mode déterministe**, l'agent ne tire toujours que dans 2.8 % (côté 1000) à 6.6 % (côté 550) des états alignés : il reste muet en jeu.
- **Réserve.** Elle est encore moins vidée : 32–40 % des tank-épisodes la vident, la première fois à 47 / 70 s en médiane, et les pas désarmés sont sous 1 %.
- **Déplacement du châssis : changement de stratégie.**
  - save_27 : dir_y (vitesse des roues, pré-tanh) à −1.04 en moyenne, soit plein régime dans un sens ; braquage dir_x quasi nul (−0.19).
  - save_49 : braquage **dir_x à +1.05** (tanh ≈ 0.78, 17 % saturé) et dir_y +0.35, soit une vitesse réduite dans l'autre sens.
  - Le tank tourne donc en rond, braqué d'un seul côté. C'est cohérent avec la vie médiane plus courte côté 550 (47 s contre 96 s) et avec des kills qui remontent vers 18–21 dans le CSV.

## Conclusions

**Établi** (même binaire, même protocole que save_27)
- La précision par tir aligné progresse (h<5° 0.43–0.47), de même que l'alignement de la tourelle.
- Le zoom est abandonné.
- Le châssis tourne en rond braqué d'un côté.

**Dans le bruit / non tranché**
- La baisse du ratio vient d'un fond hors axe plus élevé mais plat. Les effectifs alignés sont de 775 à 1070, donc le ratio reste bruité.
- On ne sait pas si l'abandon du zoom aide la précision (zoom = FOV ×4, peut-être nuisible à la détection de la cible) ou s'il est sans lien.
- On ne sait pas si tourner en rond est une stratégie de visée (balayage du lacet par le châssis) ou d'esquive.

## Suites utiles
- Mesurer corr(μ canon, erreur angulaire) et corr(μ dir_x, lacet) pour savoir qui aligne : le canon ou le châssis.
- Vérifier en jeu le comportement de rotation du châssis.

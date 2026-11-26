# Sonde train_428_ppg / save_27

- Checkpoint : index ≈ 729k, D = 1000 (curriculum saturé depuis 429k), juste après la montée du hit du CSV (0.26 @ 724k).
- Binaire : `cmake-build-release/arenai_agent/arenai_agent_probe`, recompilé le 2026-11-15 à 10:47 à 9d843717. Ce commit n'ajoute que des modèles et le preset desktop par rapport au code du run (0e5093a1) : **aucun diff d'inférence**. L'ancien binaire de sonde (7422d098) ne contenait pas le mapping a·|a| du canon et aurait faussé la sonde.
- Protocole : côtés 550 et 1000, 4 épisodes chacun, mode stochastique, économie 10 / +3 / 30.

## Résumé, comparé à 423 save_21 (567k, D = 1000)

| côté | run | ratio | p_al (n) | p_off | p_arm | hit/ob | h<5° | tirs<5° | enrich | vide_s | %vident | %désarmé | tir/s | tour<5° | σx | σy | p_zoom |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 550 | 423 | 4.86 | 0.359 (836) | 0.074 | 0.139 | 0.229 | 0.369 | 0.500 | 2.04 | 9.7 | 0.70 | 0.095 | 1.08 | 0.277 | 0.37 | 0.37 | 0.47 |
| 550 | **428** | **8.60** | 0.146 (1441) | **0.017** | 0.054 | **0.268** | 0.345 | **0.649** | 2.00 | 31.9 | **0.49** | **0.026** | 0.50 | **0.340** | 0.35 | 0.34 | 0.40 |
| 1000 | 423 | 7.53 | 0.293 (592) | 0.039 | 0.061 | 0.191 | 0.327 | 0.467 | 2.15 | 20.0 | 0.73 | 0.060 | 0.53 | 0.235 | 0.58 | 0.56 | 0.39 |
| 1000 | **428** | **21.9** | 0.149 (888) | **0.007** | 0.025 | **0.270** | 0.352 | **0.656** | 2.60 | 46.1 | **0.39** | **0.011** | 0.23 | 0.264 | 0.44 | 0.43 | 0.34 |

p_fire armé selon l'angle à la meilleure cible ([0-2[, [2-5[, [5-10[, [10-20[, [20-45[, [45-180[ °) :
- côté 550 : 0.142 / 0.092 / 0.047 / 0.026 / 0.023 / 0.012
- côté 1000 : 0.097 / 0.051 / 0.020 / 0.010 / 0.008 / 0.006

## Lecture

- **Gâchette.** Elle est nettement plus sélective qu'en 423 : ratio ×1.8 (côté 550) et ×2.9 (côté 1000). Elle reste monotone avec l'angle. Mais ce gain vient surtout de **l'extinction des tirs hors axe** (p_off divisé par 4 à 6). La probabilité de tir quand le canon est aligné baisse aussi, de moitié environ (0.36 → 0.15). Le niveau global de la gâchette p_arm est divisé par 2.5.
- **En mode déterministe** (p > 0.5), la gâchette ne tire que dans 6.8 % (côté 550) et 4.1 % (côté 1000) des états alignés, contre 24 % et 9 % en 423. **En jeu déterministe, l'agent est quasi muet.**
- **Précision.**
  - Le hit/obus passe de 0.19–0.23 à 0.27.
  - Le hit/obus des tirs alignés (h<5°) ne bouge pas : 0.35 contre 0.33–0.37.
  - Le gain vient donc de **la composition des tirs** : 65 % des tirs sont alignés, contre 47–50 % en 423. Ce n'est pas une meilleure balistique à angle égal.
- **Dump.** Il est fortement réduit :
  - la part de tank-épisodes qui vident leur réserve passe de 70–73 % à 39–49 % ;
  - la première réserve vide arrive à 32–46 s au lieu de 10–20 s ;
  - la part des pas désarmés passe de 6–10 % à 1–3 %.
- **Visée.**
  - Tourelle à < 5° d'une cible : 0.34 / 0.26 contre 0.28 / 0.24 en 423. Légèrement meilleur, dans le bruit côté 1000.
  - σ canon médian : 0.35–0.44, jamais au plafond. Pas de saturation de μ canon (≤ 1 %).
- **Zoom.** Corrélé à la gâchette (corr 0.43–0.48). La part de zoom monte avec l'alignement, comme en 423.
- **Durée de vie.** Médiane 96 s (côté 550) et 180 s (côté 1000) : jeu plus passif, cohérent avec la baisse de kill du CSV.

## Conclusions

**Établi**
- La bascule du CSV vers 560–620k (fire ↓, hit ↑) est un apprentissage de **sélection** : les tirs hors axe sont supprimés, et la réserve n'est plus vidée.
- Ce n'est ni une meilleure précision par tir aligné, ni une meilleure visée nette.
- Le prix est une gâchette globalement basse, y compris quand l'agent est aligné. D'où moins de kills et un mode déterministe muet.

**Non tranché**
- L'origine du changement : cible discrète 0.01 (α_fire au plancher, rien ne retient la gâchette basse) ou γ 0.998. Les deux sont confondus.
- La corrélation μ canon / erreur angulaire n'est pas sortie par la sonde. On ne sait donc pas si la commande canon est devenue proportionnelle (elle ne l'était pas en 423).

## Suites utiles
- Sonder `save_19` (≈ 510k, avant la bascule) pour avoir le « avant » au même D dans le même run.
- Sonder une save plus tardive pour voir si p_al remonte (gâchette sélective et active) ou continue de baisser (vers une gâchette éteinte).

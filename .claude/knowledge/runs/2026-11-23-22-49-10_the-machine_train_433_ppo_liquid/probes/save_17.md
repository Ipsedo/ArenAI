# Sonde train_433_ppo_liquid / save_17 — après le recul du curriculum (2026-11-24)

- Index 459 k, D 800 (redescendu de 850 à 429 k). Même binaire et mêmes réglages que `save_14.md` (côtés 550 / 1000, 3 épisodes, stochastique, oracles `--oracle_fov 79`), GPU partagé avec l'entraînement.

| mesure | s14 côté 550 | s17 côté 550 | s14 côté 1000 | s17 côté 1000 |
|---|---|---|---|---|
| tour<5° | 0,185 | 0,165 | 0,133 | 0,146 |
| p_fire armé, aligné < 5° (toute distance) | 0,300 | 0,194 | 0,091 | 0,041 |
| p_fire armé, aligné < 5° et < 40 m | 0,741 | 0,744 | 0,758 | 0,722 |
| p_fire armé, hors axe ≥ 20° | 0,032 | 0,018 | 0,0098 | 0,0042 |
| obus/tank-s | 0,97 | 0,70 | 0,28 | 0,17 |
| hit/obus | 0,161 | 0,205 | 0,148 | 0,180 |
| hit/obus des tirs < 5° | 0,263 | 0,359 | 0,246 | 0,352 |
| touches/tank-s | 0,156 | 0,144 | 0,041 | 0,031 |
| kills/épisode | 20,3 | 19,3 | 9,0 | 6,7 |
| touches/tank-s sous visée oracle (gâchette politique) | 0,156 | 0,128 | 0,076 | 0,049 |
| obus/tank-s sous visée oracle | 0,36 | 0,25 | 0,16 | 0,09 |
| touches/tank-s tout oracle | 0,694 | 0,780 | 0,692 | 0,629 |
| σ canon_x / canon_y médian | 0,39 / 0,35 | 0,18 / 0,29 | 0,50 / 0,48 | 0,22 / 0,36 |
| balayage, signe dominant de μ canon_x | 0,84 | 0,89 | 0,95 | 0,97 |

corr(μ canon_x, lacet), côté 1000, s14 → s17 : 100-200 m, 5-20° −0,26 → −0,32 ; 200-400 m, 5-20° +0,01 → −0,06 (n 872) : dans le bruit.

## Lecture

- **Établi** : la visée n'a ni régressé ni progressé (tour<5°, corrélations par tranche).
- **Établi** : la gâchette se ferme partout sauf à bout portant : p_fire aligné toutes distances ÷1,5 (550) et ÷2,2 (1000), inchangé à < 40 m. Le hit monte parce que les tirs lointains disparaissent ; les touches et les kills baissent en volume.
- **Établi** : sous visée parfaite la gâchette tire encore moins qu'à save_14 (÷1,4 à ÷1,8) : elle s'éloigne du comportement de 552 (1,2-1,3 obus/tank-s).
- σ canon_x divisé par deux entre les deux saves.
- Cause de la fermeture non établie.

# Sonde train_433_ppo_liquid / save_14 — visée (2026-11-24)

- Index 378 k, D 850. Agent `ppo_liquid`, CNN 4 blocs + spatial softmax (x, y, amplitude par canal → 384) + tête de présence sur la carte du bloc 4.
- Binaire : `build/arenai_agent/arenai_agent_probe` recompilé le 24/11 13:44 sur l'arbre de travail (f7435cc2 + modifications indexées non commitées = code du run, binaire d'entraînement du 23/11 22:49). Le binaire précédent (23/11 20:52) était antérieur au spatial softmax ; `--check-only` ne l'a pas signalé.
- Côtés 550 et 1000, 3 épisodes, stochastique, GPU partagé avec l'entraînement 433 en cours.
- Référence : rapport 552 (autre PC, CNN 7 blocs aplati, save_41, 1,11 M, D 1000). **Pas à D égal ni à maturité égale** : 552 a 3× plus de pas.

## Résumé

| mesure | 433 s14 côté 550 | 433 s14 côté 1000 | repère 552 s41 |
|---|---|---|---|
| tour<5° | 0,185 | 0,133 | 0,17-0,19 |
| hit/obus | 0,161 | 0,148 | 0,34-0,37 |
| touches/tank-s, politique | 0,156 | 0,041 | 0,275 / 0,119 |
| touches/tank-s, lacet+site oracle en vue, gâchette politique | 0,156 (×1,0) | 0,076 (×1,9) | 0,527 (×1,9) / 0,365 (×3,1) |
| obus/tank-s sous visée oracle | 0,36 | 0,16 | 1,2-1,3 |
| touches/tank-s, tout oracle en vue | 0,694 | 0,692 | 0,766 / 0,756 |
| kills/épisode politique → visée oracle → tout oracle | 20,3 → 21,7 → 30,7 | 9,0 → 17,7 → 35,0 | 25,3 → 32,0 → 30,7 / 17,7 → 28,3 → 43,7 |
| gâchette : ratio aligné&proche / hors axe | 22,9 (n 311) | 77,5 (n 72, bruit) | — |
| réaction par passage, bon site, < 60 m / 60-100 m | 0,76 / 0,34 | 0,83 / 0,27 | 0,58-0,91 / 0,56-0,57 |
| crédit du geste ρ (GAE λ 0,95) | −0,009 à +0,015 | 0,000 à +0,009 | ≈ 0 |

## Visée par distance (côté 1000, step ≥ 10)

corr(μ canon_x, lacet de la cible) ; vers la cible = négatif.

| distance | 1-5° | 5-20° | 20-40° | ≥ 40° |
|---|---|---|---|---|
| < 60 m | −0,39 | −0,57 | −0,69 | +0,11 |
| 60-100 m | −0,42 | −0,49 | −0,16 | +0,03 |
| 100-200 m | −0,10 | −0,26 | −0,10 | 0,00 |
| 200-400 m | −0,04 | +0,01 | +0,06 | +0,05 |

Tirs : hit/obus 0,43 (< 60 m), 0,18 (60-100 m), 0,075 (100-200 m), 0,08 (> 200 m) ; erreur site+chute au tir médiane −2,0 à −2,7° (|.| 2,5-4,4°). Balayage : tanh μ canon_x moyen −0,39 / −0,53, signe dominant 84 % / 95 % des décisions.

## Précision de la position à l'entrée de la politique (`banc/banc_position.txt`)

Images de save_14, côté 1000, zoom forcé, 134 k images dont 13,4 k d'apprentissage avec ennemi visible < 400 m, test 5,6 k (épisode non vu), 4 passes, une graine. Lacet : gauche/droite / erreur médiane / part < 1° ; site : erreur médiane.

| configuration | 60-100 m | 100-200 m | 200-400 m |
|---|---|---|---|
| 433 figé (vecteur softmax reçu par la LTC) | 0,96 / 0,87° / 55 % ; 0,41° | 0,97 / 1,51° / 35 % ; 0,68° | 0,84 / 4,18° / 16 % ; 1,20° |
| carte du bloc 4 de 433 figée, lue aplatie | 0,92 / 1,79° / 29 % ; 0,79° | 0,93 / 3,23° / 18 % ; 0,93° | 0,78 / 7,82° / 7 % ; 1,48° |
| 433 affiné entièrement | 0,97 / 0,78° / 61 % ; 0,37° | 0,98 / 0,94° / 52 % ; 0,43° | 0,89 / 1,75° / 34 % ; 0,63° |
| 4 blocs + softmax, poids neufs | 0,94 / 1,81° ; 1,05° | 0,91 / 3,27° ; 1,31° | 0,67 / 8,78° ; 2,07° |
| 7 blocs aplati, poids neufs (étalon) | 0,90 / 2,56° ; 1,19° | 0,83 / 5,31° ; 1,42° | 0,58 / 10,12° ; 1,59° |
| toujours au centre | 6,20° | 12,36° | 11,59° |
| *552 figé (rapport, autre jeu d'images, 25,6 k)* | *1,26°* | *2,19°* | *5,34°* |

Le jeu d'apprentissage est 2× plus petit que celui du rapport : l'étalon « 7 blocs neufs » y rend 2,56 / 5,31 / 10,12° contre 1,04 / 1,04 / 2,64° dans le rapport ⟹ les lectures sont sous-entraînées et les chiffres absolus ne se comparent pas à 552. Température apprise ≈ 1 (0,93-1,03) : elle n'a pas bougé.

## Lecture

- **Établi** : la représentation transmet la position (figé < 100 m : 0,87°, mieux que toutes les lectures neuves sur le même jeu) ; le softmax ne dégrade pas la carte du bloc 4, il la lit mieux qu'un aplati à données égales.
- **Établi** : la politique ne s'en sert pas au-delà de 100 m (corr ≈ 0 à 200-400 m, −0,1 à −0,26 à 100-200 m), tour<5° égal à 552, crédit du geste nul comme dans 552.
- **Établi** : la gâchette est sélective mais ne suit pas une visée parfaite (0,16-0,36 obus/tank-s sous oracle contre 1,2-1,3 dans 552).
- **Non décidable** : 433 vs 552 à D égal (552 n'est pas sur ce PC, 433 n'est pas à D 1000).

## Suite

- Resonder à D 1000 (mêmes commandes, `oracle/` + `banc/`) pour la comparaison à D égal.
- `banc/s14_zoom.frames` (13 Go) conservé pour rejouer le banc ; supprimable.

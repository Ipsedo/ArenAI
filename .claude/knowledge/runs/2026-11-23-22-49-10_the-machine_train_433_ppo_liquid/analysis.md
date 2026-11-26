# train_433_ppo_liquid — analyse (2026-11-24, run en cours)

- Lancé 23/11 22:49, dernière ligne 24/11 13:41, index 404 k, 27,1 k pas/h (868 k tank-steps/h), 32 tanks.
- Code effectif : f7435cc2 + modifications indexées non commitées (`vision.cpp/.h`, `constants.h`, `*_hyperparams.h`) — confiance haute : sources du 23/11 21:51, binaire d'entraînement 22:49, `actor_repr.txt` montre 4 blocs et une entrée LTC 448 = 64 + 3 × 128.

## Ce que le run teste

CNN 4 blocs (au lieu de 7) + spatial softmax (x, y, amplitude par canal, température par canal) + tête de présence sur la carte lue par la politique. Référence voulue : 552 (autre PC, db9a825a, 7 blocs aplati, tête au bloc 4). Vs 432 local : 5 commits + architecture ⟹ confondu, 432 n'est pas une référence.

## CSV

Jalons D : 300 @ 48 k, 400 @ 114 k, 500 @ 181 k, 600 @ 233 k, 700 @ 281 k, 800 @ 328 k, 850 @ 366 k (552 : D 1000 @ 699 k, jalons intermédiaires inconnus ici).

| D | index | hit | kill | fire | miss | ev | Hc | Hd | skip |
|---|---|---|---|---|---|---|---|---|---|
| 350 | 67-113 k | 0,142 | 24,1 | 2,04 | 34 | 0,38 | −0,31 | 0,63 | 0 |
| 500 | 181-207 k | 0,107 | 23,2 | 1,67 | 42 | 0,38 | −0,80 | 0,57 | 0,007 |
| 650 | 267-280 k | 0,085 | 17,4 | 1,33 | 43 | 0,25 | −0,68 | 0,53 | 0,05 |
| 750 | 305-327 k | 0,115 | 20,2 | 1,32 | 46 | 0,49 | −0,69 | 0,43 | 0,03 |
| 850 | 366-404 k | 0,159 | 20,9 | 1,13 | 41 | 0,47 | −0,43 | 0,32 | 0,07 |

- hit recule de 0,14 (D 350) à 0,085 (D 650) puis remonte à 0,16 (D 850) pendant que fire baisse de 2,0 à 1,1 : la remontée du hit vient de la gâchette (moins de tirs hors axe), pas de la visée (sonde save_14 : tour<5° 0,13-0,19).
- α_canon_y remonte de 4,5e-4 à 1,1e-3 (Hc du site sous la cible) ; autres α au plancher 1e-4.
- Repère 552 à D 1000 : hit 0,38, fire 0,42, kill 19,3, miss 46 m. 433 n'y est pas.

## Diagnostic (détail : `probes/save_14.md`)

- Établi : position de la cible présente à l'entrée de la politique (0,87° / 1,51° / 4,18° à 60-100 / 100-200 / 200-400 m), au moins aussi bonne que 552 figé (1,26 / 2,19 / 5,34° avec 2× plus de données de lecture).
- Établi : non utilisée au-delà de 100 m ; crédit du geste ρ ≈ 0 ; balayage à signe dominant 84-95 %.
- Dans le bruit / non décidable : vitesse de curriculum et résultat vs 552 (pas de CSV 552 ici, pas à D 1000).

## Marqueurs

- D 1000 : hit ≥ 0,38 et kill ≥ 19,3 pour égaler 552 ; touches/tank-s > 100 m à la sonde.
- Sonde à D 1000 : corr(μ canon_x, lacet) à 100-400 m doit devenir négative ; rapport oracle/politique doit baisser sous ×1,9 (550) et ×3,1 (1000).

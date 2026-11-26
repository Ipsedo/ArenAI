# Run 423 — PPG

- Lancé 2026-11-12 23:46 → dernière ligne 2026-11-13 08:15, **en cours** (8.5 h).
- Index final 581k, 68.4k pas/h (2.19 M tank-steps/h, 32 tanks) ; 422 tournait à 74.8k pas/h.
- Commit effectif **802b1b32** (confiance haute) : commit 23:45:43, binaire 23:46:37, `config.json` 23:46:38, arbre propre.
- Saves : save_21 ≈ 581k (~27.7k pas/save) ; D=1000 atteint vers save_15.

## Ce que teste ce run

Config identique à 422. Seul changement de code : **7422d098, arrêt KL par groupe d'actions**.
Avant, un seul `target_kl` sur la KL jointe coupait tout l'acteur ; maintenant le groupe visée
(canon_x, canon_y, avantage φ sans critique) et le groupe reste (déplacement, fire, zoom, avantage
critique) sont stoppés chacun sur leur propre KL (> 1.5 × 0.05). `skip` = skip du groupe reste,
`skip_canon` = skip du groupe visée, `kl` = KL jointe.

Hérité de 422 (1e171bbf), donc **confondu** avec 423 face à 421 : stabilisation du canon vis-à-vis
des mouvements du châssis, avantage visée = potentiel φ (trajectoire théorique de l'obus) avec
`aim_gae_lambda = 0`, cibles d'entropie [0, 0, 0, 0], retour à la visée en delta.

422 a été arrêté à 148k (D=500) : il n'existe pas de 422 long pour isoler l'arrêt par groupe.

## Lecture du CSV

Jalons D : 300 @ 21k, 400 @ 67k, 500 @ 145k, 600 @ 208k, 700 @ 267k, 800 @ 305k, 900 @ 366k,
**1000 @ 429k** — premier run à atteindre la borne finale du curriculum (421 : D=400 à 145k,
450 à 566k ; meilleurs SAC : D=400).

| D | hit | fire (/s) | kill | s (s) | miss (m) | ev | Hc | skip | skip_canon |
|---|---|---|---|---|---|---|---|---|---|
| 400 | 0.126 | 1.64 | 22.6 | 52 | 36 | 0.71 | −0.46 | 0.15 | 0.10 |
| 500 | 0.126 | 1.55 | 19.5 | 52 | 39 | 0.75 | 0.11 | 0.01 | 0.00 |
| 700 | 0.142 | 1.14 | 21.8 | 59 | 54 | 0.75 | 0.11 | 0.03 | 0.01 |
| 900 | 0.171 | 1.01 | 22.1 | 45 | 58 | 0.74 | 0.05 | 0.09 | 0.02 |
| 1000 (429k → 581k) | 0.193 | 0.81 | 21.2 | 48 | 64 | 0.74 | 0.06 | 0.03 | 0.03 |

- **hit monte avec D** au lieu de chuter : 0.126 (D=400) → 0.193 (D=1000), dernier 0.217 @ 581k,
  record du run. Les tranches à D=1000 : 0.186 → 0.199 → 0.194, encore en légère hausse.
- **fire baisse** 1.6 → 0.8 /s pendant que hit monte. Touches/s par tank vivant ≈ fire × hit :
  0.21 (D=400) → 0.16 (D=1000), dans une arène 6× plus grande ; kill/épisode plat ~21.
- miss 22 → 64 m croît avec D : géométrique, pas une dégradation de la visée.
- Critique stable (ev 0.70–0.75 tout le run, v ~0.04).
- Hc : creux −1.34 @ 59k (D=350), puis remonte à ~+0.1 dès D=450 et y reste. α canon actifs
  (0.01–0.035, au-dessus du plancher) ; α dir au plancher 1e-4 depuis ~400k ; α fire/zoom au plancher.
- KL : régime KL seulement à D=300–400 (kl 0.14, clip 0.5, skip 0.11–0.15), puis kl 0.02–0.05,
  clip ~0.21, skip < 0.1.

### Face à 422 (même config, arrêt KL joint) sur 0–148k

Jalons D identiques au point de log près (300 @ 21k … 500 @ 145k), hit identique dans le bruit
(D=400 : 0.126 vs 0.123 ; D=450 : 0.131 vs 0.119). Différences nettes :
- skip : 423 ≤ 0.15 vs 422 0.47 (D=400) → 0.58 (D=450) — dans 422 la KL du canon coupait la
  moitié des pas du groupe reste ;
- Hc : 423 remonte à +0.16 à D=450, 422 reste à −1.05 ;
- clip 0.24 vs 0.38 à D=450.

## Diagnostic

**Établi**
- Saut qualitatif vs 421 : curriculum déverrouillé jusqu'à D=1000 et hit ≈ 2.5× le niveau de 421
  à D=400 (0.077), plus de régime KL chronique, pas d'effondrement de σ. Attribuable au paquet
  1e171bbf + 7422d098 ; les deux sont confondus au-delà de 148k.
- L'arrêt KL joint de 422 coupait réellement le groupe reste (skip 0.5–0.6) ; l'arrêt par groupe le
  libère (skip < 0.15).
- Le curriculum est saturé : hit 0.19 ≫ ratio_high 0.08. D ne mesure plus rien depuis 429k ; seuls
  hit, fire et kill à D=1000 restent informatifs.

**Probable, non prouvé par le CSV**
- fire ↓ avec hit ↑ ressemble à une gâchette qui devient sélective (tirer moins, mieux). Le CSV ne
  le prouve pas : `fire` est aveugle au dump de début d'épisode et ne dit rien de l'alignement au tir.
- Jusqu'à 145k, l'arrêt par groupe n'a pas changé la vitesse du curriculum ; on ne sait pas si 422
  serait aussi monté jusqu'à 1000 (arrêté trop tôt).

**Dans le bruit**
- Oscillations de fire 0.39 → 1.89 /s en quelques k (fenêtre 256 décisions) ; Hc oscille ±0.3 en fin de run.
- Perte de débit −8 % vs 422 : cohérente avec la double passe de gradient quand un groupe est stoppé,
  mais à ce niveau ça peut aussi être du bruit machine.

## Options (à trancher par Samuel)

1. **Sonder un checkpoint à D=1000** (save_18–21) : sélectivité de la gâchette (angle au tir vs hors
   tir, p(fire) armé), corr(μ canon, erreur), hit/obus — c'est la question que le CSV ne tranche
   pas, et le premier run où la sonde a de quoi trouver un signal. Comparer à save_5 (D≈450).
2. **Laisser tourner** pour voir si hit plafonne à D=1000 (repère : tranche suivante ≥ 0.20).
3. Le curriculum n'a plus de marge : étendre `final_spawn_*` ou durcir la tâche autrement est une
   décision de conception.
4. Pour attribuer le gain : rejouer 422 (arrêt KL joint) au-delà de 300k.

## Repères

- hit à D=1000 : progression ou plateau sur les 100k prochains.
- fire : s'il continue sous 0.6 /s avec kill qui baisse, surveiller une extinction de la gâchette.
- skip_canon / α canon : remontée = la visée recommence à bouger fort.
- ev < 0.6 ou Hc < −0.5 durable : changement de régime.

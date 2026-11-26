# Run 420 (dossier `train_20_sac`, nom erroné) — SAC, encodeur d'état partagé

- Lancé 2026-11-11 14:21, lu à 15:44 : **en cours**, index 41k, D = 250 depuis le début.
- Débit 29.8k pas/h (955k tank-steps/h, 32 tanks) — comparable à 419.
- Commit effectif : **aed53757** (encodeur partagé), confiance haute : binaire d'entraînement compilé
  à 14:21 et contenant `StateEncoder` ; `actor_repr.txt` montre `SacActor { encoder: StateEncoder }`
  et `critic_repr.txt` `TwinQNetwork { encoder: StateEncoder, q1, q2 }`, même température de
  SpatialSoftmax (0.3145) côté acteur et critique.

## Ce que teste ce run

`config.json` identique à 419. Seul changement de code : `StateEncoder` (vision + capteurs) unique,
porté par les jumeaux et entraîné par la perte TD ; l'acteur lit l'encodage détaché et n'optimise que
sa tête. Hypothèse : en 417/419 les features vision ne codaient pas le gisement de la cible (R² ≤ 0.16)
et ∂Q/∂a_canon n'avait pas de direction ; un encodeur partagé ne change pas le signal qui l'entraîne
(TD seul), il supprime seulement l'encodeur propre de l'acteur. Aucun changement confondu.

## Lecture du CSV (moyennes par tranche d'index, D = 250 partout)

| index | run | hit | kill | fire | miss | s | q | Q | Hc |
|---|---|---|---|---|---|---|---|---|---|
| 5–10k | 419 | 0.050 | 0.31 | 0.57 | 23.4 | 47 | 0.015 | 0.000 | −2.2 |
| | 420 | **0.012** | 0 | 0.43 | 25.4 | 29 | 0.005 | −0.003 | −0.7 |
| 10–20k | 419 | 0.054 | 0.37 | 0.58 | 22.5 | 61 | 0.025 | 0.009 | −3.5 |
| | 420 | 0.025 | 0.10 | 0.52 | 24.1 | 28 | 0.016 | 0.028 | −2.7 |
| 20–30k | 419 | 0.062 | 0.47 | 0.60 | 20.8 | 59 | 0.042 | 0.009 | −3.6 |
| | 420 | 0.042 | 0.26 | 0.58 | 22.3 | 47 | 0.038 | 0.072 | −4.0 |
| 30–42k | 419 | 0.070 | 0.66 | 0.69 | 19.3 | 71 | 0.057 | 0.026 | −3.7 |
| | 420 | 0.065 | 0.65 | 0.62 | 20.4 | 71 | 0.077 | 0.114 | −5.0 |

(419 s'arrête à 33k ; 415 : hit 0.091 à 30–42k et D = 300, 417 : 0.073.)

- **Démarrage lent** : hit ≈ 0.004–0.01 jusqu'à 8k (419 : 0.03–0.05 dès 3k), kill = 0 jusqu'à 14k.
  Une partie précède `learning_starts` (5k) et relève donc de l'init/du seed.
- **Rattrapage** : pente de hit plus raide, 0.072 à 39k et 0.077 à 41k (record du run, isolé) ;
  miss 26.9 → 19.8 m ; s 20 → 72 s. Le curriculum n'a pas bougé (seuil haut 0.08).
- **Hc dépasse la cible** : −5.6 à 30k, min −5.97 à 31k (cible ≈ −3.5 pour σ 0.1 × 4 axes, que 419
  tient à −3.4/−3.7). Les αc ont baissé jusqu'à ~1.8e-4 puis remontent (3.7e-4 à 41k) : la boucle α
  corrige avec retard. La sonde montre que ce n'est pas le canon : σ canon médian 0.48/0.43 (H ≈ −0.36/−0.30,
  σ ≫ cible 0.1). Le déficit vient des axes de direction, dont μ est saturé (|μ| > 1,5 : 25 et 29 %
  des décisions) : la cible d'entropie post-tanh est atteinte par saturation, comme en 415/416.
- Q monte à +0.11 (419 : +0.01–0.03) et q ×1.3 ; cohérent avec un r plus élevé en fin de fenêtre,
  pas de dérive négative comme en 415.
- αd monte continûment (0.0037 à 41k), comme en 419.

## Diagnostic

Établi :
- Le code est bien celui de l'encodeur partagé, sans autre changement.
- À save_1 (27k), **∂Q/∂a_canon n'a toujours pas de direction** (voir `probes/save_1.md`) :
  P(bon sens) 0.50–0.52 à tous les angles, armé 0.52, jumeaux en désaccord de signe (0.45–0.49).
  Strictement identique à 417/419 : l'encodeur partagé n'a pas rendu le critique sensible à la visée
  continue.
- La gâchette (tête discrète, espérance exacte sur Q(s, ·) à K sorties) est **sélective** :
  p_fire 0.50 sous 5° contre 0.24 au-delà de 45°, décroissance monotone avec l'angle.

Dans le bruit / non tranché :
- Démarrage lent puis rattrapage : un seul seed, l'écart à 41k (0.065 vs 0.070 à 30–42k) est nul.
- On ne sait pas si la sélectivité de la gâchette est nouvelle : aucune sonde 417/419 n'a été
  passée avec le script actuel (métrique `ratio`).
- Hc sous la cible : effet de l'encodeur partagé (l'acteur suit des features qui bougent sous lui)
  ou simple lenteur de α ; non attribuable.

## Options (à trancher par Samuel)

1. Sonder 419 save_1 avec le même script (côté 250) pour savoir si la gâchette sélective est due à
   l'encodeur partagé ou était déjà là (≈ 7 min avec l'entraînement en parallèle).
2. Laisser tourner jusqu'à ~80–100k et sonder save_2/save_3 : la question est de savoir si ∂Q/∂a
   prend une direction plus tard, ou si seul le canal discret apprend.
3. Comme en 419, tester les features de l'encodeur partagé (`--features`) : le gisement est-il codé
   maintenant que l'encodeur est unique ?

## Repères

- D passe 300 (hit > 0.08 tenu) : 415 l'avait à ~20k, 420 n'y est pas à 41k.
- Hc : doit remonter vers −3.5 à mesure que les αc remontent ; s'il reste < −5, regarder σ direction.
- Sonde save_2 : `ratio` de la gâchette et P(∂Q/∂a bon sens) armé.

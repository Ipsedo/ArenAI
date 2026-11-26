# train_432_ppo_liquid — analyse à 122k (en cours)

- Lancé 2026-11-22 13:31, analysé à l'index 122k (D=400), ~48k pas/h, 32 tanks.
- Code : arbre de travail non commité au lancement, au-dessus de `dcca274e`. Les sources ont une
  mtime 14:16 (commit `7bccaa04` + pre-commit), contenu supposé identique au lancement : confiance
  moyenne.
- Sonde : binaire `build/` du 13:28, aucun diff d'inférence signalé par le script.

## Ce que teste ce run

Vs la lignée précédente (431 ppg, 430 liquid) :

- un seul groupe d'optimisation pour la politique : plus de log-prob séparée canon / reste, plus
  d'avantage de visée (`aim_credit` supprimé), plus d'arrêt KL par groupe (`skip_canon` disparu) ;
- cible d'entropie continue −1 sur les 4 actions (0 avant) ;
- bootstrap de la valeur à la limite de temps (`dcca274e`).

Ces trois changements sont confondus. Pas de run de référence liquid récent (430 = 4k pas).

## CSV

| tranche (index) | D | hit | kill | fire | miss (m) | ev | Hc | kl | clip |
|---|---|---|---|---|---|---|---|---|---|
| 0-10k | 250 | 0.038 | 2.8 | 0.78 | 22 | 0.45 | 0.10 | 0.004 | 0.07 |
| 30-41k | 250 | 0.135 | 19.9 | 1.44 | 28 | 0.56 | −0.20 | 0.012 | 0.15 |
| 51-61k | 300 | 0.154 | 24.3 | 1.59 | 34 | 0.47 | −0.23 | 0.013 | 0.14 |
| 92-102k | 350 | 0.137 | 25.5 | 1.49 | 43 | 0.51 | −0.35 | 0.019 | 0.17 |
| 122k | 400 | 0.129 | 24.7 | 1.76 | 45 | – | – | – | – |

- `hit` ×4 entre 10k et 60k, puis recule lentement quand D monte ; `miss` suit D.
- `skip` ≈ 0, `kl` 0.02 sous la cible 0.05 : pas de régime KL.
- Jalons D : 300 @ 48k, 350 @ 67k, 400 entre 102k et 120k. 431 (ppg) à D=350 : hit 0.154,
  kill 27.5, miss 47 ; écart dans le bruit.

## Sondes (4 épisodes, stochastique)

| save | index | côté | tour<5° | hit/obus | h<5° | tirs<5° | ratio gâchette | %vident | σx canon |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 350 | 0.053 | 0.029 | 0.122 | 0.089 | 1.00 | 1.00 | 0.299 |
| 1 | 27k | 350 | 0.154 | 0.122 | 0.329 | 0.199 | 1.07 | 0.92 | 0.186 |
| 3 | 81k | 350 | 0.188 | 0.135 | 0.368 | 0.242 | 1.27 | 0.83 | 0.089 |
| 3 | 81k | 550 | 0.128 | 0.100 | 0.313 | 0.202 | 1.35 | 0.93 | 0.168 |

Asservissement en gisement, mesuré hors sonde standard : régression de μ_canon_x sur l'erreur
signée vers la cible la plus proche (`yaw`), convertie en fraction d'erreur fermée par décision.
Convention mesurée : une action canon_x de +1 déplace l'erreur de +12.7° par décision, le bon
sens est donc une corrélation négative.

| erreur | save_0 corr | save_1 corr / fermeture | save_3 corr / fermeture |
|---|---|---|---|
| 0-5° | +0.04 | −0.19 / 13 % | −0.19 / 14 % |
| 5-10° | +0.06 | −0.30 / 9 % | −0.24 / 8 % |
| 10-20° | +0.08 | −0.30 / 5 % | −0.26 / 5 % |
| 20-45° | +0.10 | −0.20 / 1.4 % | −0.15 / 1.4 % |
| 45-180° | +0.05 | −0.05 / 0.1 % | −0.06 / 0.1 % |

## Diagnostic

Établi :

- La visée en gisement est apprise : boucle de rappel absente à l'init, présente à 27k, et la
  tourelle passe 3,5× plus de temps à moins de 5° d'une cible (0.053 → 0.188).
- Elle est acquise avant 27k et son gain ne progresse plus entre 27k et 81k. Elle n'agit que sous
  ~45° d'erreur et ne ferme que 5 à 14 % de l'erreur par décision.
- Le gain de tour<5° entre save_1 et save_3 (0.154 → 0.188) coïncide avec la baisse de σ canon
  (0.186 → 0.089), pas avec un μ plus directif.
- Gâchette : sélectivité faible mais naissante (1.07 → 1.27), toujours couplée au zoom
  (corr 0.99) ; 83 % des tanks vident la réserve, en 3.4 s en médiane.

Non tranché :

- Site : corr(μ_canon_y, erreur de site) entre −0.02 et −0.09, et l'action canon_y ne déplace pas
  la colonne `pitch` de la sonde (pente 0.2°) : la mesure n'est pas valide en l'état.
- Effet propre du groupe unique : confondu avec la cible d'entropie −1 et le bootstrap.

## Mise à jour save_6 (162k, D=450, côté 350)

| save | tour<5° | hit/obus | h<5° | ratio gâchette | p_al / p_off | %vident | vide_s | σx canon p10/p50/p90 |
|---|---|---|---|---|---|---|---|---|
| 3 | 0.188 | 0.135 | 0.368 | 1.27 | 0.75 / 0.59 | 0.83 | 3.4 | 0.040 / 0.089 / 0.328 |
| 6 | 0.175 | 0.162 | 0.405 | 2.43 | 0.78 / 0.32 | 0.70 | 6.9 | 0.021 / 0.246 / 0.476 |

Asservissement en gisement à save_6 : corr −0.24 / −0.33 / −0.24 / −0.13 et fermeture
14.5 % / 8.8 % / 3.4 % / 0.8 % par décision sur 0-5° / 5-10° / 10-20° / 20-45°.

- Le gain de visée reste plat pour la 3e sonde consécutive (27k, 81k, 162k) ; tour<5° ne monte
  plus (0.175).
- Ce qui progresse entre 81k et 162k est la gâchette : extinction hors axe (p_off 0.59 → 0.32),
  ratio ×2, réserve vidée deux fois plus tard, découplage partiel du zoom (corr 0.99 → 0.77).
  Le gain de hit/obus (0.135 → 0.162) vient de là.
- σ canon devient dépendant de l'état : médiane remontée à 0.25, étalée de 0.02 à 0.48.

## Mise à jour save_13 (351k, D=800)

CSV : `fire` 1.40 → 0.35 tir/s entre 250k et 360k, `kill` 25.8 → 15.6, `r_hit` 0.0095 → 0.0035,
`hit` plafonne à 0.21-0.22. Bascule nette à 260k (D 600 → 650).

| save | côté | tir/s | p_al (<5° & <40 m) | p_fire <5° seul | p_off | hit/obus | h<5° | touches/s | tour<5° | dist. cible méd. | σx canon |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 | 350 | 1.77 | 0.78 | 0.56 | 0.32 | 0.162 | 0.405 | 0.287 | 0.175 | 147 m | 0.246 |
| 13 | 350 | 0.57 | 0.57 | 0.33 | 0.030 | 0.245 | 0.497 | 0.139 | 0.077 | 294 m | 0.478 |
| 13 | 800 | 0.30 | 0.70 | 0.24 | 0.016 | 0.221 | 0.503 | 0.067 | 0.050 | 415 m | 0.491 |

- Hors axe éteint quelle que soit la réserve (0.012-0.035 de 0 à 50 %) : ce n'est plus une
  économie de fin de réserve.
- À côté égal (350), touches/s divisées par 2, tanks 2× plus éloignés, tourelle <5° divisée par 2,
  σ canon doublé : la visée recule.
- Pénalité de réserve vide : −D/1000, traitée comme terminale (`cuts` inclut `truncated`, pas de
  bootstrap). Reward positif médian d'une touche 0.30, moyen 0.48. À D=800 la pénalité vaut 0.8,
  soit 0.08 par obus du stock initial, contre ~0.01 d'espérance pour un obus hors axe (2 % de
  touche) : le hors-axe est déficitaire dès D=250 (0.025 par obus).
- Un obus à <5° touche 1 fois sur 2 : espérance +0.24 et réserve neutre (2 rendus par touche). La
  pénalité ne justifie pas p_fire 0.24-0.33 dans ces états.
- Précédent : 428 (lancé avant `ee1ca624`, pénalité fixe) a fermé sa gâchette de la même façon
  (fire 0.9 → 0.3, kill 22 → 15).

## Repères

- Resonder save_6 à save_8 côté 350 avec la même régression : le gain de fermeture décolle-t-il de
  5 % à 10-20° ?
- `Hc` descend (−0.35) alors que le gain de μ est plat : surveiller si σ continue à se resserrer
  sans que μ suive.

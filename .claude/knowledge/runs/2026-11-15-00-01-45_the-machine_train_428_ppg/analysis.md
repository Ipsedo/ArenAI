# train_428_ppg — analyse du CSV

- Algo : PPG, 32 tanks. Lancé le 2026-11-15 à 00:01. Analysé à 10:44, run **en cours**, index 739k.
- Débit : 69 k pas/h (2,21 M tank-steps/h). 427 tournait à 74 k pas/h.
- Commit effectif : **0e5093a1**, confiance haute. Le binaire `cmake-build-release/.../arenai_agent_train` a été compilé à 00:01:45, soit au lancement. Le commit 9d843717, arrivé pendant le run, ne touche que le preset desktop et les modèles livrés.

## Ce que teste ce run (vs 427, d1bb4020)

| changement | 427 | 428 |
|---|---|---|
| `discrete_target_proba` (cible d'entropie fire/zoom) | 0.05 (H ≈ 0.199 nat) | **0.01** (H ≈ 0.056 nat) |
| `gamma` | 0.995 (horizon ≈ 20 s) | **0.998** (horizon ≈ 50 s) |
| `ev` | calculée par bloc GPU (256 lignes) | calculée par batch d'optimisation (512 lignes), toujours in-sample |

Les commits c33e68de/4c099cb6 (suppression du zoom côté agent, vision 512×256, bloc CNN supplémentaire) sont entièrement revenus en arrière : le diff net hors commentaires ne contient que les trois lignes ci-dessus. **Le changement de γ et celui de la cible discrète sont confondus.**

Héritage de 427 : potentiel de visée en ratio `miss / AIM_TARGET_RADIUS`, action canon mappée par a·|a|, économie 10 / +3 / 30.

## Lecture du CSV

**Curriculum.** Les jalons D sont identiques à l'index près pour 424 à 428 (300 @ 21k … 1000 @ 429k). Le curriculum monte donc à sa vitesse maximale dans tous ces runs, et ces jalons ne permettent pas de les départager. La comparaison se fait à D = 1000, à index égal.

| D = 1000, index | run | hit | fire (/s) | kill | s (s) | miss (m) | Hd | ev |
|---|---|---|---|---|---|---|---|---|
| 429–530k | 423 | 0.194 | 0.84 | 21.3 | 48 | 64 | 0.41 | 0.74 |
| | 427 | 0.137 | 1.17 | 22.4 | 44 | 81 | 0.55 | 0.73 |
| | **428** | 0.174 | 0.79 | 22.0 | 45 | 76 | 0.46 | 0.76 |
| 530–640k | 423 | 0.204 | 0.70 | 21.2 | 50 | 62 | 0.41 | 0.75 |
| | 427 | 0.165 | 0.70 | 21.9 | 50 | 77 | 0.45 | 0.78 |
| | **428** | 0.192 | **0.39** | 18.7 | 53 | 68 | 0.35 | 0.85 |
| 640–740k | 423 | 0.203 | 0.83 | 23.1 | 49 | 67 | 0.36 | 0.73 |
| | 427 | 0.147 | 1.09 | 23.4 | 44 | 89 | 0.52 | 0.71 |
| | **428** | **0.232** | 0.44 | **16.9** | 51 | **54** | 0.35 | 0.82 |

**Trajectoire.**
- **Jusqu'à 530k**, 428 suit 427 de près. Seul le hit est un peu au-dessus (0.17 vs 0.14).
- **Bascule vers 560–620k.** La cadence de tir tombe de 0.9–1.0 à 0.18–0.25 /s (minimum 0.084 @ 709k). Au même moment, le hit monte : 0.19 → 0.26 @ 724k, record de la lignée. Le miss descend de 76 à 49–55 m.
- En contrepartie, kill passe de 22 à 14–17 par épisode et r de 4.1e-3 à 2.3–3.3e-3.
- Ce régime « peu de tirs, plus précis, moins de kills » ressemble à celui de 425 (économie 2/20) à D = 950 : hit 0.23, fire 0.43, kill 14.5.
- **α discrets.** `alpha_fire` et `alpha_zoom` restent au plancher (1e-4) dès 64k et pendant tout le run (max 1.02e-4 après 100k). Dans 427, `alpha_fire` est monté jusqu'à 0.017 et `alpha_zoom` valait 3e-3 en fin de run : l'entropie discrète passait sous la cible H(0.05). Avec la cible H(0.01), la contrainte n'est **jamais active** dans 428, et rien ne retient plus la gâchette de devenir quasi déterministe.
- **α continus** : actifs entre 1e-3 et 0.03, comme 427. Hc reste autour de 0 (creux −0.5 @ 648k), sans effondrement.
- **Critique** : ev de 0.82 à 0.85 au-delà de 560k, au-dessus de 427 (0.71–0.78). Elle n'est pas strictement comparable (ev calculée sur un lot plus grand, et γ différent). Il y a un pic à −1.86 @ 512k, un artefact ponctuel déjà vu dans 427 à 1.17M.
- **Optimiseur** : clip ≈ 0.2, kl ≈ 0.03, skip ≤ 0.07 et skip_canon ≤ 0.02. Rien d'anormal.

## Diagnostic

**Établi**
- La contrainte d'entropie discrète est inerte dans 428 (α au plancher), alors qu'elle était intermittente dans 427.
- La baisse de la cadence de tir et la hausse du hit/tir arrivent ensemble, vers 560–620k, à D constant (1000). Ce n'est donc pas un effet du curriculum.
- À 640–740k, hit (0.23) et miss (54 m) sont les meilleurs de la lignée à index égal. kill est le plus bas (16.9 contre 23 pour 423 et 427).

**Non tranché**
- **Ce qui cause la bascule.** Deux explications sont possibles :
  - la gâchette libérée de sa cible d'entropie devient sélective ;
  - avec γ = 0.998, le coût de tir et la réserve (horizon 50 s) se valorisent plus, et la politique tire moins partout.

  Les deux changements sont confondus, et le CSV ne mesure pas la sélectivité.
- **Le gain de hit/tir.** Il peut venir d'une meilleure visée ou seulement d'une sélection des tirs faciles (moins de tirs, donc les tirs restants sont les plus favorables). Le miss en baisse va dans le sens de la sélection, sans trancher.
- **Ce que devient kill.** Moins de kills avec un hit plus élevé indique un jeu globalement plus passif. Ce n'est pas forcément une régression de compétence (la partie se joue en self-play), mais l'objectif « kills en jeu » recule.

## Options (pas de décision)

1. Sonde `save_27` (729k) : regarder la gâchette, la réserve et l'alignement. Voir `probes/save_27.md`.
2. Sonder une save d'avant la bascule (≈ `save_19`, 510k) pour comparer la sélectivité avant et après au même D.
3. Pour séparer les deux causes : un run 427 + γ 0.998 seul, ou 427 + cible 0.01 seule.
4. Laisser tourner pour voir si kill remonte à cadence basse, ou si fire continue vers 0 (régime « gâchette éteinte » de 392).

**Repères** : fire < 0.15 /s de façon durable = alerte gâchette éteinte. kill < 12 = régression. Le hit/tir reste à lire à côté de fire.

## Sonde save_27 (729k) — voir `probes/save_27.md`

La gâchette est sélective : ratio 8.6 (côté 550) et 21.9 (côté 1000), contre 4.9 et 7.5 en 423 save_21. Ce gain vient de l'extinction des tirs hors axe (p_off 0.017 / 0.007), mais la probabilité de tir aligné est divisée par 2 (0.15).

Le hit/obus monte à 0.27, alors que le hit/obus des tirs alignés reste à 0.35 : le gain vient de la composition des tirs, pas de leur précision. La réserve est beaucoup moins vidée (39–49 % des tank-épisodes contre 70 %). En mode déterministe, l'agent tire dans 4–7 % des états alignés : il est quasi muet en jeu.

## Sonde save_49 (1323k) — voir `probes/save_49.md`

La précision par tir aligné passe à 0.43–0.47, au-dessus des 0.33–0.37 de toute la lignée, et l'alignement de la tourelle progresse. Le zoom est abandonné (p_zoom 0.02–0.03), ce qui correspond à Hd ≈ 0.08 dans le CSV dès 1.18M. Le châssis tourne en rond, braqué d'un côté.

La gâchette est moins nette en angle (ratio 7.0 / 11.1, fond hors axe plat au-delà de 10°) et toujours muette en mode déterministe (3–7 % des états alignés).

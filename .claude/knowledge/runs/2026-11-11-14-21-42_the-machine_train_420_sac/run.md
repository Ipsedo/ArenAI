# train_420_sac

| | |
|---|---|
| machine | the-machine |
| utilisateur | samuel |
| lancement | 2026-11-11 14:21:42 |
| commit au lancement | `f9342001` — returns for SAC computed on multiple steps (10 nexts for the moment) |
| commits ≤ 90 min après | `f9df436a`, `aed53757` |
| diff au lancement | inconnu (identité relevée après la fin du run) |
| données brutes | `outputs/train_420_sac` sur the-machine |

## Faits établis

_SAC run 420 (dossier outputs/train_20_sac, nom erroné) = 419 + encodeur d'état partagé (aed53757) : démarrage lent puis hit 0.077 à 41k (D=250) ; sonde save_1 : ∂Q/∂a_canon toujours sans direction (0.50–0.52) mais gâchette sélective (ratio 2.2/2.4) ; Hc < cible par saturation de μ direction_

Run 420 lancé 2026-11-11 14:21 dans `outputs/train_20_sac` (Samuel s'est trompé de nom ; ne pas renommer
pendant qu'il tourne). Config identique à 419 ; code = aed53757 (StateEncoder partagé par les jumeaux,
acteur sur encodage détaché), confirmé par les repr des checkpoints.

CSV à 41k (D=250) : hit ≈ 0.01 jusqu'à 8k (419 : 0.05), puis rattrapage à 0.065 sur 30–42k (419 : 0.070),
record isolé 0.077 ; Hc −5 à −6 (cible −3.5) ; Q +0.11.

Sonde save_1 (27k, côtés 250/550, 4 épisodes) : ∂Q/∂a_canon armé P(bon sens) 0.50–0.52, jumeaux en
désaccord de signe (0.45–0.49), comme 417/419 → l'encodeur partagé ne lève pas le verrou de la visée
continue. Gâchette sélective (p_fire 0.50 sous 5° → 0.24 au-delà de 45°, ratio 2.2/2.4) ; dump inchangé
(réserve vide à 3.6 s, 73 % désarmé) ; σ canon 0.48 ≫ cible 0.1, μ direction saturé 25–29 %.

**Why:** la sélectivité n'a pas de référence (pas de sonde 417/419 avec le script probe-checkpoint),
donc elle n'est pas attribuable à l'encodeur partagé.

**How to apply:** avant de créditer l'encodeur partagé, sonder 419 save_1 avec le même script ; suivre
`ratio` et P(∂Q/∂a) sur save_2+. Voir run 419, sac-agent-2026-11,
run 416 (saturation post-tanh).

# train_419_sac

| | |
|---|---|
| machine | the-machine |
| utilisateur | samuel |
| lancement | 2026-11-11 11:08:12 |
| commit au lancement | `f9342001` — returns for SAC computed on multiple steps (10 nexts for the moment) |
| commits ≤ 90 min après | aucun |
| diff au lancement | inconnu (identité relevée après la fin du run) |
| données brutes | `outputs/train_419_sac` sur the-machine |

## Faits établis

_SAC run 419 (2026-11-11) = 417 + n_step 10, seule diff de config ; à 17k pas : Hc à la cible (−3.4) dès 12k (415 : 36k), perte q ×8, Q ≈ 0 (biais −0.09 disparu), Hd 0.15, hit 0.053 = plancher aléatoire ; trop tôt, juge = sonde ∂Q/∂a sur save_1 (27k)_

Run 419 lancé 2026-11-11 10:52, config identique à 417 sauf `n_step` 10. Lu à 17k pas (1 h 23).

Moyennes 9k–17k (415 / 417 / 419) : hit 0.075 / 0.059 / 0.053, fire 0.68 / 0.49 / 0.57 obus/s,
miss 21.2 / 21.8 / 22.7 m, perte q 0.0036 / 0.0030 / 0.023, Hd 0.23 / 0.30 / 0.15. D = 250 partout.
Hc atteint −3.3 (≈ cible σ 0.1 sur 4 axes) dès 12k ; Q passe de −0.09 (415/417) à ≈ +0.01.

**Why:** perte q ×8 attendue (cible = somme de 10 rewards bruitées) ; à 17k le hit est au plancher
aléatoire comme 417, donc rien n'est encore mesurable côté visée.

**How to apply:** ne pas conclure sur les métriques CSV avant ~50k ; le test décisif est la sonde
∂Q/∂a_canon (qgrad.py) sur save_1 : P(bon sens) armé ≫ 0.5 (417 : 0.47–0.50).
Voir [[sac-train-417-diagnosis]], [[sac-agent-2026-11]].

2026-11-11, sonde ∂Q/∂a sur save_1 (27k, D=250, 2 épisodes), 419 vs 417 save_1 au même stade :
P(∂Q/∂a bon sens) armé 0.51 vs 0.52, toutes tranches 0.49–0.56 vs 0.50–0.54, corr ≈ 0 dans les deux ;
|∂Q/∂a| ×3 (2.5e-3 vs 8e-4) ; P(μ bon sens) sur |yaw| 10–30° 0.54 vs 0.64 ; armé 30 % vs 15 % des
décisions. → à 27k, le n-step rend le critique plus sensible à l'action sans lui donner de direction.
Même le terme de potentiel (dépendant de l'action dès s_{t+1}) n'apparaît pas dans ∂Q/∂a.

2026-11-11, jumeaux (sonde : colonnes dq1/dq2 par critique, 1 épisode D=250, save_1) : accord de signe
Q1/Q2 sur ∂Q/∂a_canon 0.48–0.55 et corr(Q1, Q2) 0.03–0.16 dans 419 comme dans 417 (même armé, |yaw|<10).
→ la dépendance de Q à l'action est du bruit d'ajustement propre à chaque critique, pas un signal
partagé. Ordre de grandeur attendu du seul potentiel armé |yaw|<10 : ~1e-2 à 1e-1, mesuré 8e-3 de signe
aléatoire. Poids de la 1re couche : colonnes action ≈ init, pas d'atrophie. Piège sonde : un seul
autograd::grad à plusieurs sorties renvoie le gradient de leur somme.

2026-11-11, sonde de features (probe --features exporte pour SAC : acteur [vision 192 | capteurs 64],
puis Q1, Q2 ; vision = SpatialSoftmax [x 64 | y 64 | présence 64]), 1 épisode D=250, split par tank,
cible = tank le plus proche. Ridge yaw (|yaw|<30°, <60 m) : R² 0.02 (419 s1), 0.14–0.16 (417 s1),
−0.11 (417 s5) ; MLP pire (surapprend). Signe du yaw à |yaw|<10° : 0.52–0.57 partout. Détection
« tank < 40 m dans le champ (|yaw|<15°, |pitch|<10°) » AUC 0.74–0.86. Contrôle : capteurs → réserve
R² 0.67, alignement CSV/bin exact. → encodeurs SAC (acteur ET critiques) détectent vaguement la
présence d'un tank mais ne codent pas son gisement, et ça ne s'améliore pas avec l'entraînement.
Limites : pas de témoin positif (aucun run PPO local), petit n, cible = plus proche ≠ plus saillant.
Distinct de [[cnn-perception-validated]] (CNN PPO) : ici CNN SAC entraîné par TD seul.

# train_433_ppo_liquid

| | |
|---|---|
| machine | the-machine |
| utilisateur | samuel |
| lancement | 2026-11-23 22:49:10 |
| commit au lancement | `f7435cc2` — update harness |
| commits ≤ 90 min après | aucun |
| diff au lancement | inconnu (identité relevée après la fin du run) |
| données brutes | `outputs/train_433_ppo_liquid` sur the-machine |

## Faits établis

_run 433 (CNN 4 blocs + spatial softmax + tête de présence) — sonde save_14 à D 850, comparaison au rapport 552 de l'autre PC_

Run 433 `ppo_liquid` (lancé 2026-11-23, code = f7435cc2 + diff indexé non commité) : CNN 4 blocs + spatial softmax (x, y, amplitude) + présence sur la carte lue. Référence = run 552 de l'autre PC (7 blocs aplati), connu seulement par `2026-11-24_rapport_visee_552` (pas de CSV/checkpoint ici).

Sonde save_14 (378 k, D 850), 2026-11-24 :
- position à l'entrée de la politique 0,87° / 1,51° / 4,18° (60-100 / 100-200 / 200-400 m), ≥ 552 figé (1,26 / 2,19 / 5,34°) malgré un jeu de lecture 2× plus petit ; affiné 0,78 / 0,94 / 1,75° ; température du softmax restée ≈ 1 ;
- mais visée inchangée : tour<5° 0,185 (550) / 0,133 (1000) ≡ 552 ; corr(μ canon_x, lacet) ≈ 0 à 200-400 m, −0,1 à −0,26 à 100-200 m ; crédit du geste ρ ≈ 0 ; hit/obus 0,15-0,16 ;
- gâchette sélective (ratio 23 côté 550) mais ne suit pas une visée oracle (0,16-0,36 obus/tank-s contre 1,2-1,3 pour 552 s41).

**Why:** la représentation n'était pas le seul verrou ; précision disponible ≠ utilisée, cohérent avec [[ppo-liquid-train-432-diagnosis]] et [[cnn-perception-validated]].
**How to apply:** comparer à 552 seulement à D 1000 ; les chiffres du banc de position ne se comparent qu'à taille de jeu égale (étalon « 7 blocs neufs » à relancer). `probe_checkpoint.py --check-only` ne voit pas un binaire plus vieux qu'un diff indexé : vérifier l'heure du binaire à la main.

Suite 2026-11-24 : D 850 → 800 à 429 k, kill 21 → 12-15, fire 1,0 → 0,2-0,6, hit monte à 0,19. Sonde save_17 (459 k) vs save_14 : visée inchangée ; gâchette fermée partout sauf < 40 m (p_fire aligné toutes distances 0,30 → 0,19 côté 550, 0,091 → 0,041 côté 1000 ; 0,74 inchangé à < 40 m) ; sous visée oracle elle tire encore moins (0,25 / 0,09 obus/tank-s). Même motif que [[ppg-train-428-diagnosis]] (hit ↑ par extinction, kill ↓). Cause non établie.

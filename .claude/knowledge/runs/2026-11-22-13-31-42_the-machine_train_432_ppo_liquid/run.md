# train_432_ppo_liquid

| | |
|---|---|
| machine | the-machine |
| utilisateur | samuel |
| lancement | 2026-11-22 13:31:42 |
| commit au lancement | `dcca274e` — bootstrap the value on the last state at the episode time limit (GAE and return scaler no longer run into the next episode), continuous target entropy -1 on the four actions, shells_per_hit default read from GAME_SHELL_ECONOMY, critic value and per-step rewards in probe exports |
| commits ≤ 90 min après | `7bccaa04` |
| diff au lancement | inconnu (identité relevée après la fin du run) |
| données brutes | `outputs/train_432_ppo_liquid` sur the-machine |

## Faits établis

_run 432 liquid, groupe d'optim unique pour la politique — visée en gisement apprise avant 27k puis gain plat ; méthode et convention de signe pour mesurer l'asservissement du canon_

Run 432 (ppo_liquid, lancé 2026-11-22, non commité au-dessus de dcca274e) : un seul groupe d'optim
pour la politique (plus d'avantage de visée ni de KL par groupe), confondu avec cible d'entropie
continue −1 et bootstrap à la limite de temps.

Sondes côté 350, save_0 / save_1 (27k) / save_3 (81k) : tour<5° 0.053 / 0.154 / 0.188, hit/obus
0.029 / 0.122 / 0.135. corr(μ_canon_x, erreur de gisement) à 10-20° : +0.08 / −0.30 / −0.26, soit
~5 % de l'erreur fermée par décision, identique à 27k et 81k. La visée en gisement est donc
apprise tôt puis son gain stagne ; le progrès de tour<5° vient de σ canon (0.19 → 0.09). Gâchette
peu sélective (ratio 1.27), 83 % vident la réserve en 3.4 s.

save_6 (162k, D=450) : gain de visée toujours plat (3e sonde), tour<5° 0.175 ; c'est la gâchette
qui apprend (ratio 2.43, p_off 0.59 → 0.32, réserve vidée à 6.9 s, hit/obus 0.162) ; σ canon
étalé 0.02-0.48 selon l'état.

save_13 (351k, D=800) : gâchette fermée (CSV fire 1.40 → 0.35 dès 260k, kill 25.8 → 15.6) ; p_off
0.02-0.03 quelle que soit la réserve, p_fire <5° 0.24-0.33 alors qu'un obus aligné touche 1/2 ;
à côté 350 égal touches/s ÷2, tour<5° 0.175 → 0.077, tanks 2× plus éloignés, σ canon 0.48.
Pénalité réserve vide = −D/1000 terminale (ee1ca624) : rend le hors-axe déficitaire dès D=250
(0.025/obus vs ~0.01 d'espérance), mais n'explique pas la fermeture en état aligné, et 428 (pénalité
fixe) avait fermé pareil.

**How to apply:** pour mesurer la visée, régresser μ_canon_x sur `yaw` (erreur signée vers la
cible la plus proche) par tranche d'erreur dans le CSV de sonde, et toujours sonder save_0 comme
référence. Convention mesurée : action canon_x +1 ⟹ erreur +12.7°/décision, le bon sens est une
corrélation **négative** (les « corr ≈ 0 » de run 423 étaient globales, or le
signal n'existe que sous ~45°). La colonne `pitch` ne réagit pas à canon_y : le site n'est pas
mesurable ainsi. Script : scratchpad jetable, à réécrire (20 lignes pandas, `.venv/bin/python`).

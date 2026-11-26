# train_416_sac

| | |
|---|---|
| machine | the-machine |
| utilisateur | samuel |
| lancement | 2026-11-10 15:31:00 |
| commit au lancement | `39ae97f9` — update probe for actor (no liquid) |
| commits ≤ 90 min après | aucun |
| diff au lancement | inconnu (identité relevée après la fin du run) |
| données brutes | `outputs/train_416_sac` sur the-machine |

## Faits établis

_SAC run 416 (2026-11-10, 415 + target_sigma 0.3 + replay stratifié 70 % armé + plafond σ ≤ 0.7) — à 69k : plus rapide à D=300 (hit 0.104) mais pic plus bas (0.114) et recul dès D=350 (0.103 ; 0.096 à D=400), kill 8 vs 12. Sonde save_2 : σ canon collé au plafond 0.7 sur 94 % des décisions, bruit d'action ×1.7, angle au tir médian 19.6° vs 11.8° ; p(fire|can_fire) 0.88, réserve vidée en 2 s, à sec 84 % du temps ; corr(μ, erreur) ≈ 0_

Run 416 (lancé 2026-11-10 15:31, analysé à 69k pas / 4,6 h, encore en cours). Trois changements
simultanés vs 415 : `target_sigma` 0.1 → 0.3 (cible Hc −3.58 → +0.52 sur 4 axes), replay
stratifié `--armed_fraction 0.7` (non commité), et le plafond `LOG_SIGMA_MAX` −0.3567 (σ ≤ 0.7,
commit e52e76ff 09:25, postérieur au save_0 de 415 à 08:28 → 415 tournait sans).

CSV, comparaison à D égal (moyennes par palier) :
- D=300 : hit 0.104 vs 0.090, kill 5.6 vs 4.3 — 416 apprend plus vite.
- D=350 : hit 0.103 vs 0.117, kill 8.0 vs 11.9, fire 0.79 vs 1.54 obus/s, miss 25.9 vs 28.6.
- D=400 (3 lignes) : hit 0.096 vs 0.111, kill 7.8 vs 13.2.
Pic hit 0.114 à 37k (D=300) puis recul continu ; 415 culminait à 0.121 à 60k (D=350).
Hc collé à sa cible 0.52, αc stable 5e-4 ; αd0 monte 2× plus vite que 415 (5.2e-3 à 46k) puis
redescend. Q démarre à −0.47 (états armés au tir aléatoire) et finit −0.08 vs −0.33, mais mesuré
sur un batch stratifié : non comparable.

Sonde save_2 (54k, D=350, 2 épisodes, échantillonnée) vs 415 save_2 :
- **σ canon = 0.70 (le plafond) sur 94 % des décisions** (415 : 10 %, σ 0.38/0.52), |μ| 0.43 vs
  0.79–0.97. La cible d'entropie σ 0.3 est atteinte avec σ pré-tanh au max et μ modéré ; bruit
  d'action (std de act − tanh μ) 0.48 vs 0.26–0.30. Angle au tir médian 19.6° vs 11.8°, 75 % des
  obus à > 10° (415 : 58 %). Hit par obus à angle égal identique à 415 (0.21/0.19/0.15/0.07/0.02)
  → la perte de hit vient entièrement de l'alignement plus bruité.
- **p(fire | can_fire) 0.88** (415 save_2 : 0.43, save_3 : 0.81), réserve vidée en 2 s, à sec
  84 % des décisions (415 : 53 %), vies 20 % plus longues → 0.77 vs 1.42 obus/s de vie. La chute
  de cadence n'est PAS une tête de tir plus stochastique (hypothèse CSV réfutée) : c'est moins de
  hits → moins de recharges, et une vie à sec plus longue.
- corr(μ canon, erreur yaw/pitch) −0.06 / −0.01 : toujours aucune visée (comme 415).

Observation structurelle : dans les deux runs l'acteur satisfait la cible d'entropie post-tanh par
saturation, jamais par un σ petit — 415 par |μ| → 1 (tanh écrase), 416 par σ au plafond. La cible
σ 0.3 laisse donc un bruit pré-tanh bien plus grand que voulu.

**Why:** 416 n'améliore pas 415 ; le facteur dominant est `target_sigma` 0.3 (via la saturation du
σ au plafond), pas la stratification, dont l'effet reste non isolé.

**How to apply:** pistes non décidées, à valider avec Samuel : revenir à `target_sigma` 0.1 en
gardant le replay stratifié (isole la stratification) ; ou définir la cible d'entropie pré-tanh
(gaussienne seule) pour qu'elle ne soit plus contournable par saturation. La sonde
`arenai_agent/probe/main.cpp` a été réparée (SacActor → Actor, non commité) et compilée dans
cmake-build-release. Voir [[sac-train-415-diagnosis]], [[sac-agent-2026-11]].

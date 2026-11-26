# train_415_sac

| | |
|---|---|
| machine | the-machine |
| utilisateur | samuel |
| lancement | 2026-11-10 08:30:12 |
| commit au lancement | `a15ee8f1` — sac : follow develop (log-sigma constants, alphas written to the CSV only) |
| commits ≤ 90 min après | `15b32ece` |
| diff au lancement | inconnu (identité relevée après la fin du run) |
| données brutes | `outputs/train_415_sac` sur the-machine |

## Faits établis

_SAC run 415 (2026-11-10, α₀ 5e-4) — phase morte supprimée, hit 0.12 à D=350 puis 0.11 à D=400 (91k pas) ; sonde : tir quasi inconditionnel, visée non corrélée à l'erreur, hit = alignement fortuit (<2° → 0.30, >30° → 0.01) ; Q dérive négatif (−0.34) alors que le retour réalisé est positif_

Run 415 (config 414 sauf α₀ 5e-4, lr α 3e-5, 32 tanks, 128×64, D 250→400) analysé à 91k pas / 6 h
(encore en cours). Saves tous les 27k : save_1 = 27k, save_2 = 54k, save_3 = 81k.

CSV : l'apprentissage démarre dès ~3k (hit 0.03 à 2.7k, 0.08 à 19k) — **la phase morte de 414 était
bien la décroissance d'α**. Hc 2.0 → −3.5 à 43k puis plateau −3.5/−4.2 (α_c 4.8e-4 → 5e-5 à 44k,
puis remonte à 1.3e-4 : la régularisation agit). Curriculum 300 (21k), 350 (48k), 400 (67k). Pic hit
0.121 / kill 14.7 à 58–66k (D=350) ; à D=400 : hit 0.109–0.112, kill 12.3, miss 31 → 28 m. Pas la
régression de 414 (0.117 → 0.067). α_d0 (fire) monte sans arrêt 5e-4 → 3.7e-3 : l'entropie du tir
reste sous sa cible. **Q (q1 aux actions du replay) décroît de −0.06 à −0.34 pendant que r/décision
monte à 0.0038 (V ≈ r/(1−γ) ≈ +0.4)** : sous-estimation ≈ 0.7, miroir de la surestimation +0.4 de
414 ; non expliquée (bonus d'entropie ≈ 0.1 au plus). Perte critique q ×10 (0.0017 → 0.019).

Sonde (D=350, 2 épisodes, échantillonnée) sur save_1/2/3 :
- fire quasi inconditionnel : p(fire | can_fire) 0.79 / 0.43 / 0.81, dépend à peine de l'angle
  (save_3 : 0.91 à <2°, 0.72 à >30°). Réserve vide 83 / 53 / 71 % des décisions, can_fire 0.10–0.34.
  save_2 (le meilleur, hit obus 0.122) est le seul sélectif (p 0.29 au-delà de 30°).
- hit par obus selon l'alignement au tir (save_3) : <2° 0.31 (miss 10 m), 2–5° 0.25, 5–10° 0.15,
  10–30° 0.08, >30° 0.01 (miss 45 m) ; 59 % des obus partent à >10°. Distance : <100 m 0.19, >100 m
  0.04–0.06 ; 45 % des tirs à <100 m.
- **corr(mu_canon, erreur yaw/pitch) ≈ −0.05** : la commande de visée ne réagit pas à l'écart cible ;
  |yaw| médian vers le tank le plus proche 80°. Le hit = alignement fortuit sur le meilleur des 31 tanks.
- σ pré-tanh 0.42–0.50 avec |μ| 0.77–0.95 (saturation |act|>0.9 : 17 % lacet, 25 % site) : la cible
  d'entropie σ 0.1 est atteinte par saturation du tanh, pas par un σ petit.

**Why:** le gain 414 → 415 vient de l'α initial ; le mécanisme reste "tirer ce qui passe devant",
comme les PPO (voir ppo-actor-audit-2026-10-29). La sonde `arenai_agent/probe` charge désormais
aussi l'acteur SAC (branche `is_sac` = absence de `neuron_number` dans config.json ; modif non commitée
au 2026-11-10).

**How to apply:** pistes à valider avec Samuel, pas décidées : vérifier le signe de Q (retour MC vs
Q dans la sonde), comprendre pourquoi p(fire) remonte à 0.8 malgré α_d0 croissant, tester save_2 en
jeu. Voir run 414, sac-agent-2026-11.

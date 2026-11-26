---
name: reward-redesign-2026-11-06
description: "Refonte complète du reward validée par Samuel le 2026-11-06 (dégâts en fraction de PV, potentiel de visée, famine supprimée, coût de tir fixe, baselines EMA supprimées, win supprimé, économie 10/+2/20) et le raisonnement d'incitation derrière"
metadata:
  node_type: memory
  type: project
  originSessionId: 9d59aa39-f9a4-464d-b746-df05b503fcf4
  modified: 2026-11-06T18:19:33.893Z
---

Refonte du reward implémentée le 2026-11-06 sur develop (choix « Refonte complète » de Samuel,
sans run sonde visée-seule préalable). Remplace les diagnostics 382-395 comme référence courante.

- Diagnostic retenu : dump et famine étaient *optimaux* — obus à espérance nulle (baselines EMA
  aim/hit) et sans valeur future ; famine = troncature gratuite → passivité vaut 0, s'engager < 0
  au plancher aléatoire ; visée payée au tir, jamais au pointage.
- Reward (jolt_enemy_tank.cpp, constants.h) : hit = DAMAGE_REWARD(1) × fraction de PV retirée
  (0.2 roue/tourelle/canon, 0.1 châssis) ; reçu = même fraction en négatif ; kill +1 / mort −1
  (suicide retourné inclus) ; FIRE_COST 0.03 à la mise à feu (0.005 au run 407, monté au-dessus
  de l'espérance aléatoire ~0.02 le soir même) ; plus de win, plus de baselines.
- Potentiel de visée (arenai_agent core/potential_shaping, restauré de 682bc8f3) : Φ = Cauchy du
  miss prédit depuis la pose canon courante (`EnemyTank::get_aim_potential`, [0,1]), F = γΦ'−Φ,
  AIM_POTENTIAL_SCALE 0.3 (choix Claude, non discuté), payé à chaque décision, γ lu dans la
  config agent. Pourquoi le potentiel avait été retiré le 03/11 : inconnu, Samuel n'a pas répondu.
- Famine supprimée : plus d'`apply_timeout`/`is_timeout`, l'épisode à durée fixe (90 s) tronque
  les survivants via final_state ; `IsTruncated` du cœur est toujours false (plomberie gardée).
- Économie GAME_SHELL_ECONOMY 10 / +2 par hit / max 20 (le jeu suit).
- `ShellContactInfo.damages` = dégâts réels infligés ; `LifeItem::initial_health()` ;
  `RewardDetail` : hit, kill, received, death, fire, nb_hits (plus d'aim ni d'opportunité).

**Why:** Samuel voulait « repenser complètement » après que hit +0.3/reçu −0.2 n'a pas levé
dump/famine sur l'autre PC. Le principe : seul l'objectif du jeu dans le reward, le gradient de
visée via un potentiel policy-invariant, aucune sortie gratuite de l'épisode.
- Run 407 PPG (0.005, 73k décisions, arrêté par Samuel) : hit/fire 0.06 → 0.157 en 30k (jamais vu
  à D=250) puis plateau, mais dump confirmé (fire = 0 hors des premières secondes, Hd 0.33 → 0.39,
  s 14 → 7.5) et Hc 0.21 → 0.07 (motif 383). Samuel : « si il dump c'est foutu ». Mort à réserve
  vide refusée comme anti-dump (preuve 378/392 : ne déplace que la valeur du dernier obus).
- Métrique `s` = âge moyen des morts seulement (survivants non comptés depuis la fin de la famine) ;
  4.0 s = retournés au spawn. Colonnes r/r_hit/r_rcv/r_aim illisibles (fenêtre 691k, 3 décimales).

**How to apply:** juger les prochains runs sur hit/fire, miss, r_aim (shaping) et la cadence de
tir dans les premières secondes de vie (dump) ; si la visée ne monte pas malgré le potentiel,
le verrou est la perception → sonde visée-seule. Voir [[aim-fire-levers-audit-2026-11]],
[[ppg-train-395-diagnosis]], [[fire-cost-per-shell]], [[out-of-shells-termination]].

---
name: ppo-reward-redesign-2026-08
description: "Diagnostic non-convergence PPO (2026-08) et refonte reward validée — réserve d'obus, phi angle gaussien; entropie inchangée par choix de Samuel"
metadata:
  node_type: memory
  type: project
  originSessionId: 00b21891-b35b-4c4e-a1c1-c94e1bedfd5e
  modified: 2026-09-05T10:56:32.486Z
---

**Samuel n'utilise plus SAC : PPO est l'algorithme d'entraînement courant** (confirmé
explicitement le 2026-08-26). Les diagnostics SAC ([[sac-train-293-diagnosis]],
[[sac-train-309-diagnosis]]) sont historiques.

Diagnostic (2026-08-06) du plateau PPO à 300k steps : politique épinglée près de l'entropie
max (Hc≈2.0/2.77, Hd≈0.61/0.69), miss_cost rendait le tir net-négatif (break-even ~20 % de
précision vs 0,5 % réel), terme d'angle de φ `(cosθ+1)/2` quasi plat → le hit ne pouvait pas
s'amorcer. En self-play symétrique 32 tanks, `r` moyen est ~conservé : suivre **hit/fire**
(précision) et `s` (décollage du plafond de troncature 30 s sans hit), pas `r`.

Refonte validée par Samuel :
- réserve d'obus 30, +5 par hit, vide = tir refusé (prédicat can_fire dans CanonItem,
  joueur illimité) ; miss_cost supprimé ; réserve exposée en proprioception (ratio, taille 133)
- φ : angle_score = exp(−½(θ/10°)²)
- métrique fire = obus réellement tirés (has_fired_shell), ActionStats supprimé du code

**Why:** Samuel a explicitement refusé l'annealing des coefs d'entropie (piste C) — ne pas
re-proposer sauf si la précision monte alors que Hc/Hd restent au max.

Comportement attendu (2026-08-07) : spam de tir en début d'épisode tant que hit/fire est bas
— tir = avantage positif (gaussienne ≥ 0, coût nul), la réserve ne taxe que via V(réserve)
qui n'apprend qu'une fois les hits fréquents. Ne pas re-signaler. Vigilance : γ=0.997
(horizon ~11 s) — si le premier engagement après spawn arrive à > ~20 s, le coût
d'opportunité de la réserve vide est hors horizon et retenir le tir au spawn peut ne jamais
devenir optimal.

Ajustements validés le 2026-09-05 (analyse reward pour la convergence) :
- aim centré : `0.1 × (quality − 0.5)` — un obus qui retombe loin coûte, le subside
  inconditionnel au tir est supprimé (le « spam positif » attendu ci-dessus ne tient plus)
- mort de famine (timeout) = −2, tué/suicide = −1 : mourir caché doit coûter plus cher
  que mourir au combat
- bonus win +2 conditionné à ≥ 1 kill dans l'épisode (nb_kills_per_tanks)
- observation timer famine normalisée par max_frames_without_hit (plage [0,1] pleine ;
  les vieux checkpoints voient une autre échelle sur cette entrée)

**How to apply:** pour juger la convergence des prochains runs, regarder hit/fire et s ;
si blocage persiste, les pistes restantes discutées : curriculum d'adversaires immobiles,
queue épaisse sur la gaussienne de near-miss. Voir [[sac-train-counter-known-issue]] pour
un autre choix assumé à ne pas re-signaler.

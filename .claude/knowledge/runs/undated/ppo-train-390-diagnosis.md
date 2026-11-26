---
name: ppo-train-390-diagnosis
description: "Run 390 (ppo_liquid, p(fire) init 0.5, lu à 78k steps) — 10 obus partent en ~3 s, hit plat 0.065, miss 20 m ; recommandation initial_fire_proba 0.1 (plancher 0.05 à cause de la famine 30 s) ; Hd dominé par le zoom, illisible pour le tir"
metadata:
  node_type: memory
  type: project
  originSessionId: 0f29c4ea-1ad8-4117-bb16-b86533e1d8a5
  modified: 2026-11-01T18:18:08.649Z
---

Run `outputs/train_390_ppo-liquid` (2026-11-01 17h16, lu à 78k steps, à comparer à [[ppo-train-389-diagnosis]]).
Binaire compilé après l'édition non commitée out_of_shells -0.5, donc probablement -0.5 dans ce run.

- hit 0.065 → 0.047 quand D passe à 350, miss 20-22 m, Hc 0.21 → 0.14, ev 0.2-0.5.
- Calcul : décision 10 Hz + cooldown 5 frames → intervalle moyen 0.1·(1+1/p) s. p=0.5 : 10 obus en
  3 s ; p=0.1 : 11 s ; p=0.05 : 21 s ; p=0.03 : 34 s, au-delà de la famine (30 s).
- **Hd ne se lit pas pour le tir** : moyenne sur les décisions non masquées, le zoom y domine.

**Why:** Samuel a demandé si l'acteur devait démarrer à p(fire) ≤ 0.1. Hypothèse : tirs moins corrélés ; pas une
preuve, et cela ne traite pas directement la visée.

**How to apply:** tester avec `--initial_fire_proba 0.1` (flag existant, ne change pas le défaut). En jeu
l'action est p > 0.5 : à 0.1 l'agent peut ne jamais tirer tant que l'état ne remonte pas le logit de +2.2.

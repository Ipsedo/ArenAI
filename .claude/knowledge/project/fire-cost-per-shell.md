---
name: fire-cost-per-shell
description: "2026-11-01 coût de -0.03 par obus tiré, facturé au tir dans JoltEnemyTank::get_reward ; valeur d'un obus estimée à +0.08 ; risque : ne plus tirer est sûr tant que la famine reste sans pénalité"
metadata:
  node_type: memory
  type: project
  originSessionId: ddf7f066-7a3c-44fe-9404-af636009ed4a
  modified: 2026-11-01T19:54:17.551Z
---

2026-11-01 : Samuel a validé un coût de -0.03 par obus, appliqué au tir, sans métrique dédiée. À sa demande, il vit dans `JoltEnemyTank::get_reward` avec le reste de la récompense (`fire_cost`, compteur `nb_uncharged_shells`) et non dans `TrainTankEnvironment`. Il n'est pas dans `RewardDetail` : le contrat public ne change pas.

**Why:** rendre le tir sélectif. Un obus valait environ +0.08 en espérance même raté (visée ~0.06, hit ~0.018, estimation calculée à partir du code et des runs [[ppo-train-389-diagnosis]] / [[ppo-train-390-diagnosis]], pas mesurée). À -0.03, le bonus de visée ne couvre plus le coût au-delà d'un miss de ~50 m à 300 m.

**How to apply:** `r` n'est pas comparable aux runs antérieurs. Juger sur `hit` qui monte pendant que `fire` baisse ; le coût ne donne aucun signal de direction, donc `miss` n'est pas le bon indicateur. Si `fire` s'effondre sans gain de `hit`, le coût est trop élevé : la famine étant une truncation sans pénalité, ne plus tirer est une option sûre (voir [[out-of-shells-termination]]). Avec `initial_fire_proba` à 0.03, l'intervalle de tir (~34 s) dépasse la famine (30 s).

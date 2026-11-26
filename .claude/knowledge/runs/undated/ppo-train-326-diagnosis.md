---
name: ppo-train-326-diagnosis
description: "PPO vanilla Beta (train_326) — pas de coquille d'inférence ; politique restée quasi-uniforme (Hc≈0.65), mode/argmax en jeu ≠ politique échantillonnée"
metadata:
  node_type: memory
  type: project
  originSessionId: 988732ed-3afd-427f-b7ba-2661f0c7f9b5
  modified: 2026-09-17T19:26:00.279Z
---

Diagnostic 2026-09-17 : le PPO vanilla (train_326_ppo_beta_trunc, commit db5f02a1 « beta law
for vanilla actor ») ne vise pas et ne tire pas en jeu malgré des métriques d'entraînement
« convergées ». Analyse complète : **aucune coquille** dans le chemin d'inférence — agents,
collector, trainer, save/load, factory et boucle de jeu sont symétriques avec ppo_liquid
(qui marche). Checkpoint sain (poids bougés, sorties dépendantes de l'entrée).

Cause : en jeu l'agent joue en déterministe (`beta_law_mode_action` + argmax tir) ; les
métriques d'entraînement mesurent la politique **échantillonnée**. Or Hc final du 326 ≈ 0.65
≈ log 2 = 0.693 (entropie max d'une Beta sur [−1,1]) ≈ l'entropie à κ init = 3 : la Beta est
restée quasi uniforme on-policy pendant 2M steps (Hc était même redescendu à 0.556 puis
remonté). Les hits (~0.064/s, équivalents au liquid) venaient du spray aléatoire ; le mode
d'une Beta quasi-uniforme est arbitraire → canon en bas. Hd ≈ 0.36 → p(tir) moyenne ≈ 0.1 <
0.5 → argmax ne tire jamais. Le liquid (train_325) : Hc 0.42 et en baisse → mode significatif.

Règle de lecture : pour prédire le comportement en jeu, regarder **Hc et Hd**, pas seulement
hit/fire/s. Hc proche de 0.69 = déterministe sans rapport avec l'échantillonné.

Pièges structurels notés (pas des bugs stricts) : à κ→2, ∂logp/∂mode ∝ (κ−2) → tête mode sans
gradient ; la tête κ (sigmoid log-échelle [2.01, 2000]) sature au plancher ; le PID Lagrangien
α ≥ 0 ne pousse jamais l'entropie vers le bas → rien ne force κ à monter si l'avantage ne le
fait pas. Voir [[ppo-beta-entropy-init]], [[ppo-reward-redesign-2026-08]].

Landmine `"hidden_sizes"` **corrigée le 2026-09-17** (migration config.json en JSON natif) :
`AgentFactory` lit désormais `"actor_hidden_sizes"`, tous les champs sont requis (`at()`,
plus de défauts silencieux) et les tableaux sont des arrays JSON natifs — les anciens
config.json à valeurs string ne se chargent plus sans migration.

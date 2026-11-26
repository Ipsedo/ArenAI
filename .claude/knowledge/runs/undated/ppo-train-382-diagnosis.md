---
name: ppo-train-382-diagnosis
description: "Run 382 (ppo vanilla, 32 tanks, sans aim reward, out_of_shells terminal) — curriculum monte 250→500 sur le plancher aléatoire (seuil 0.06 < hit aléatoire 0.05-0.06 à D=250), biais Bernoulli inchangé (p≈0.5) mais logits dépendants de l'état saturés (Hd 0.15), σ figé 0.5, 0 kill en jeu"
metadata:
  node_type: memory
  type: project
  originSessionId: 78e69fb6-b9f9-4692-83f0-49ca42cd00a5
  modified: 2026-10-31T01:48:47.373Z
---

Run `outputs/train_382_ppo` (2026-10-30/31, ppo vanilla, 32 tanks, entropie c 0.01 / d 0.001,
initial_fire_proba = initial_zoom_proba = 0.5, curriculum ratio_low 0.03 / ratio_high 0.06,
probe_window 24, boundary 0.2). Testé en jeu (save_4) : aucune visée, aucun kill.

- **Curriculum faux positif** : à D=250 avec 32 tanks, une politique quasi aléatoire fait déjà
  hit/fire ≈ 0.05-0.06 (steps 1k-10k). Le seuil 0.06 est franchi vers 8k steps sans apprentissage
  de visée ; D monte 250→500 alors que hit reste 0.06-0.10 et miss passe de 20 à 26 m.
  Le signal hit/fire mesure la densité de cibles, pas la visée.
- **Tête Bernoulli** : le biais ne bouge pas (sigmoid(biais) ≈ 0.49/0.52 dans tous les saves,
  init orthogonal 0.01 + biais logit), mais les poids grandissent (normes 1.9/1.4) → probas
  dépendantes de l'état, saturées (Hd 0.68 → 0.15). fire 5.9/s à l'init → 0.5-0.7/s ensuite :
  l'agent rationne ses 10 obus sur ~15 s. Épisodes s = 13-30 s = mort par out_of_shells (-0.3),
  max 180 s. En jeu = argmax (sample=false) : tire dès que p > 0.5, cf. [[ppo-train-326-diagnosis]].
- **σ figé** à l'init : biais sigma 2.6 → σ ≈ 0.5 sur [-1,1], comme [[ppo-train-371-diagnosis]].
  La moyenne μ ne reçoit aucun signal exploitable → en jeu (action = moyenne) aucune visée.
- kill 0.2-1.4 / épisode pour 32 tanks ≈ rien.

**Why:** Samuel se demandait si l'init Bernoulli à 0.5 était la cause ; ce n'est pas le levier
(le biais reste à 0.5, les logits sont pilotés par l'état). Les vrais blocages : seuils de
curriculum sous le plancher aléatoire, et σ figé.

**How to apply:** avant toute lecture d'un curriculum, mesurer le hit/fire aléatoire à D initial
et placer ratio_low/high au-dessus. Lire les biais des têtes avec un petit outil LibTorch
(torch::jit::load sur actor.pt, pas de torch python dans la venv).

**Suite (2026-10-31, validée par Samuel)** : levier bruit = initial_sigma 0.5 → 0.05 (bruit ≈ 0.9° lacet / 0.36° site par décision, sous la taille d un tank à 250 m : 1.5°×0.7°) + continuous_entropy_factor 0.01 → 0.001, pour les 3 agents (ppo, ppo_liquid, ppg) ; out_of_shells -0.3 → -0.1. Levier perception (spawn 100 m / zoom) refusé : « déjà convergé avec le même CNN ». Marqueur à lire au prochain run : miss < 15 m dans les 200k premiers steps.

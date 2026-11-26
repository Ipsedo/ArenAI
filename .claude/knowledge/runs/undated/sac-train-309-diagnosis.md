---
name: sac-train-309-diagnosis
description: "train_309_sac ne converge pas — hits uniquement pendant la phase haute-entropie initiale, α effondré en ~30k steps, buffer 300k = ~10 épisodes (quasi on-policy), spray+famine = retour net positif"
metadata:
  node_type: memory
  type: project
  originSessionId: 5f4519c6-b897-4f67-9c43-2cd8ad263043
  modified: 2026-08-21T16:59:16.840Z
---

Diagnostic du 2026-08-21 sur `outputs/train_309_sac` (~295k env steps × 32 tanks, 330 épisodes) :

- hit/s : 0.02–0.06 au début (α_c=0.046, Hc≈0.69) → ≈0 dès ~50k steps, jamais revenu. r décroît de façon monotone 0.0013 → 0.0002. s_μ collé à 27–30 s tout le run = mur de famine (comme [[sac-train-293-diagnosis]]).
- α_c : 5e-2 → ~3e-4 en ~30k steps (entropies convergées vers cibles basses : target_sigma, fire 2.5 %). Les seuls hits du run venaient de la politique initiale quasi uniforme.
- Buffer : 300 000 / 32 tanks = 9 375 steps ≈ 10 épisodes ≈ 3 % du run → les transitions avec hits ont été évincées dès ~40k steps ; SAC quasi on-policy.
- Critic : MSE ~1e-3 → RMSE ~0.03/step vs reward moyenne ~0.0005/step. Au début, les targets Q étaient dominés par le bonus d'entropie (α·H ≈ 0.1/step vs r ≈ 0.001).
- Reward hacking : r reste positif avec hit=0 — la gaussienne de visée (0.5×exp, ~18 obus/épisode) + bonus winner compensent la pénalité famine −1 → « spray sans toucher puis mourir » est un optimum local stable.
- γ=0.99 @30 Hz → horizon effectif ~3 s, la famine (30 s) est invisible pour Q sauf en fin de vie.

**Pistes proposées (aucune validée par Samuel à ce stade)** : buffer bien plus grand, α min clamp/lr pour ralentir l'effondrement, curriculum pour densifier les hits, rééquilibrer gaussienne vs famine, γ plus haut. Attention : φ gaussien et réserve d'obus déjà validés ([[ppo-reward-redesign-2026-08]]), annealing d'entropie déjà refusé.

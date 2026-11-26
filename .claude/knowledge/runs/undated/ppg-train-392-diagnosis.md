---
name: ppg-train-392-diagnosis
description: Run 392 (PPG, 1er long run avec coût de tir -0.03) — le tir s'effondre à 0 entre 120k et 160k steps, arène silencieuse, D retombe 350→250 ; cause = valeur espérée d'un obus négative au plancher aléatoire, pas PPG
metadata:
  type: project
---

Run 392 (PPG, 2026-11-01/02, 488k steps, 32 tanks, fire cost 0.03, fire/zoom proba 0.1,
entropie 0.001/0.001, n_pi 8, beta_clone 1) : hit 0.07-0.08 au plancher aléatoire, D monte
250→350 sur ce plancher (seuil 0.06), puis **fire 1.0 → 0.000 entre 120k et 160k**, descente
régulière à chaque rollout, pas de saut aux phases aux (toutes les 8 rollouts) → PPG hors de
cause. Ensuite r ≈ -0.0002, r_hit = r_rcv = 0 (personne ne tire, 32 copies de la même
politique), s ≈ 27-28 s = tout le monde meurt de famine (truncation sans pénalité), D retombe à
250, Hc 0.2 → -0.31 (σ se resserre sans signal), ev bruité. save_1 (108k) = meilleur checkpoint.

**Why:** au plancher hit ≈ 0.07, un obus rapporte ≈ 0.2×0.07 + aim ≈ 0.015 et coûte 0.03 ;
la famine coûte 0 et out_of_shells -0.1 : ne jamais tirer est l'optimum strict du reward actuel.
Le liquid monté à D=900 (run 345, 2026-09) n'avait ni coût de tir ni terminaison réserve vide,
et entropie 0.01 : la comparaison 392 vs 345 est confondue par le reward, pas par l'algo.

**How to apply:** juger un run à coût de tir d'abord sur `fire` ; si fire → 0 c'est l'économie
du tir, pas l'exploration. Options à proposer à Samuel : coût < gain espéré (≈ 0.01),
ou pénalité de famine symétrique à out_of_shells. Voir fire-cost-per-shell,
out-of-shells-termination, [[ppo-train-382-diagnosis]] (seuil curriculum < plancher aléatoire).

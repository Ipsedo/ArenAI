---
name: ppo-train-383-diagnosis
description: "Run 383 (382 + initial_sigma 0.05, entropie c 0.001, out_of_shells -0.1) — pire que 382 : PPO a rendu σ dépendant de l'état (poids tête σ 0.01 → 0.6, biais inchangé), σ moyen 0.05 → 0.24 (100k) → 0.02 (300k+), clip 0.35 / skip 0.5-0.7 dès le début, hit 0.065 → 0.025 sous le plancher aléatoire, miss 21 m plat"
metadata:
  node_type: memory
  type: project
  originSessionId: 19779ead-cf94-40e1-8f65-e2fde406d5ae
  modified: 2026-10-31T13:55:56.466Z
---

Run `outputs/train_383_ppo` (2026-10-31) = 382 + initial_sigma 0.05 + continuous_entropy 0.001
+ out_of_shells -0.1. Résultat pire : hit 0.065 → 0.025 (sous le plancher aléatoire 0.05 de 382),
kill ≈ 0.1, miss 21 m plat, D 250 → 300 → 250.

- **σ a été reparamétré par l'état, pas appris** : dans les 5 saves, le biais σ reste à 0.05 sur
  les 4 axes, mais les normes des lignes de poids de la tête σ passent de 0.01 à 0.44-0.67.
  Avec des features LayerNorm (variance 1), z varie de ±0.6 → log σ de ±1.2 (la SigmaOutput
  sigmoid en log sur [1e-4, 1] a une pente ≈ 2 nats de log σ par unité de logit à σ = 0.05).
  Hc (entropie moyenne par axe) : -1.57 (σ 0.05) → 0.0 à 100k (σ ≈ 0.24) → -2.5 à 300k (σ ≈ 0.02).
- **Le gradient σ écrase le gradient μ** : ∂logπ/∂σ ∝ ((a-μ)²/σ² - 1)/σ, 10× plus grand qu'à
  σ = 0.5. Conséquence dès la première fenêtre : clip 0.24-0.33 (382 : 0.09), kl 0.019-0.023,
  skip 0.49 puis 0.5-0.75 après 250k. Le budget KL part dans la remodélisation de σ(état).
- μ des axes de visée (lignes 2-3) a bougé (normes 0.26 / 0.45) mais pointe systématiquement
  à côté (miss 21 m ≈ 8° à 150 m) ; politique devenue déterministe et fausse → moins de hits
  chanceux que le spray de 382.
- fire 0.9-2.0/s au début (tir moins cher à -0.1, s ≈ 8 s = réserve vide en 8 s) puis 0.45/s,
  s ≈ 20 s. ev 0.5-0.8 (retours plus prévisibles, pas un meilleur critique).

**Why:** réduire σ₀ seul sur une tête σ dépendante de l'état donne à PPO un levier σ(état) qu'il
exploite immédiatement ; le lr 3e-4 et le target_kl 0.05 étaient calibrés pour σ = 0.5.

**How to apply:** avant de rebaisser σ, neutraliser le levier : σ indépendant de l'état (paramètre
libre par axe) et/ou plage SigmaOutput resserrée et/ou lr acteur abaissé. Lire Hc comme
σ ≈ exp(Hc − 1.42) (entropie moyenne par axe, hors troncature).

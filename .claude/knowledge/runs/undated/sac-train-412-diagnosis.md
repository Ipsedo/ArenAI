---
name: sac-train-412-diagnosis
description: "Run SAC 412 (2026-11-08, α₀ 0.01, sinon = 411) lu à 39.5k pas : Q ≈ 0.9 au lieu de 12, fire tient 0.4-0.5/s, Hc décolle du max (2.73 → 2.46-2.52, σ ≈ 0.65), hit 0.038 → 0.0445 encore sous le plancher aléatoire ; α ÷e par ~33k pas inchangé"
metadata:
  node_type: memory
  type: project
  originSessionId: 51b10db1-fb50-4f0f-81d1-b1b5b214142f
  modified: 2026-11-08T19:10:47.825Z
---

Run `outputs/train_412_sac`, seule différence avec [[sac-train-411-diagnosis]] : `initial_alpha` 0.01. Lu le 2026-11-08
à l'index 39.5k (≈1.27 M transitions, 90 min). Comparaison à index égal (33k) 411 → 412 :
Q 10.0 → 0.80 ; Hc 2.73 → 2.51 ; fire 0.30 → 0.46 tir/s ; hit 0.039 → 0.044 ; s 40 → 62 s ; kill 0.44 → 0.44.

- Q = toujours α·H/(1−γ) en forme (pic 0.93 à 20k puis suit α), mais 13× plus petit : le bonus de survie
  anti-tir (≈ γ·V) est passé de ≈10 à ≈0.9, et fire ne glisse plus vers 0.
- **Hc baisse de 2.72 à ≈2.5 à partir de 24k** : σ quitte le maximum d'entropie (0.9 → ≈0.65 par axe) ;
  premier signe que ∂Q/∂a pèse sur l'acteur. Cible 0.51 encore loin.
- hit monte lentement et régulièrement 0.029 → 0.0445 mais reste sous le plancher aléatoire 0.05 ; miss 22.5 m plat.
  Pas encore de visée mesurable. D figé à 250.
- kill/vie 1.0 → 0.44 dans les deux runs, indépendant d'α : pas un effet de l'entropie.
- α 0.01 → 0.0031 au même rythme qu'au 411 (lr 3e-5) ; ≈1.5e-4 attendu vers 100k. Risque à surveiller :
  Hc doit s'arrêter à 0.51 (Lagrangien), pas s'effondrer.
- Unités vérifiées : `s` = secondes de vie (frames à 30 Hz), `fire` = tirs par tank-seconde ; une vie couvre
  plusieurs réserves (respawn à réserve vide, 401b7874).

**Why:** confirmer que l'échelle entropie/reward était le verrou du 411.

**How to apply:** laisser tourner ; relire vers 100k pas (α ~1e-4) : attendre Hc → 0.51 et hit > 0.06.
`gradient_steps` > 1 reste la piste suivante à proposer (UTD 1/32).

**Suite (2026-11-08 soir) :** 412 arrêté par Samuel à 45k (Hc 2.43, hit 0.046, miss 22.8 m : rien de convergé, ≈445 pas/min).
Run 413 lancé 20:30 : réseau réduit (CNN 16/32/64, MLP 128/64), **batch 8192, gpu_batch 1024**, gradient_steps toujours 1
→ **43 pas/min, 10× plus lent** (GPU 100 %), même nombre d'updates par pas d'env, même décroissance d'α (÷e / 33k pas ≈ 13 h).
Diagnostic : le budget d'updates est le verrou (UTD 1/32, 33k updates en 75 min), pas la variance du gradient ;
levier = `gradient_steps` > 1 avec batch 256-512. Voir run 413 si écrit.

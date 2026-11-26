---
name: ppo-train-371-diagnosis
description: "Runs 370/371 ppo_impala (batch optim 512) — critique réparé (ev 0.75-0.85) mais acteur immobile en substance : σ figé à l'init (Hc ≈ 0.5, entropie 0.01), Hd = ln2, tir 0.85/s, KL budget brûlé sur du bruit (clip 0.4, skip 0.6), hit ≈ init 0.02"
metadata:
  type: project
---

Runs train_370 (actor_epochs 4) et train_371 (actor_epochs 8), 2026-10-28, rollout 300, batch 512/128, lr 3e-4, entropie c 0.01 / d 0.001.

- ev 0.72-0.88 dès 3.6k → le découpage batch d'optim a réparé le critique (ppo-full-batch-update).
- Acteur : kl ≈ 0.04 (moyenne incl. 1er pas à 0), skip 0.33 (370) / 0.62 (371), clip 0.35-0.44 → tout le budget KL consommé à chaque rollout ; Hc 0.42-0.53 = valeur d'init (σ₀ 0.5), Hd 0.61-0.69 ≈ ln2, fire 0.85/s plat, hit 0.015-0.025 ≈ niveau aléatoire initial, D=250. 370 et 371 identiques → epochs pas en cause.
- Récompense : r ≈ −0.0036/décision ⇒ ~1 mort −1 par vie ; s ≈ 23.5 s = 20 obus / 0.85 tir/s → quasi toutes les vies finissent en suicide out_of_shells (avant la famine à 30 s). ev élevé = surtout le compte à rebours des obus (observable), pas la valeur tactique.
- Signal de visée par rollout : ~408 tirs, ~8 hits (+0.2), aim ≈ 0.1·exp(−27 m / ~6 m) ≈ 0 pour un tir typique.
- σ 0.5 sur une visée en vitesse (180°/s plein) ≈ ±3°/frame, ~±9°/décision de bruit vs tolérance 2° → précision impossible ; l'entropie 0.01 (que j'avais recommandée) maintient σ à l'init faute de gradient de politique cohérent.

Voir [[ppo-train-369-diagnosis]], [[ppo-train-345-diagnosis]], ppo-beta-entropy-init.

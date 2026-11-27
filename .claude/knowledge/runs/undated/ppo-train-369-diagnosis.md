---
name: ppo-train-369-diagnosis
description: "Run 369 ppo_impala rollout 300 + entropy 0.01/0.001 — opposite regime to 368: frozen policy (kl 3e-4, clip ≈ 0), Hc rises, ev ≤ 0; no full-batch run (365-369) has a learning critic, unlike 364"
metadata:
  type: project
---

Run train_369_ppo-impala (2026-10-28, 100k decisions in 3h36, commit aad2ab41): rollout 300 (4 800 rows), lr 3e-4, epochs 4/8, full-batch (1 Adam step/epoch).

- clip 0.000-0.02, kl 0.0003-0.001 (target 0.05), skip 0 → actor nearly immobile; Hc 0.56 → 0.66 (entropy bonus 0.01 wins), Hd 0.69 → 0.45.
- ev oscillates between −0.7 and +0.05 → critic worse than a constant; hit flat 0.02, kill ~0.1, D=250.

Key comparison (final ev): 364 (true minibatches 128, 224 steps/rollout, lr 1e-4) **0.69**; 365 full-batch lr 1e-3 −0.37; 367 critic 16 epochs −0.06; 368 0.000; 369 < 0. At 100k decisions: 364 ≈ 25k critic steps, 369 ≈ 2.7k (≈10× fewer).

Retained cause: full-batch (ppo-full-batch-update) gives too few optimiser steps for the critic to learn; without a critic, advantages = MC noise → policy that does not move (369) or runs away (368). Proposed to Samuel: decouple optimisation batch size (e.g. 1024 rows, accumulated in blocks of 128) from memory size.

See [ppo-train-368-diagnosis](ppo-train-368-diagnosis.md), [ppo-train-364-diagnosis](ppo-train-364-diagnosis.md).

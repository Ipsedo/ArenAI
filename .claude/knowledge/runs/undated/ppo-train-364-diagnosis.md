---
name: ppo-train-364-diagnosis
description: "Run 364 ppo_impala (IMPALA CNN + SpatialSoftmax, minibatch 128) — ev 0.7 but σ collapses without hit gain, 2.2× slower than vanilla 362"
metadata:
  node_type: memory
  type: project
  originSessionId: 890e3bc3-16ec-4388-85c1-e9912581e991
  modified: 2026-10-27T18:39:38.066Z
---

Run train_364_ppo-impala (2026-10-27, 328k steps in 5.5 h, stopped) vs vanilla train_362_ppo at same steps:
- **ev ≈ 0.70 stable** (362: 0.30-0.38) → the IMPALA critic is clearly better.
- **Hc 0.55 → −0.25** from 150k (362: 0.36 at 330k); the fall coincides with a hit peak 0.031-0.035 (160-260k) then hit falls back to 0.021 at end → σ tightens without progress (premature commitment).
- hit/kill behind 362 (0.030 vs 0.038-0.048 at 300-360k); curriculum never triggered (D=250; 362 takes off ~580k).
- clip 0.03→0.07, kl ≤ 0.006 (≪ target 0.05): no optimisation instability.
- Throughput: ~1h50 / 108k steps vs ~50 min for 362 (stride-1 convs at full 128×256 resolution).
- minibatch 128 (vs 512 in vanilla) = 4× more gradient steps per rollout — suspect #1 for the σ drop.

See [ppo-train-345-diagnosis](ppo-train-345-diagnosis.md), ppo-vanilla-liquid-alignment, ppo-beta-entropy-init.

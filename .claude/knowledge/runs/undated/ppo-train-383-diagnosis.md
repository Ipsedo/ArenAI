---
name: ppo-train-383-diagnosis
description: "Run 383 (382 + initial_sigma 0.05, entropy c 0.001, out_of_shells -0.1) — worse than 382: PPO made σ state-dependent (σ head weights 0.01 → 0.6, bias unchanged), mean σ 0.05 → 0.24 (100k) → 0.02 (300k+), clip 0.35 / skip 0.5-0.7 from the start, hit 0.065 → 0.025 below the random floor, miss 21 m flat"
metadata:
  node_type: memory
  type: project
  originSessionId: 19779ead-cf94-40e1-8f65-e2fde406d5ae
  modified: 2026-10-31T13:55:56.466Z
---

Run `outputs/train_383_ppo` (2026-10-31) = 382 + initial_sigma 0.05 + continuous_entropy 0.001
+ out_of_shells -0.1. Worse result: hit 0.065 → 0.025 (below 382's random floor 0.05),
kill ≈ 0.1, miss 21 m flat, D 250 → 300 → 250.

- **σ was re-parameterised by state, not learned**: in the 5 saves, the σ bias stays at 0.05 on
  the 4 axes, but the row norms of the σ head weights go from 0.01 to 0.44-0.67.
  With LayerNorm features (variance 1), z varies ±0.6 → log σ ±1.2 (the SigmaOutput
  log-sigmoid on [1e-4, 1] has slope ≈ 2 nats of log σ per logit unit at σ = 0.05).
  Hc (mean entropy per axis): -1.57 (σ 0.05) → 0.0 at 100k (σ ≈ 0.24) → -2.5 at 300k (σ ≈ 0.02).
- **The σ gradient overwhelms the μ gradient**: ∂logπ/∂σ ∝ ((a-μ)²/σ² - 1)/σ, 10× larger than at
  σ = 0.5. Consequence from the first window: clip 0.24-0.33 (382: 0.09), kl 0.019-0.023,
  skip 0.49 then 0.5-0.75 after 250k. The KL budget goes into remodelling σ(state).
- μ of the aim axes (rows 2-3) moved (norms 0.26 / 0.45) but systematically points
  off (miss 21 m ≈ 8° at 150 m); policy became deterministic and wrong → fewer lucky
  hits than 382's spray.
- fire 0.9-2.0/s early (cheaper fire at -0.1, s ≈ 8 s = reserve empty in 8 s) then 0.45/s,
  s ≈ 20 s. ev 0.5-0.8 (more predictable returns, not a better critic).

**Why:** reducing σ₀ alone on a state-dependent σ head gives PPO a σ(state) lever that it
exploits immediately; lr 3e-4 and target_kl 0.05 were calibrated for σ = 0.5.

**How to apply:** before lowering σ again, neutralise the lever: state-independent σ (free
parameter per axis) and/or narrower SigmaOutput range and/or lower actor lr. Read Hc as
σ ≈ exp(Hc − 1.42) (mean entropy per axis, excluding truncation).

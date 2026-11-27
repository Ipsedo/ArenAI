---
name: ppo-train-326-diagnosis
description: "Vanilla Beta PPO (train_326) — no inference shell; policy stayed near-uniform (Hc≈0.65), in-game mode/argmax ≠ sampled policy"
metadata:
  node_type: memory
  type: project
  originSessionId: 988732ed-3afd-427f-b7ba-2661f0c7f9b5
  modified: 2026-09-17T19:26:00.279Z
---

Diagnosis 2026-09-17: vanilla PPO (train_326_ppo_beta_trunc, commit db5f02a1 « beta law
for vanilla actor ») neither aims nor fires in game despite « converged » training
metrics. Full analysis: **no shell** in the inference path — agents,
collector, trainer, save/load, factory and game loop are symmetric with ppo_liquid
(which works). Checkpoint healthy (weights moved, outputs input-dependent).

Cause: in game the agent plays deterministic (`beta_law_mode_action` + fire argmax); training
metrics measure the **sampled** policy. Final Hc of 326 ≈ 0.65
≈ log 2 = 0.693 (max entropy of a Beta on [−1,1]) ≈ entropy at init κ = 3: the Beta
stayed near-uniform on-policy for 2M steps (Hc had even dropped to 0.556 then risen). Hits
(~0.064/s, equal to liquid) came from random spray; the mode of a near-uniform Beta is
arbitrary → barrel down. Hd ≈ 0.36 → mean p(fire) ≈ 0.1 < 0.5 → argmax never fires.
Liquid (train_325): Hc 0.42 and falling → meaningful mode.

Reading rule: to predict in-game behaviour, look at **Hc and Hd**, not only
hit/fire/s. Hc near 0.69 = deterministic unrelated to sampled.

Structural pitfalls noted (not strict bugs): at κ→2, ∂logp/∂mode ∝ (κ−2) → mode head with no
gradient; the κ head (sigmoid log-scale [2.01, 2000]) saturates at the floor; the Lagrangian PID
α ≥ 0 never pushes entropy down → nothing forces κ up unless advantage does.
See ppo-beta-entropy-init, ppo-reward-redesign-2026-08.

`"hidden_sizes"` landmine **fixed 2026-09-17** (config.json migration to native JSON):
`AgentFactory` now reads `"actor_hidden_sizes"`, all fields are required (`at()`,
no silent defaults) and arrays are native JSON arrays — old config.json with string
values no longer load without migration.

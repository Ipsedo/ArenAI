---
name: ppo-train-344-diagnosis
description: "train_344 (ppo_liquid, fixed entropy factors 0.001, 256x128, 27.7M steps, 2026-10-11→16) — peak at 0.3-0.7M then slow regression, curriculum falls 550→350, Hc collapses without floor, Hd ≈ max"
metadata:
  node_type: memory
  type: project
  originSessionId: cf6947bf-0442-45af-815d-943234f1d08a
  modified: 2026-10-16T16:44:01.196Z
---

Run analysed 2026-10-16 (finished at 27.7M steps, 256 saves, save_k ≈ k×108k steps).
Config: ppo_liquid, truncated normal, fixed `continuous/discrete_entropy_factor = 0.001`
(Lagrangian PID removed in 00866e3d), aim reward, epochs 2, minibatch 2048, neuron_number 128.

- Hit peak (precision/episode) 0.081, kill 4.3 at 0.3-0.7M (D 250-300) → save_3..6 best.
  Worse than the peak of [ppo-train-339-diagnosis](ppo-train-339-diagnosis.md) (0.095).
- Curriculum rises to D=550 (5.1M), stays 11M, then **falls** 500 (16.2M) → 350 (26.8M).
  At D=350 late: kill 0.76 vs 3.9 at same D at 0.83M → real regression, not a D effect.
- Hc decreases steadily 0.51 → −1.9 (mean σ ≈ 0.04): 0.001 retains nothing. Hd rises
  0.62 → 0.686 (≈ ln2): fire/zoom stay ~coin-flip, the discrete head does not learn.
- clip 0.2-0.27, kl ~0.02, skip spikes up to 0.7; ev 0.3-0.4 (isolated −137 spike at 5M).
- fire ≈ 1.07 shells/s constant, s ≈ 33 constant; miss 50 → 29 (partly D effect).

Reading: fixed factors = worse than the PID floor (uncontrolled Hc). See ppo-beta-entropy-init.

Follow-up validated by Samuel (2026-10-16), **ppo_liquid only**: state-independent σ
(parameter `sigma_logit` [nb_continuous] → `SigmaOutput` log-sigmoid [SIGMA_MIN, SIGMA_MAX],
init = `initial_sigma` 0.5, orthogonal μ head ×0.01 unchanged) CleanRL/Andrychowicz 2021 style,
and fixed entropy factors 0.01 (Atari/Procgen standard, advantages normalised per rollout).
Old ppo_liquid checkpoints (head `sigma.0.*`) no longer reload. Options set aside for now:
back to PID, per-dim σ floor (DreamerV3), pool of past opponents.

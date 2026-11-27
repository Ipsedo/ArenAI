---
name: ppo-train-373-diagnosis
description: run 373 impala = 372 + aim A3 + out_of_shells −0.3 — miss still 28-30 m while σ collapses (Hc → −0.3): the aim mean learns nothing
metadata:
  type: project
---
Run 373 (2026-10-29, ~194k decisions/tank, binary rebuilt after A3+B2): r_aim ×2 (0.0012/decision, ≈ 0.014 per shot, as predicted), r ≈ −0.0003.
- miss 27 → 30 m, hit 0.006-0.02 (random), fire 0.85/s, s ≈ 23 s (still shell dump), D=250, ev 0.7-0.88.
- Hc 0.38 → −0.3 (σ ≈ 0.5 → 0.18) without miss decrease: 3rd time impala's state-dependent σ collapses without competence (364, 368, 373).
- Hd 0.69 → 0.55-0.6, no effect.
Reading: the reward gap between a good and a bad shot (≈ 0.03) remains ≈ 10× smaller than the random ±0.3 events (hit received, death); normalised advantage ≈ 0.1 σ, diluted over ~20 shell-flight decisions. Options: hit_reward_scale 0.1 → ~1, control run ppo/ppo_liquid (independent σ) at identical config. See [ppo-train-372-diagnosis](ppo-train-372-diagnosis.md).

2026-10-29: verified in code — ppo, ppo_liquid, ppo_liquid_impala all have a state-dependent σ (Linear + SigmaOutput). Samuel switched ppo_impala alone to a free `log_sigma` parameter per action (clamped [1e-4, 1]) to test the collapse hypothesis.

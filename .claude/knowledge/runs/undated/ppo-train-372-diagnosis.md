---
name: ppo-train-372-diagnosis
description: run 372 impala (entropy c 0.001 / d 0, target_kl 0.02) — actor calmed, σ decreases without hit gain, fire still random
metadata:
  type: project
---
Run 372 (2026-10-29, early ~39k decisions/tank ≈ 130 rollouts) = 371 + continuous_entropy 0.001, discrete 0, target_kl 0.02 (Samuel's commits cd60f19b, 588b35c6).
- clip 0.21-0.28 (vs 0.4 in 371), kl ≈ 0.018 ≈ target, skip 0.70 (≈ 2.4 useful epochs of 8).
- Hc 0.57 → 0.20-0.30: σ decreases this time, but hit 0.047 → 0.012-0.02 (random level), miss 27 m unchanged.
- Hd ≈ ln 2: fire/zoom still coin-flip; fire ~0.85/s, s ≈ 24-25 s = still out_of_shells suicide.
- ev 0.70-0.86.
Reading: hyperparams fixed the actor's agitation but not the lack of signal (rewarded aim ≈ 0, suicide −1 dominant); risk of σ collapse without competence as in 368. See [ppo-train-371-diagnosis](ppo-train-371-diagnosis.md).

Follow-up decided 2026-10-29 (uncommitted at time of writing): aim A3 = 0.5·exp(−m/tol) + 0.5·exp(−m/tol_large), tol_large = 10 m + 6°·d; out_of_shells stays a termination but costs −0.3 (B2), flipped/combat stay −1. Options refused/not retained: per-shot cost (B1), dense shaping (A4).

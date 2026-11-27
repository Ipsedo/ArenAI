---
name: ppg-train-395-diagnosis
description: "Run 395 (PPG, Cauchy aim, no fire cost) — hit 0.077 → 0.03 locked on D 250→300 while r rises; cause = tipping-point economy (pair exchange ≤ 0), not Cauchy nor PPG hyper-parameters; follow-up = hit +0.3 / received -0.2"
metadata:
  node_type: memory
  type: project
  originSessionId: 22b3c1ad-5726-43a8-bbb0-26231322ea86
  modified: 2026-11-02T21:07:38.502Z
---

Run 395 PPG (2026-11-02, binary ef514d06 = Cauchy aim reward 1/(1+(miss/tol)²), tol 2.5 m + 3°·d,
no fire cost, out_of_shells -0.5, γ 0.99 / λ 0.95). Same config as 393/394.

- At D=250: hit 0.055 → 0.078, kill 0 → 1.1. Iteration 59 (53k): D 250 → 300, hit 0.077 → 0.05
  in 4 iterations, brief recovery to 0.06 (66k), then 0.03 and kill = 0 at 89k. 394 absorbed the same
  step (0.054 → 0.07, then D=350); hit (16-episode window) noisy ±0.015 at constant D.
- r decomposition: r_aim, r_hit **and** r_rcv drop together, deaths less frequent, s 15 → 20 s,
  fire 0.8 → 0.45, miss constant 20 m, **total r rises**. The agent optimises well: the population
  converges to the calm arena, through cadence/precision (392 did it through fire → 0).
- **Cauchy is not the cause** (first reading wrong, corrected by Samuel): ppo_liquid
  on the other PC runs with Cauchy and reaches D=400. Cauchy keeps a slope far from the target
  (useful to learn aim) and pays a bit more for missing shells: it makes the
  problem comfortable, it does not create it.
- Retained cause = tipping-point economy: a hit = +0.2 (+aim ≤ 0.1) for the shooter, -0.3
  for the target → pair exchange ≤ 0, only the kill (+2/-1, 5 hits) is positive; a life = 10
  shells then out_of_shells -0.5 (s ≈ 17 s ≈ 10 shells at 0.6/s), which firing slowly pushes out of
  the γ horizon (10 s). Engaging pays only if p(hit) stays above a threshold; 395 crossed it
  at the D=300 step, 394 stayed just above, the liquid (memory) holds.
- PPG hyper-parameters ruled out: Hc 0.21 → 0.10 = σ 0.30 → 0.27; clip 0.2-0.25, kl
  0.013-0.016 < 0.05, skip 0, identical to 393/394. Note: aux distils on the last
  rollout only, not the accumulated n_pi.
- **Decision 2026-11-02 (Samuel)**: positive exchange for the pair, hit +0.3 / received -0.2
  (jolt_enemy_tank.cpp, tests aligned: death + received = -1.2).

**Why:** Samuel overruled the Cauchy reading with the liquid counter-example; do not conclude
from a comparison to one run per variant.
**How to apply:** judge on hit/fire + r_rcv/death/s decomposition, not on r. See
fire-cost-per-shell, [ppg-train-392-diagnosis](ppg-train-392-diagnosis.md), out-of-shells-termination,
aim-fire-levers-audit-2026-11.

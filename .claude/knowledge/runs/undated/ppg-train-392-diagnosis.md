---
name: ppg-train-392-diagnosis
description: Run 392 (PPG, 1st long run with fire cost -0.03) — fire collapses to 0 between 120k and 160k steps, silent arena, D falls 350→250; cause = expected value of a shell negative at the random floor, not PPG
metadata:
  type: project
---

Run 392 (PPG, 2026-11-01/02, 488k steps, 32 tanks, fire cost 0.03, fire/zoom proba 0.1,
entropy 0.001/0.001, n_pi 8, beta_clone 1): hit 0.07-0.08 at the random floor, D rises
250→350 on that floor (threshold 0.06), then **fire 1.0 → 0.000 between 120k and 160k**, steady
descent at every rollout, no jump at aux phases (every 8 rollouts) → PPG ruled out. Afterwards
r ≈ -0.0002, r_hit = r_rcv = 0 (nobody fires, 32 copies of the same policy), s ≈ 27-28 s = everyone
dies of starvation (truncation without penalty), D falls back to 250, Hc 0.2 → -0.31 (σ tightens
without signal), ev noisy. save_1 (108k) = best checkpoint.

**Why:** at the floor hit ≈ 0.07, a shell yields ≈ 0.2×0.07 + aim ≈ 0.015 and costs 0.03;
starvation costs 0 and out_of_shells -0.1: never firing is the strict optimum of the current reward.
The liquid that reached D=900 (run 345, 2026-09) had neither fire cost nor empty-reserve termination,
and entropy 0.01: the 392 vs 345 comparison is confounded by the reward, not by the algorithm.

**How to apply:** judge a fire-cost run first on `fire`; if fire → 0 it is the fire economy,
not exploration. Options to propose to Samuel: cost < expected gain (≈ 0.01),
or a starvation penalty symmetric to out_of_shells. See fire-cost-per-shell,
out-of-shells-termination, [ppo-train-382-diagnosis](ppo-train-382-diagnosis.md) (curriculum threshold < random floor).

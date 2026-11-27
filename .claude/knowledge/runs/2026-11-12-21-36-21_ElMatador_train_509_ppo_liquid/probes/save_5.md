# Probes 509 save_3 (81 k, D 400) and save_5 (135 k, D 450) vs 508 save_5

`build/` binary of 12/11 21:06, commit 1e171bbf = run code, no inference diff. 4 episodes per side, stochastic. 508 save_5: same index, same D (12/11 probe, binary of 508's code).

| run | side | p_arm | ratio | hit/sh | h<5° | shots<5° | empty_s | %disarmed | shots/s | turret<5° | σ cx/cy | \|μ\|cx |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 509 s1 | 550 | 0.72 | 1.02 | 6.7 % | 13.7 % | 21 % | 2.5 | 68 % | 1.18 | 15.6 % | 0.13/0.14 | 0.085 |
| 509 s3 | 550 | 0.77 | 1.02 | 8.1 % | 17.8 % | 25 % | 2.4 | 64 % | 1.39 | 18.1 % | 0.09/0.13 | 0.067 |
| **509 s5** | 550 | 0.70 | 1.12 | **12.2 %** | **32.0 %** | 26 % | 2.5 | 58 % | 1.42 | 18.8 % | 0.15/0.19 | 0.12 |
| 508 s5 | 550 | 0.31 | 1.9 | 9.9 % | 21.3 % | n/a | 4.4 | n/a | 1.26 | 17.4 % | 0.40/0.36 | 0.26 |
| 509 s1 | 1000 | 0.72 | 1.02 | 3.3 % | 7.9 % | 19 % | 2.4 | 71 % | 1.16 | 14.2 % | 0.13/0.14 | 0.076 |
| 509 s3 | 1000 | 0.76 | 1.03 | 4.3 % | 11.9 % | 19 % | 2.3 | 70 % | 1.22 | 12.9 % | 0.09/0.13 | 0.058 |
| **509 s5** | 1000 | 0.66 | 1.19 | **6.5 %** | **23.9 %** | 20 % | 2.4 | 74 % | 0.89 | 10.9 % | 0.22/0.27 | 0.12 |
| 508 s5 | 1000 | 0.26 | 2.2 | 4.3 % | 12.0 % | n/a | 4.4 | n/a | 0.93 | 13.1 % | 0.43/0.39 | 0.26 |

CSV at same indices (509 / 508): D milestones identical (350 at 54 k, 400 at 81 k, 450 at 135 k, 500 at 149 k); hit 15.0 / 12.5 %, kill 22.8 / 18.3 at 135 k; skip 0.28 → 0.46 (108 k) → 0.40 vs 0; kl 0.028-0.034 vs 0.010-0.014; alpha_canon 0.004 → 0.019 vs 0.002 → 0.0006; Hc −0.55 (54 k) → −0.20.

## Reading

- **Aligned-shot precision**: h<5° goes from 13.7 to 32.0 % (550) and from 7.9 to 23.9 % (1000) between save_1 and save_5. That is 1.5 to 2× 508 save_5. hit/shell +23 % / +51 % vs 508. New gain since save_1: for a given turret alignment, the shell hits much more often. Consistent with a canon (pitch included) credited directly on Φ.
- **Turret**: save_1's lead did not hold. 18.8 / 10.9 % vs 508 17.4 / 13.1 %: tie at 550, behind at 1000.
- **Trigger**: just starting to sort (ratio 1.02 → 1.12-1.19), but far from 508 at same index (1.9-2.2). Stays in dump: p_arm 0.66-0.70 vs 0.26-0.31, reserve empty at 2.4-2.5 s vs 4.4 s, 58-74 % of time disarmed.
- **σ canon**: trough 0.09 at save_3, back to 0.15-0.27 at save_5 thanks to canon alphas (×4). Entropy control caught the tightening. No μ saturation.

## Established / hypothesis

- Established (two sides, three saves, consistent with CSV): factorised credit makes the aligned shot much more precise. CSV lead comes from there, not turret nor trigger.
- Established: 509's trigger lags 508's at equal index.
- **Hypothesis, unverified**: the 40 % skip brakes the trigger. The joint importance ratio (used by KL and skip) is dominated by canon, which moves a lot under the new signal. Skipped minibatches are skipped for all heads, so the trigger loses ~40 % of its updates. CSV lacks per-action-group KL, preventing direct confirmation. Single seed: trigger lag may also be noise (507 and 508 had variable dump-exit timing).

## Next

- Verify the skip hypothesis: measure KL per group (canon vs rest) on a rollout, without touching training code if possible.
- save_9 (243 k) vs 508 save_9 (ratio 6.8 / 11.6, reserve 6.5 / 13.8 s, hit/shell 15.2 / 7.9 %): does the trigger catch up, and does precision hold?

## KL per action group (save_5, 1 episode side 1000, 6 rollouts)

Tool: `arenai_agent_sigma_gradients --kl_by_group` (uncommitted at measurement time). Each rollout collected with save_5, then actor update replayed with the save's Adam state (`actor_optim.pt`) over the 4 full epochs, without target_kl stop. Truncated KL read before each step, joint (what skip reads) and per group: movement (dir_x, dir_y), canon (canon_x, canon_y), discrete (fire, zoom). Alphas stay at the save's values, critic not updated, policy reloaded from save after each rollout.

| epoch | joint | movement | canon | discrete | share of steps joint > 0.075 |
|---|---|---|---|---|---|
| 0 | 0.021 | 0.004 | 0.015 | 0.0002 | 14 % |
| 1 | 0.057 | 0.009 | 0.046 | 0.0003 | 33 % |
| 2 | 0.054 | 0.007 | 0.041 | 0.0003 | 21 % |
| 3 | 0.067 | 0.008 | 0.057 | 0.0007 | 42 % |

- Threshold (1.5 × 0.05) crossed at step 10-13 of 44-64 planned in 5 rollouts out of 6. At the stopping step, canon is 77 % of joint KL (median), movement 12 %, discrete 1 %.
- Movement + discrete cross the threshold in none of 304 steps. Without canon, no update would be cut.
- **Established**: 509's skip is caused by canon. Each cut also deprives trigger and movement of their remaining epochs.
- **Not established**: that this is the cause of the trigger's lag. Its per-step KL is tiny (0.0002-0.0018) even without cut: it moves little per update anyway.
- **Gap with CSV**: replay cuts earlier (~75 % of steps skipped) than the real run (skip 0.40-0.46). Possible causes: frozen alphas, a single episode side 1000 vs curriculum distribution, end-of-episode rollout sizes. The decomposition's direction does not depend on it, its magnitude does.

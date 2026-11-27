# CSV `ev` computed per GPU micro-batch was an artefact — fixed to optimiser batch

Runs: 515, 516, 517, 518

2026-11-14: CSV `ev` was computed per GPU micro-batch (negative artefact at gpu_batch 256/128 in 516-518); fixed to cover the whole optimisation batch in ppo/ppo_liquid/ppg (uncommitted).

`backward_critic` now returns (returns, values) detached, and `explained_variance()` (update_utils) computes `ev` over the concatenation of the blocks of one optimisation batch. Identical to the old computation when gpu_batch_size = batch_size (≤ 515: 512/512). Test: tests_update_utils.cpp.

**Why:** 516-518 (critic_gpu_batch_size 256 ⟹ 8 sequences of 30 steps per computation) displayed ev −1 to −14 while critic loss was lower than in 515.

**Rule:** `ev` of 516, 517 and 518 (launched before the fix) is not interpretable; compare the `v` loss instead. Runs launched after the release rebuild: ev comparable to ≤ 515.

# PPO liquid train 439 — 5 changes at once, D stuck at 250 (2026-10-26)

Host: ElMatador (memory notes); raw data no longer on disk.
Source note date: 2026-10-26. Run 439 ppo_liquid (commit `d01d5432`), 1.75 M steps, D = 250 throughout. Changes: no entropy bonus, γ 0.994/λ 0.99, p_init fire 0.01 / zoom 0.05, new critic head skip 256.

- **Curriculum never passed**: hits/shots max ~4.9 % (1 window > 6 % of 1953), same ceiling as 435 (9 windows) and 437 (17). High threshold 0.06 unreachable for this lineage.
- **Peak 375-525 k** (kill 0.24-0.36/episode, hit 4.5-4.9 %, fire 0.7-0.78/s) then **collapse at 525-550 k**: kill ÷3-4 (0.05), hit 3 %, fire 0.55-0.6/s. Mechanism = 437: the more it fires, the more it dies with empty reserve (s ≈ 29 s ≈ 20 shells / 0.7 per s), r most negative at the peak ⟹ learns to fire less ⟹ the curriculum metric collapses.
- **Hc 0.54 → −0.35** linear (σ < 0.25), same slope as 435 with bonus ⟹ the bonus is not the factor; but kl 0.005 → 0.043 (target 0.05), clip 0.31, skip 12 % at the end = slowed 433 regime.
- **Hd 0.13 → 0.27** (vs 0.55-0.6 on 435/437): discrete heads born near-deterministic (H(0.01)+H(0.05))/2 = 0.127; fire climbs back to 0.65/s in 75 k, zoom probably never explored.
- **Critic**: v (MSE on normalised returns) starts 4.9 (435: 0.23, 437: 1.5), 0.3-0.6 after (2× 435); ev ≤ 0 on 898/1947 lines, plateaus at −2/−3/−7.5 = a single outlier lingering in the 4096 window. New `to_value` head (skip 256 + hidden layer) + longer γ/λ.
- Why: the commit bundles 5 changes; none attacks the cause (net negative reward with empty-reserve death + hit ceiling 5 %), so nothing is demonstrated on the entropy bonus or γ. Do not conclude « no entropy = failure » from 439; compare 439 to 437 (same empty-reserve death) rather than 435.
- Related: runs 437, 433, 416; cooldown refused (see docs/decisions.md: 2026-10-25).

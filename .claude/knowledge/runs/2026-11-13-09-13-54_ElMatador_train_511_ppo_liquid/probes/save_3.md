# Probe 511 save_3 (81 k, D 400) vs 510 save_3

`build/` binary of 13/11 09:13, commit deca02d4 (no respawn at empty reserve).
- 511: no inference diff.
- 510 save_3: re-probed with this binary in `--force`, only diff = respawn removed. Both policies compared under the same rule.
- Old 510 save_3 probe (binary 1e171bbf, with respawn) archived in `train_510_ppo_liquid/probes/binaire_1e171bbf_avec_respawn/`.

4 episodes per side, stochastic, 32 tanks.

| run | side | p_arm | ratio | <5°-only ratio | hit/sh | h<5° | empty_s p50 (p90) | %empty | %disarmed | shots/s | turret<5° | σ cx/cy | \|μ\|cx |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 510 s3 | 550 | 0.80 | 1.05 | 1.03 | 15.0 % | 24.2 % | 2.3 (5.9) | 98 % | 89 % | 0.089 | 35.0 % | 0.18/0.23 | 0.11 |
| **511 s3** | 550 | 0.75 | **1.13** | **1.10** | **21.4 %** | **35.6 %** | 2.4 (7.6) | 94 % | 84 % | 0.116 | 36.4 % | **0.29/0.35** | 0.21 |
| 510 s3 | 1000 | 0.78 | 1.03 | 1.02 | 5.0 % | 10.2 % | 2.2 (2.6) | 99 % | 94 % | 0.066 | 25.9 % | 0.19/0.23 | 0.11 |
| **511 s3** | 1000 | 0.72 | **1.22** | **1.14** | 7.0 % | **20.4 %** | 2.3 (3.0) | 99 % | 94 % | 0.070 | 29.3 % | **0.31/0.36** | 0.20 |

Aligned & <40 m counts: 511 122/29, 510 43/12. Too small, hence the « <5°-only » ratio (n 846/334 vs 497/238), which goes the same way.

CSV at 79 k: 511 hit 23 % (510: 13 %), kill 7.8 (23), skip_canon 0.39 (0.17), kl 0.023 (0.041). 511's kills not comparable: early deaths, then zombies.

## Reading

- **Reserve**: still not longer. Median 2.3-2.4 s vs 2.2-2.3 s, and 94-99 % of tank-episodes empty. Only the tail lengthens a bit at 550 (p90 7.6 vs 5.9 s, 94 % empty vs 98 %).
- **Trigger**: first onset of sorting in 511 (ratio 1.10-1.22 vs 1.02-1.05), but stays high everywhere (p_arm 0.72-0.75, > 0.5 deterministic). Nascent selectivity does not yet translate into economy.
- **Precision**: h<5° 36/20 % vs 24/10 %, hit/shell 21 % vs 15 % at 550; thinner gap at 1000. CSV goes the same way (hit 23 % vs 13 %).
- **Canon**: σ trough seen at save_1 (0.09/0.10) is gone, σ at 0.29-0.36, above 510. `skip_canon` stays high (0.33-0.39).

## Established / noise

- Established (two saves, two sides): without respawn, the 1st empty reserve does not recede at 27 k nor 81 k.
- Probable (two sides, one seed): 511 fires better than 510 at same index (h<5° ×1.5-2) and its trigger starts sorting somewhat earlier.
- Possible noise: turret gaps (36/29 % vs 35/26 %).

## Next

- save_5 (135 k, around 11:20): in 510, this is where the trigger took off (ratio 1.9-2.1, reserve 3.0-3.4 s with respawn). Question: does 511 drop off more?

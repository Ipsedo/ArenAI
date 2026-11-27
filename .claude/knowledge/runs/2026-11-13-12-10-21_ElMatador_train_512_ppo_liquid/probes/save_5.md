# Probes 512 save_3 and save_5 vs 510 and 511

Run 512 = 511 + delayed respawn in place: 10 s disarmed and vulnerable, then `truncated` + V cut (commit 68b9697d), identical config.

`build/` binary of 12:05, compiled from the working tree which already contained the 68b9697d patch identically (commit at 12:10). Script dates it to the previous commit, hence a false diff: launched in `--force`.

4 episodes per side, stochastic, 32 tanks.

**Probe rules:**
- 512 is probed with delayed respawn; 510 and 511 were probed without respawn (binary deca02d4).
- Comparable: 1st empty reserve and trigger on armed decisions.
- Not comparable: %disarmed and shots/s, which 512's reloads mechanically change. Hit/shell and h<5° are less so too, since 512 fires throughout the 180 s, in successive waves.

| run | save | side | p_arm | ratio | <5°-only ratio | empty_s p10/p50/p90 | %empty | hit/sh | h<5° | turret<5° | σ cx/cy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 510 | 3 | 550 | 0.80 | 1.05 | 1.03 | 2.0/2.3/5.9 | 98 % | 15.0 % | 24.2 % | 35.0 % | 0.18/0.23 |
| 511 | 3 | 550 | 0.75 | 1.13 | 1.10 | 2.0/2.4/7.6 | 94 % | 21.4 % | 35.6 % | 36.4 % | 0.29/0.35 |
| 512 | 3 | 550 | 0.57 | 1.17 | 1.17 | 2.3/3.0/6.7 | 95 % | 13.5 % | 23.0 % | 30.1 % | 0.29/0.29 |
| 510 | 5 | 550 | 0.58 | 1.58 | 1.46 | 2.2/3.1/7.8 | 90 % | 20.7 % | 37.4 % | 44.6 % | 0.25/0.26 |
| 511 | 5 | 550 | 0.82 | 1.06 | 1.08 | 1.9/2.3/6.7 | 94 % | 19.0 % | 34.3 % | 44.2 % | 0.38/0.34 |
| **512** | 5 | 550 | **0.42** | **1.65** | **1.52** | 2.3/**3.7**/**9.8** | 92 % | 13.3 % | 21.3 % | 34.3 % | 0.39/0.36 |
| 510 | 5 | 1000 | 0.47 | 1.52 | 1.41 | 2.2/3.0/5.2 | 98 % | 12.2 % | 24.2 % | 32.7 % | 0.39/0.35 |
| 511 | 5 | 1000 | 0.80 | 1.09 | 1.02 | 1.9/2.2/4.0 | 99 % | 12.8 % | 29.0 % | 29.9 % | 0.49/0.42 |
| **512** | 5 | 1000 | **0.35** | **1.49** | **1.57** | 2.1/**3.2**/**6.6** | 97 % | 8.4 % | 14.9 % | 31.1 % | 0.47/0.43 |

512 counts (<5° only): 8 660/9 358 at save_5, vs 967/791 for 510. Reloads multiply armed decisions, so 512's ratios are much less noisy.

CSV at ~108 k: 512 `s` 43 s, fire 0.69/s, hit 17 %, kills 17 (510: 49 / 1.47 / 14.5 % / 23; 511: 5 / 0 / 25 % / 8).

## Reading

- **511's dump equilibrium: broken.** 512's trigger takes off between save_3 and save_5 like 510's (ratio 1.17 → 1.65/1.49). It is even lower (p_arm 0.42/0.35 vs 0.58/0.47), whereas 511's stayed flat at 0.8.
- **Reserve:** longest of the three at save_5 (p50 3.7/3.2 s vs 3.1/3.0 in 510 and 2.3/2.2 in 511; p90 9.8/6.6 s). But 92-97 % of tanks still empty their first reserve in ~3-4 s: dump attenuates without having disappeared.
- **Precision:** behind (h<5° 21/15 % vs 37/24 % in 510, turret<5° 34/31 vs 45/33). A serious reservation, but partly confounded with probe rules (continuous fire waves in 512, single shoot-out at start in 510/511). 512's CSV hit (17 %) is moreover above 510's (14.5 %) at same index.

## Established / noise

- Established (two sides, large counts): delayed respawn restores the trigger learning that 511 had lost, at 510's pace.
- Probable: 1st reserve holds a bit longer than in 510 (+0.2 to +0.6 s median, p90 +2 s). One seed only.
- Undecided: is 512's lower precision due to policy or probe rules? To settle, probe 512 save_5 and 510 save_5 under the same rule.

## Next

- save_9 (243 k) vs 510 save_9 (ratio 6.4/10.8, reserve 9.1 s with respawn): this is where 510 really left the dump.

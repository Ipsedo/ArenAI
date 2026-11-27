# Probe 512 save_9 (243 k, D 650) vs 510 save_9, same rule

`build/` binary of 12:05: code of 68b9697d (respawn delayed 10 s in place). Both checkpoints launched in `--force`.
- 512: false diff due to binary dating, cf. save_5.
- 510: only diff = respawn rule. Both policies played under the same rule.
- Old 510 save_9 probe (binary 1e171bbf, immediate respawn) archived in `train_510_ppo_liquid/probes/binaire_1e171bbf_avec_respawn/`.

4 episodes per side, stochastic, 32 tanks.

| run | side | p_arm | ratio | <5°-only ratio | empty_s p10/p50/p90 | %empty | %disarmed | hit/sh | h<5° | enrich | turret<5° | σ cx/cy | p_zoom |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 510 | 550 | 0.10 | **5.9** | 2.50 | 2.6/**9.5**/32.1 | 81 % | 36 % | **17.0 %** | **32.8 %** | 1.83 | 24.2 % | 0.55/0.49 | 0.36 |
| 512 | 550 | 0.17 | 4.0 | 2.41 | 2.7/7.1/18.3 | 87 % | 47 % | 15.0 % | 24.1 % | 1.53 | **32.8 %** | 0.43/0.34 | 0.67 |
| 510 | 1000 | 0.05 | **10.9** | 2.10 | 4.2/**14.6**/59.9 | 91 % | 30 % | 8.8 % | **24.5 %** | 1.68 | 19.1 % | 0.60/0.53 | 0.30 |
| 512 | 1000 | 0.10 | 4.3 | 2.72 | 3.4/10.5/29.7 | 91 % | 44 % | **11.6 %** | 21.3 % | **1.95** | **24.6 %** | 0.53/0.41 | 0.64 |

Aligned & <40 m counts: 512 545/322, 510 593/227; <5° only: 16-20 k for both.

CSV at 230-256 k: D 650 for both (same curriculum pace). 512: fire 0.55/s, hit 17 %, kills 15.9; 510: 0.81/s, 19.4 %, 22.5.

## Reading

- **511's problem is fixed.** Between save_5 and save_9, 512's 1st reserve goes from 3.2-3.7 s to 7.1-10.5 s, and the trigger becomes selective (ratio 4, p_arm 0.10-0.17). 511 stayed flat at 2.2 s.
- **But 512 does no better than 510.** Under the same rule, 510's policy holds longer (9.5/14.6 s), sorts harder (ratio 5.9/10.9 vs 4.0/4.3) and fires truer when aligned (h<5° 33/25 vs 24/21).
- **512's only advantage: turret** (turret<5° 33/25 vs 24/19), with equal or higher enrichment at 1000.
- **Orders of magnitude of gaps:** on reserve, they stay within known seed noise (×2 between identical configs, cf. memory). On ratio, they are consistent on both sides.

## Conclusion

- **Established:** the 10 s delayed respawn, cut at respawn, breaks 511's dump equilibrium. Trigger and shell economy relearn at 510's pace (same D milestone, same reserve profile).
- **Not established:** the hoped economy gain vs 510. At 243 k and under the same rule, 510 is equal or better on reserve and selectivity, one seed per run.
- **For the game, which has no respawn:** both policies still empty their first reserve in more than 80 % of tank-episodes. Neither has learned to finish a fight with shells left.

## Next

- Let 512 run and compare with 510 at D = 1000 (reached by 510 at 428 k), on save_28 vs 510 save_28 under the same rule.
- If 512 stays behind at equal D, the simplest training rule becomes 510's again (immediate respawn, cut at empty): it gives at least as well under the three probe rules tested.

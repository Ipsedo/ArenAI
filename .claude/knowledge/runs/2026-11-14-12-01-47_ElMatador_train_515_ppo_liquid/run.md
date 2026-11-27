# train_515_ppo_liquid

| | |
|---|---|
| machine | ElMatador |
| user | samuel |
| launch | 2026-11-14 12:01:47 |
| commit at launch | `d1bb4020` — * aim potential: ratio = miss / AIM_TARGET_RADIUS, full score only through the center * agent turret/canon action mapped through a·|a| (fine control near 0), probe oracle uses the inverse * shell economy : add 3 shells per hit |
| commits ≤ 90 min after | none |
| diff at launch | unknown (identity recorded after the run ended) |
| raw data | `outputs/train_515_ppo_liquid` on ElMatador |

## Established facts

- 2026-11-14: Run 515 = config 514 except gamma 0.995 → 0.998 (`liquid_ppo_hyperparams.h`, not committed). Effective commit `d1bb4020`, probe in `--force` (same binary as 514). Runs 510/513/514/515 all pass D=450 at the same step (114 173), and 510/513/514 also D=500 and D=1000 at the same step: curriculum does not discriminate runs, compare at same index. Stopped by Samuel at 380 k (save_14) when preparing run 516.
- 2026-11-14: save_4 (108 k): ratio 1.09/1.10, p_fire 0.82 → 0.75 flat, reserve empty 2.3 s ⟹ not yet taken off (513/514 save_5: 1.25-1.33). shots/s 1.32/0.98 vs 1.5-1.8 but %disarmed 65/75 vs 51-67 ⟹ lower CSV `fire` = more time empty, not selectivity. Turret <5° 15.2/10.4 % vs ~20/13-19 % (only gap, to confirm). Analysis `outputs/train_515_ppo_liquid/probes/save_4.md`. Do not read the CSV `fire` drop as reduced dump.
- 2026-11-14: save_9 (243 k, 15:15, `probes/save_9.md`): ratio 3.2/4.0 ≡ 513/514 (510: 6/11), but lower trigger (p_arm 0.18/0.12), reserve 7.6/13.9 s, disarmed 17/26 % (best of 4), hit/shell 17.3/12.5 % (best of 4), turret 18/14 ≡ 514. CSV 240-300 k on 510's trajectory (hit 16 → 20 %, fire 0.7). Less dump than 514 without more selectivity; one seed, not yet attributable to gamma.

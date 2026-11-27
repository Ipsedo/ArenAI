# Probe train_515_ppo_liquid / save_4

- Index 108 k, D = 400 at the time of the save (CSV passes D=450 at 114 k, at the same step as 510/513/514).
- Run: 514's config with a single change, gamma 0.995 → 0.998 (uncommitted modification in `liquid_ppo_hyperparams.h`). Effective commit is d1bb4020.
- Binary: `build/arenai_agent/arenai_agent_probe`, compiled 14/11 at 01:13 from the working tree committed 6 min later as d1bb4020. Script dates it 3cd9e210 by its mtime. Same situation as 514's probe: gamma serves only training and economy is read from config.json. Launched in `--force`.
- Sides 550 and 1000, 4 episodes, stochastic, economy 10/3/30.

## Summary

No probe exists at 108 k for the other runs, so 515 save_4 is bracketed by save_3 (81 k) and save_5 (135 k) of 513 and 514.

| run | side | p_arm | ratio | n al&near | hit/sh | enrich | empty_s | %disarmed | shots/s | turret<5° | σ cx/cy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 514 s3 | 550 | 0.81 | 1.03 | 494 | 7.9 % | 1.08 | 2.3 | 58 % | 1.65 | 20.2 % | 0.20/0.21 |
| 513 s3 | 550 | 0.81 | 1.05 | 460 | 10.2 % | 1.09 | 2.3 | 51 % | 1.79 | 20.5 % | 0.17/0.19 |
| **515 s4** | 550 | 0.79 | 1.09 | 544 | 9.5 % | 1.13 | 2.3 | 65 % | 1.32 | **15.2 %** | 0.33/0.35 |
| 514 s5 | 550 | 0.64 | 1.25 | 600 | 11.7 % | 1.18 | 2.9 | 43 % | 1.83 | 20.6 % | 0.39/0.37 |
| 513 s5 | 550 | 0.76 | 1.26 | 517 | 11.4 % | 1.17 | 2.4 | 51 % | 1.69 | 19.5 % | 0.35/0.31 |
| 514 s3 | 1000 | 0.81 | 1.03 | 299 | 4.5 % | 1.08 | 2.2 | 66 % | 1.49 | 16.7 % | 0.20/0.21 |
| 513 s3 | 1000 | 0.80 | 1.04 | 414 | 5.5 % | 1.07 | 2.2 | 67 % | 1.45 | 19.0 % | 0.17/0.20 |
| **515 s4** | 1000 | 0.78 | 1.10 | 247 | 5.8 % | 1.21 | 2.3 | 75 % | 0.98 | **10.4 %** | 0.34/0.36 |
| 514 s5 | 1000 | 0.59 | 1.33 | 339 | 7.2 % | 1.28 | 3.0 | 61 % | 1.31 | 12.8 % | 0.42/0.40 |
| 513 s5 | 1000 | 0.67 | 1.30 | 293 | 7.4 % | 1.24 | 2.4 | 68 % | 1.15 | 14.1 % | 0.41/0.37 |

510 s3/s5 JSON comes from an earlier probe version (shots/s 0.07-0.11, n al&near 12-120), so 510 is excluded from the table.

## Reading

- **Trigger**: flat and in dump. p_fire is 0.82 between 0 and 5° and 0.75-0.76 beyond 20°, on both sides. Exceeds 0.5 everywhere in deterministic and reserve is empty at 2.3 s in 98-99 % of tank-episodes. Ratio (1.09/1.10) sits between save_3 and save_5 of the other runs, as expected at 108 k.
- **Less frequent fire without being more selective**: 1.32/0.98 shells/s vs 1.5-1.8 for 513/514. Trigger does not restrain itself for that (p_arm 0.79). The gap comes from the higher disarmed-step share (65/75 % vs 51-67 %): the tank empties as fast, then stays disarmed longer. The CSV `fire` drop is thus explained by more time empty, not less dump.
- **Turret**: 15.2/10.4 % at <5°, under 513/514 at save_3 (20/17-19) and at save_5 (20/13-14). Only notable gap, in the wrong direction. With a 4-episode probe it may be seed noise.
- **σ canon**: 0.33-0.36, on the trajectory of 513/514 (0.2 at 81 k then 0.4 at 135 k). Nothing abnormal.
- **Aligned fire**: hit/shell 9.5/5.8 % and enrichment 1.13/1.21, at the other runs' level.

## Established / noise

- Established: at 108 k, gamma 0.998 does not change the trigger (flat, dump at 2.3 s), like 513/514 at the same stage.
- Possible noise: turret behind (−25 to −40 %) and longer disarmed time.
- No visible improvement at this stage.

## Useful next

- save_5 (135 k) vs 513/514 save_5 (ratio 1.25-1.33): does the trigger start to detach?
- save_9 (~243 k): landmarks 514 ratio 2.9/4.4, 513 3.1/3.7, 510 5.9/10.9.
- Turret: check whether the lag is confirmed on 2 more saves before attributing it to gamma.

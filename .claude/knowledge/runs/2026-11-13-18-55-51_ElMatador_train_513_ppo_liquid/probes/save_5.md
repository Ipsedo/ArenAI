# Probes 513 save_3 / save_5 / save_6 vs 510 save_3 / save_5

`build/` binary compiled 13/11 18:59, commit b3092240: no inference diff with run code, empty-reserve respawn rule identical to training's. Sides 550 and 1000, 4 episodes, stochastic, 32 tanks. Economy read from `config.json`: 10 shells at start, 5 per hit, max 30.
510 references are the probes made with binary 1e171bbf, under the same respawn rule (`train_510_ppo_liquid/probes/binaire_1e171bbf_avec_respawn/`, tables of `save_3.md`/`save_5.md`). 510's economy was 2/20: hit/shell, %disarmed and reserve are therefore comparable only in trend.

| run | save (index, D) | side | p_arm | ratio | hit/sh | h<5° | empty_s | %empty | %disarmed | shots/s | turret<5° | σ cx/cy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 510 | s3 (81 k, 400) | 550 | 0.77 | 1.05 | 10.4 % | 24.2 % | 2.5 | n/a | 54 % | 1.78 | 20.8 % | 0.19/0.24 |
| 513 | s3 (81 k, 400) | 550 | 0.81 | 1.05 | 10.2 % | 22.1 % | 2.3 | 92 % | 51 % | 1.79 | 20.5 % | 0.17/0.19 |
| 510 | s3 | 1000 | 0.75 | 1.05 | 5.2 % | 14.0 % | 2.3 | n/a | 66 % | 1.41 | 16.6 % | 0.19/0.24 |
| 513 | s3 | 1000 | 0.80 | 1.04 | 5.5 % | 13.4 % | 2.2 | 99 % | 67 % | 1.45 | 19.0 % | 0.17/0.20 |
| 510 | s5 (135 k, 450) | 550 | 0.42 | **1.86** | 12.8 % | 27.6 % | 3.4 | n/a | 47 % | 1.44 | 22.1 % | 0.38/0.35 |
| 513 | s5 (135 k, 450) | 550 | 0.76 | 1.26 | 11.4 % | 25.3 % | 2.4 | 88 % | 51 % | 1.69 | 19.5 % | 0.35/0.31 |
| 510 | s5 | 1000 | 0.34 | **2.12** | 7.3 % | 18.3 % | 3.0 | n/a | 55 % | 1.09 | 19.7 % | 0.46/0.41 |
| 513 | s5 | 1000 | 0.67 | 1.30 | 7.4 % | 19.3 % | 2.4 | 98 % | 68 % | 1.15 | 14.1 % | 0.41/0.37 |
| 513 | s6 (162 k, 500) | 550 | 0.64 | 1.40 | 12.0 % | 23.6 % | 2.5 | 95 % | 50 % | 1.79 | 23.7 % | 0.37/0.30 |
| 513 | s6 | 1000 | 0.57 | 1.51 | 7.2 % | 18.1 % | 2.4 | 100 % | 61 % | 1.40 | 18.0 % | 0.42/0.34 |

Aligned & <40 m counts: 293-659 (side 1000 of save_5 noisy). Deterministic (p > 0.5), save_6 fires on 89/76 % of aligned decisions and 62/51 % off-axis. No μ saturation, σ never at ceiling.

## Reading

- **Trigger**: at save_3, same start as in 510 (flat, ratio 1.05). Then it learns slower. At save_5, aligned/off-axis gap is 0.18-0.19 vs 0.30-0.35 in 510. Rises to 0.23-0.27 at save_6. At save_6, 513 has still not reached 510 save_5, neither in ratio nor level (p_arm 0.57-0.64 vs 0.34-0.42).
- **Dump**: 1st empty reserve stays frozen at 2.2-2.5 s over the three saves, whereas in 510 it was already at 3.0-3.4 s at save_5. 88 to 100 % of tank-episodes empty their reserve. The 5-shell refund does not relieve the reserve: armed share (17-24 % of decisions) is under 510 save_5's (31-34 %).
- **Precision and aim**: at equal fire precision, h<5° is at 510's level and hit/shell too (11-12 % / 7 %). Turret equivalent or slightly behind at save_5 (19.5/14.1 vs 22.1/19.7), then back to 23.7/18.0 at save_6.
- **Heads**: σ canon rises from 0.17-0.20 to 0.30-0.42, same trajectory as in 510.

## Established / noise

- Established, on two sides and two saves, and consistent with CSV (fire 1.7 /s vs 1.17 and hit 12 % vs 17.5 % at 150-180 k): 513's trigger lags 510's by about one save, and the dump does not recede.
- Undecided: cause. Compatible with off-axis fire less penalised by the 5/30 economy and full respawn, but 513 and 510 have one seed each.

## Next

- save_9 (243 k) vs 510 save_9 under the same rule (ratio 6.4/10.8, reserve 9.1 s).

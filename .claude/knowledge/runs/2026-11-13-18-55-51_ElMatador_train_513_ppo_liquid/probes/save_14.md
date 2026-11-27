# Probes 513 save_9 / save_14 vs 510 save_9 / save_14

Same binary as for save_5: b3092240, no inference diff with 513. For 510 save_14 the only diff is `probe.cpp` (PPG support), no effect on ppo_liquid, hence `--force`.
Sides 550 and 1000, 4 episodes, stochastic. Each probe reads its run's economy: 5/30 for 513, 2/20 for 510. For 510 save_9, values are those of the memory note (binary 1e171bbf, same respawn rule).

| run | save (index, D) | side | p_arm | ratio | <5°-only ratio | hit/sh | h<5° | shots<5° | empty_s | %empty | %disarmed | shots/s | turret<5° | σ cx/cy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 510 | s9 (243 k, 650) | 550 | n/a | 6.4 | n/a | 20.9 % | n/a | n/a | 9.1 | n/a | n/a | n/a | 20.8 % | n/a |
| 513 | s9 (243 k, 650) | 550 | 0.33 | 3.1 | 2.7 | 14.9 % | 30.4 % | 36 % | 6.0 | 82 % | 29 % | 1.59 | 21.8 % | 0.48/0.40 |
| 510 | s9 | 1000 | n/a | 10.8 | n/a | 14.9 % | n/a | n/a | 19.6 | n/a | n/a | n/a | 15.7 % | n/a |
| 513 | s9 | 1000 | 0.21 | 3.7 | 2.3 | 10.1 % | 25.9 % | 29 % | 7.6 | 87 % | 36 % | 1.04 | 15.0 % | 0.54/0.45 |
| 510 | s14 (378 k, 900) | 550 | 0.09 | **9.2** | 6.4 | **28.0 %** | **41.3 %** | 54 % | 13.6 | 60 % | 6.3 % | 0.76 | 23.1 % | 0.57/0.37 |
| 513 | s14 (378 k, 900) | 550 | 0.16 | 6.1 | 4.3 | 19.4 % | 31.1 % | 47 % | **21.7** | 60 % | 7.6 % | 1.27 | **25.8 %** | 0.44/0.33 |
| 510 | s14 | 1000 | 0.04 | **17.3** | 5.0 | **25.0 %** | **38.8 %** | 54 % | 38.8 | 62 % | 4.5 % | 0.34 | 19.5 % | 0.64/0.44 |
| 513 | s14 | 1000 | 0.08 | 10.0 | 3.3 | 16.4 % | 29.5 % | 45 % | 27.3 | 77 % | 7.3 % | 0.71 | **22.1 %** | 0.55/0.40 |

Aligned & <40 m counts: 409-904. Deterministic (p > 0.5), 513 save_14 fires on 32/16 % of aligned decisions, vs 21/9 % for 510 save_14. Median miss: 19.7/51.9 m in 513, vs 10.0/23.2 m in 510. No μ nor σ saturation.

## Reading

- **Dump resolved**: between save_6 and save_14, 513's 1st empty reserve goes from 2.4 s to 22-27 s and the disarmed share from 50-60 % to 7 %. Same level as 510 save_14 (4.5-6.3 %). There is a jump between save_6 and save_9, then a second between save_9 and save_14. 510 was at 9.1/19.6 s at save_9.
- **Selective but more tolerant trigger**: ratio reaches 6.1/10.0 at save_14. That is 510 save_9's level: at equal index, about 135 k lag remains. The aligned/off-axis gap is nonetheless larger than in 510 (0.40-0.47 vs 0.30-0.34). The difference comes from level: 513 fires 2× more often (1.27/0.71 vs 0.76/0.34 shots/s). It fires off-axis 2× more (0.079/0.052 vs 0.036/0.021) and fires farther: at <5° angle, with no distance condition, p_fire is 0.335/0.172 vs 0.230/0.105.
- **Precision per shell behind**: even on shots at <5°, hit/shell is 31/30 % vs 41/39 % in 510, and the turret is not responsible. Median miss 2× larger. Matches shots taken at longer distance, or with less tuned pitch.
- **Turret**: 513 slightly better (25.8/22.1 % vs 23.1/19.5 %). σ canon slightly lower (0.44-0.55 vs 0.57-0.64).

## Established / undecided

- Established: at 378 k, both runs have resolved the dump. 513 holds its reserve, but with 2× more frequent fire and 30 % less precision per shell.
- Probable: this regime is the optimum of the 5/30 economy, not a lag. At 5 shells per hit, a shot is profitable from 20 % hit chance (vs 50 % at 2/20). Shots at <5° toward a distant target (≈ 30 % hits) are thus profitable at 5/30 and losing at 2/20. 513 takes them and 510 discards them. CSV: kill 22-24 vs 20-22 in 510, reward r identical at equal D.
- Consequence to settle (Samuel): game stays at 2/20. With this model, in game, a tank at 19 % hits loses about 0.6 shell per shot and ends dry faster than with 510.

## Next

- save_16 and save_18, first save at D 1000 in 510 (429 k): does per-shell precision rise, or settle at the break-even threshold?
- If the gap with the game must be measured: probe 513 save_14 with `--shells_per_hit 2 --max_shells 20`.

## 510 save_14 replayed at 5/30 (13/11, 23:55)

510's checkpoint copied to the scratchpad with a `config.json` switched to 5/30 (same binary, `--force`). This is the game's current economy; the nb_shells/max_shells observation thus also changes for 510. First attempt rewrote `train_510_ppo_liquid/probes/save_14_side*`, but still at 2/20 (script falls back to `outputs/` when metrics.csv is missing). That is a 2nd probe seed at 2/20.

| probe | side | ratio | hit/sh | shots/s | empty_s | %disarmed | turret<5° | hits/min alive |
|---|---|---|---|---|---|---|---|---|
| 510 at 2/20, seed 1 | 550 / 1000 | 9.2 / 17.3 | 28.0 / 25.0 % | 0.76 / 0.34 | 13.6 / 38.8 | 6.3 / 4.5 % | 23.1 / 19.5 % | n/a |
| 510 at 2/20, seed 2 | 550 / 1000 | 8.2 / 14.2 | 25.4 / 26.8 % | 0.79 / 0.37 | 10.6 / 25.1 | 9.5 / 5.7 % | 26.0 / 20.0 % | 13.0 / 6.3 |
| 510 at 5/30 | 550 / 1000 | 7.6 / 19.6 | 22.7 / 21.2 % | 0.78 / 0.35 | 18.5 / 38.4 | 4.5 / 2.9 % | 22.9 / 20.1 % | 11.5 / 4.7 |
| 513 at 5/30 | 550 / 1000 | 6.1 / 10.0 | 19.4 / 16.4 % | 1.27 / 0.71 | 21.7 / 27.3 | 7.6 / 7.3 % | 25.8 / 22.1 % | **16.2 / 7.7** |

- At 5/30, 510 keeps its way of firing: same cadence, same sorting. It almost never empties its reserve (disarmed 3-5 %) and does not exploit the extra shells.
- 513 hits 1.4 to 1.6 times more per minute of living tank. More lethal, at the cost of slightly more disarmed time (7 %).
- Noise between the two seeds of 510 at 2/20: reserve ±25-35 %, ratio ±15 %, hit/shell ±3 points.
- Limit: in homogeneous self-play, lethality measures the population against itself. Not a 510 vs 513 duel.

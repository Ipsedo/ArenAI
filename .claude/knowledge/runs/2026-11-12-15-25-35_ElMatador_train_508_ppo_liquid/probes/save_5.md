# Probe 508 save_5 (135 k, D 450) vs 507 save_5

`build/` binary of 12/11 15:24, at commit 25a75a34 with the uncommitted stabiliser (GunLaying, 20 Hz hinges): the run's code. No inference diff. 4 episodes per side, stochastic, economy 10 / +2 / 20. 507 save_5: same index, same D, probed with the 08:01 binary (without stabiliser), so each with its own game.

| run | side | p_arm | ratio | p_al / p_off | hit/sh | h<5° | empty_s | shots/s | hits/tank/min | turret<5° | σ cx/cy | \|μ\|cx |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 508 | 550 | 0.31 | 1.9 | 0.51 / 0.27 | 9.9 % | 21.3 % | 4.4 | 1.26 | 7.5 | 17.4 % | 0.40/0.36 | 0.26 |
| 507 | 550 | 0.15 | 2.9 | 0.30 / 0.11 | 10.0 % | 18.4 % | 6.1 | 0.75 | 4.5 | 13.2 % | 0.45/0.49 | 0.15 |
| 508 | 1000 | 0.26 | 2.2 | 0.50 / 0.23 | 4.3 % | 12.0 % | 4.4 | 0.93 | 2.4 | 13.1 % | 0.43/0.39 | 0.26 |
| 507 | 1000 | 0.12 | 3.7 | 0.33 / 0.09 | 5.1 % | 12.1 % | 8.2 | 0.60 | 1.8 | 9.6 % | 0.49/0.52 | 0.14 |

CSV at equal D (508 / 507): kill per episode 15.5 / 10.5 at D 300, 18.4 / 13.8 at D 400, 18.0 / 15.3 at D 450; fire 1.3-1.6 / 0.9-1.1 shots/s; hit per shot identical (0.11-0.12). D milestones identical up to 450.

## Pitch (attitude columns, turret aligned <3°, target 10-130 m)

| | gap std | \|e\| < 2° |
|---|---|---|
| 508 save_5, 550 / 1000 | 8.8° / 11.1° | 28 % / 27 % |
| 507 save_12 (12/11 probe) | 10.9-11.9° | 17-19 % |
| stabilised barrel without command (prediction) | 4.8-5.5° | 28-36 % |

- Predicted gain is there: 27-28 % of aligned decisions within 2° pitch, vs 17-19 % before. Bottom of the « stabilised barrel without command » range.
- Agent's pitch command still brings nothing: corr(command, target) ≈ 0 and gap (std 9-11°) stays above that of the stabilised barrel left as is (5°). Own agitation persists.
- Aligned shots <5° at target <50 m: 69-75 % hits vs 59-67 % for 507. All ranges under 50 m: 40-41 % vs 35-37 %.

## Reading

- **Kill record comes mostly from fire rate.** 508 fires 1.5-1.7× more and empties its reserve faster (4.4 s vs 6-8 s), with same hit per shell. Hits/tank/min: 7.5 / 2.4 vs 4.5 / 1.8, i.e. 1.7× side 550 and 1.3× side 1000.
- **Stabiliser shows in aim.** Turret <5°: 17.4 / 13.1 % vs 13.2 / 9.6 % (+35 % on both sides). Pitch <2°: 28 % vs 17-19 %. Short-range pitch precision. Expected stabiliser effect, and 508's only difference vs 507.
- **Trigger less selective.** Ratio 1.9-2.2 vs 2.9-3.7, because p_off rises (0.23-0.27 vs 0.09-0.11). Dump remains: 90-98 % of tanks empty. Deterministic: trigger stays almost mute (0.12-0.22 aligned).
- **Nothing improves beyond 100 m.** Hits there equal or lower (1-4 % per shell), consistent with oracle-measured ceiling.

## Established / noise

- Established, single seed: pitch improves as predicted by attitude measure, turret more often aligned and CSV kills +20-45 % at equal D. Both sources agree.
- Undecided: is the more prolific trigger a consequence (fire becomes more profitable when the barrel holds) or seed noise? 504/507 already showed ×1.5 gaps in fire rate at equal D.
- CSV has not passed D 450. Curriculum not faster for now (milestones identical to 507).

## Next

save_9 (243 k) and save_12 vs 507 at same saves: turret<5° (507: ≤ 13 %), pitch <2°, hits/tank/min (507 save_12: 5.4 / 1.9), D milestones beyond 450 (507: D 500 at 145 k, D 700 at 267 k).

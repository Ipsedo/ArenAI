# train_514_ppo_liquid

| | |
|---|---|
| machine | ElMatador |
| user | samuel |
| launch | 2026-11-14 01:17:39 |
| commit at launch | `3cd9e210` — logging level |
| commits ≤ 90 min after | `d1bb4020` |
| diff at launch | unknown (identity recorded after the run ended) |
| raw data | `outputs/train_514_ppo_liquid` on ElMatador |

## Established facts

- 2026-11-14: Run 514 = a·|a| on turret/canon, centred potential, economy 10/3/30. Effective commit `d1bb4020`. The `build/` probe binary (01:13) dates from `3cd9e210` by mtime but contains a·|a| (compiled from working tree): `--force` legitimate. Test of a·|a| + centred potential + 3/hit vs 510 and 513.
- 2026-11-14: save_3 (81 k, D 400): trigger flat and in dump (ratio 1.03, reserve empty 2.3 s), turret 20/17 % ≡ 510/513; |μ|cx 2× larger, mechanical effect of a·|a|. save_23 (621 k, D 1000): ratio 4.9/11.1, hit/shell 22/18 %, h<5° 34/32 %, reserve 9.6/16 s, disarmed 12/15 %, turret <5° 26/19 % ⟹ ≡ 510 save_28, ahead of 513 save_18 (hit/shell 17.6/15.1). Movement saturated at fixed direction (μ dir ≈ −1, −0.9). CSV 600-630 k: hit 17-18 %, kill 22-23. Analyses `outputs/train_514_ppo_liquid/probes/save_3.md`, `save_23.md`.
- 2026-11-14: save_20-22: oscillating regime like 513: save_20 ≡ save_23 (ratio 6.2/14.7, reserve 5.8/19.3 s); save_21 relapse (ratio 3.3/7.1, disarmed 27/23 %, fire 1.65/1.05/s, σ canon trough 0.20/0.16), save_22 half. Invariants over the 4: turret <5° 26-28/19-23 %, aligned shots 40-53 %, h<5° 28-34 %. Mean ≈ 510 save_28 (hit/shell 19/17 vs 21/19.5). Movement saturated at fixed direction on the 4 saves. Analysis `probes/save_20_23.md`.
- 2026-11-14: Effect of a·|a| + Φ (`probes/a_abs_a_phi_effet.md`): trigger takes off like 513 (ratio save_9 2.9/4.4 vs 510 5.9/10.9) ⟹ economy or seed, not a·|a|. Fine aim (shots <5° on target = nearest, 50-200 m, filter `|fire_best_distance − fire_distance| < 1` since fire_yaw/pitch are relative to the nearest): pitch 1.0-1.2° from save_9 vs 1.3-1.8 (510 only reaches it at save_16-28, 513 never), hits 48-52 % vs 37-39 %; yaw no net lead; final level ≡ 510. Reading: a·|a| accelerates fine pitch precision, does not change the ceiling. One seed, a·|a| and Φ confounded. Only one late save probed per point: judge over 3-4 saves.
- 2026-11-14: save_34 (918 k) stochastic vs deterministic (`probes/save_34.md`): deterministic reserve 13.8/28 s vs 9.8/15.2, disarmed 6/3 % vs 13/11 %, cadence at 1000 ÷2 ⟹ dump is half as strong in game. But share of shells fired at ≥ 20° identical (16-19 % determ. vs 17-21 % stoch) ⟹ off-axis fire = μ decision (p > 0.5 on 2-8 % of off-axis states), not noise. Turret 30/23 % (best of 514, ≡ 510 s28). New: μ dir_x saturated 13.5 %, p_zoom 0.04-0.11 (zoom abandoned), corr fire/zoom 0.86 → 0.35.

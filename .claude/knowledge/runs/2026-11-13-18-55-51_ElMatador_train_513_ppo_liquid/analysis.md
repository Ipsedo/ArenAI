# Run 513 (ppo_liquid) — start, 0-178 k

Launched 13/11 18:55, last row 20:43, **running**. Index 178 k, D 500. Throughput 99 k steps/h, 3.2 M tank-steps/h (≡ 510).
Effective commit: **b3092240**, high confidence. `cmake-build-release` binary dates 18:55 and the commit 18:59. Code diff vs 510 (7422d098) limited to `main.cpp`, no inference difference with the probe binary (b3092240).

## What the run tests

- 510's rule restored (full respawn + truncation at empty reserve): net diff of `train_environment.cpp` vs 7422d098 is empty.
- **Only change**: training economy `shells_per_hit` 2 → 5 and `max_shells` 20 → 30 (`main.cpp` defaults). Game stays 2/20.
- Hypothesis (memory note on run 513 economy): reserve becomes sustainable from 20 % hits, a level 510 reached.

## CSV reading vs 510 at same D

D milestones **identical at index** in 513 and 510 (300 @ 21 k, 400 @ 67 k, 450 @ 114 k, 500 @ 145 k), as in 508-510: they do not discriminate runs.

| index | run | hit | fire /s | kill | s | Hc | kl | skip_canon |
|---|---|---|---|---|---|---|---|---|
| 0-40 k | 513 / 510 | 10.9 / 9.9 % | 1.67 / 1.56 | 17.6 / 13.1 | 47 / 56 | −0.23 / −0.20 | 0.068 / 0.038 | 0.08 / 0.06 |
| 40-80 k | 513 / 510 | 12.4 / 12.9 % | 1.72 / 1.70 | 23.5 / 22.1 | 47 / 50 | −0.41 / −0.52 | 0.086 / 0.101 | 0.22 / 0.13 |
| 80-120 k | 513 / 510 | 12.7 / 13.7 % | 1.75 / 1.61 | 24.3 / 23.2 | 46 / 54 | −0.14 / −0.09 | 0.043 / 0.043 | 0.20 / 0.16 |
| 120-150 k | 513 / 510 | 12.9 / 14.8 % | 1.66 / 1.37 | 24.3 / 23.0 | 46 / 48 | 0.07 / 0.06 | 0.043 / 0.048 | 0.12 / 0.14 |
| 150-180 k | 513 / 510 | 12.2 / 17.5 % | 1.71 / 1.17 | 22.5 / 23.7 | 49 / 49 | 0.13 / 0.15 | 0.055 / 0.045 | 0.11 / 0.11 |

- **Up to 120 k, 513 and 510 are confounded** within noise. 513 starts faster at D 250 (kill 14.4 vs 9.5).
- **From 120 k the runs diverge.** In 510, cadence falls (1.6 → 1.17 /s) and precision rises (13.7 → 17.5 %): trigger selectivity sets in. In 513, nothing moves: cadence 1.7 /s, precision 12-13 %. Kills stay equal, ~23 per episode.
- Rest equal: r_rcv, ev (0.47 vs 0.49 at D 500), Hc and α. α_canon_x rises to 0.03 in both. Joint skip null, final kl (0.055) just above target, no KL regime.

## Diagnosis

- **Established (CSV + probes save_3/5/6, see `probes/save_5.md`)**: at 135 k and 162 k, 513's trigger lags 510's. Probe aligned/off-axis ratio 1.26-1.30 at save_5 and 1.40-1.51 at save_6, vs 1.86-2.12 for 510 at save_5. 513's trigger also stays higher (p_arm 0.57-0.76 vs 0.34-0.42), first reserve empties at 2.4 s vs 3.0-3.4 s. Start at save_3 identical in both (flat trigger, ratio 1.04-1.05).
- **Probable mechanism, undecided**: with 5 shells refunded per hit, emptying the reserve costs less. An aligned shot earns +0.25 net shell at 25 % hits, an off-axis shot costs −0.95. Also, respawn at empty reserve reloads the tank full. Pressure to sort shots is thus weaker than at 2/20, where even an aligned shot loses (−0.5). Trigger still learns but slower. With a single seed per run, the lag may also be seed noise.
- **Within noise**: aim. Probe turret <5° 19.5/14.1 % (save_5) then 23.7/18.0 % (save_6), vs 22.1/19.7 % for 510 at save_5. hit/shell 11.4/7.4 % vs 12.8/7.3 %.

## Options (no decision)

1. Let run and judge at save_9 (243 k) vs 510 save_9 under same rule (ratio 6.4/10.8, reserve 9.1 s). 513's hypothesis is only truly tested once hit ≥ 20 %, at D ≥ 600 in 510.
2. If lag persists at save_9, separate it from seed: relaunch 510 and 513 with a second seed each.

## Landmarks

- CSV: `fire` and `hit` at D 550-650. In 510 hit passes 20 % at D 550 and `fire` falls under 0.9 /s.
- Probe save_9: ratio, p_arm, 1st reserve empty and %empty. 513 only makes sense if the reserve eventually holds.

## Update 13/11 23:00 — index 390 k, D 900

| index | run | D | hit | fire /s | kill |
|---|---|---|---|---|---|
| 180-240 k | 513 / 510 | 600-650 | 12-13 / 20-21 % | 1.6-1.7 / 0.9 | 24 / 22-23 |
| 270-300 k | 513 / 510 | 750 | 14.4 / 23.2 % | 1.05 / 0.50 | 22.7 / 19.5 |
| 300-420 k | 513 / 510 | 850-950 | 14-18 / 20-22 % | 0.85-1.4 / 0.8-1.0 | 21-23 / 19.5-20.5 |

- D milestones remain identical at index (D 900 at 366 k). Reward r equal at equal D (4.4e-3 at D 900).
- In 513, cadence only falls from 270 k (1.7 → 0.85-1.4 /s), about 100 k after 510. CSV hit plateaus at 14-18 %, vs 20-23 % for 510. Kills stay slightly higher in 513.
- Probes save_9 and save_14 (`probes/save_14.md`): dump resolved (1st reserve at 22-27 s, disarmed 7 %, like 510). Trigger selective (ratio 6/10) but fires 2× more than 510's, farther, with 30 % fewer hits per shell. Probable reading: optimum of the 5/30 economy (break-even at 20 %), not a lag. This policy is offset from the game, which stays at 2/20.

# Probe 528 save_14: trigger closes at 50-100 m, turret has not learned to hold

Run 528 (ppo_liquid, kills/tank curriculum + `shells_per_hit` 5, tree uncommitted on 68c41e20). save_14 = 378 k, D 750 (blocked since 304 k). Binary of 2026-11-17 08:32, no inference diff. Sides 550 and 1000, 4 episodes, stochastic, run economy.

| | tour<5° | tour<20° | ratio | p_al | p_off | hit/sh | h<5° | shots<5° | shots/s | touches/tank-s | 1st empty | disarmed | σ canon | \|μ\| cx | mean μ cx |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **528 s14 @550** | 13.9 % | 56.3 % | 21 | 0.34 | 0.016 | 24.0 % | 41 % | 36 % | 0.40 | 0.100 | 62 s | 2.4 % | 0.44-0.53 | 0.52 | +0.21 |
| **528 s14 @1000** | 12.1 % | 54.3 % | 39 | 0.39 | 0.010 | 19.6 % | 36 % | 28 % | 0.19 | 0.037 | 83 s | 1.6 % | 0.48-0.56 | 0.51 | +0.26 |
| 528 s9 @550 | 15.6 % | 55.8 % | 11.4 | 0.64 | 0.056 | 17.3 % | 35 % | 33 % | 0.83 | 0.150 | 27 s | 4.6 % | 0.49-0.57 | 0.45 | +0.19 |
| 528 s9 @1000 | 15.0 % | 58.5 % | 21 | 0.66 | 0.031 | 14.7 % | 34 % | 32 % | 0.44 | 0.067 | 37 s | 4.9 % | 0.53-0.60 | 0.47 | +0.25 |
| 524 s16 @550 | 10.1 % | 48.6 % | — | — | 0.012 | 28.0 % | — | — | 0.26 | 0.073 | 38 s | 3.8 % | — | — | — |
| 525 s12 @550 | 6.4 % | 41.2 % | — | — | 0.006 | 33.6 % | — | — | 0.12 | 0.039 | 84 s | 1.5 % | — | — | — |
| 522 s16 @550 (separate credit) | 45.1 % | 77.1 % | — | — | 0.006 | 46.2 % | — | — | 0.16 | 0.075 | 85 s | 0.1 % | — | — | — |
| 520 s16 @550 (separate credit) | 33.3 % | 69.9 % | — | — | 0.015 | 30.2 % | — | — | 0.29 | 0.088 | 60 s | 1.4 % | — | — | — |

CSV: D 750 since 304 k (95 k index), kill 22.8 (D 600) → 14.4-15.9, fire 0.94 → 0.3-0.5, hit 0.17 → 0.24. At D 750: 525 8.5, 524 13.1, 522 16.3, 520 22.9.

## By distance band (best-aligned target)

Touches per tank-second:

| | total | < 50 m | 50-100 m | 100-200 m | > 200 m |
|---|---|---|---|---|---|
| **528 s14 @550** | 0.100 | 0.062 | **0.018** | 0.006 | 0.014 |
| 528 s9 @550 | 0.150 | 0.074 | 0.050 | 0.013 | 0.014 |
| 528 s5 @550 | 0.121 | 0.081 | 0.016 | 0.008 | 0.015 |
| 524 s16 @550 | 0.073 | 0.052 | 0.007 | 0.004 | 0.010 |
| 525 s12 @550 | 0.039 | 0.029 | 0.002 | 0.003 | 0.005 |
| 522 s16 @550 | 0.075 | 0.029 | 0.028 | 0.010 | 0.008 |
| 520 s16 @550 | 0.088 | 0.040 | 0.030 | 0.009 | 0.010 |
| **528 s14 @1000** | 0.037 | 0.025 | **0.004** | 0.001 | 0.008 |
| 528 s9 @1000 | 0.067 | 0.035 | 0.018 | 0.002 | 0.013 |
| 524 s16 @1000 | 0.019 | 0.013 | 0.001 | 0.001 | 0.004 |
| 522 s16 @1000 | 0.035 | 0.011 | 0.014 | 0.005 | 0.005 |
| 520 s16 @1000 | 0.031 | 0.012 | 0.013 | 0.001 | 0.004 |

hit/shell all shots — aligned shots < 5°:

| | < 50 m | 50-100 m | 100-200 m | > 200 m |
|---|---|---|---|---|
| **528 s14 @550** | 54 — 70 % | 24 — **39 %** | 7 — 16 % | 10 — 21 % |
| 528 s9 @550 | 56 — 78 % | 28 — 47 % | 7 — 17 % | 4 — 9 % |
| 524 s16 @550 | 61 — 76 % | 17 — 31 % | 8 — 22 % | 12 — 24 % |
| 522 s16 @550 | 74 — 84 % | 55 — 62 % | 30 — 44 % | 21 — 27 % |
| 520 s16 @550 | 76 — 84 % | 40 — 52 % | 14 — 27 % | 10 — 16 % |
| **528 s14 @1000** | 56 — 72 % | 18 — **31 %** | 3 — 7 % | 8 — 16 % |
| 528 s9 @1000 | 64 — 81 % | 26 — 43 % | 4 — 10 % | 5 — 14 % |

Armed p_fire on target aligned < 5° (count):

| | off-axis | < 50 m | 50-100 m | 100-200 m | > 200 m |
|---|---|---|---|---|---|
| **528 s14 @550** | 0.016 | 0.35 (1 889) | **0.18 (2 738)** | 0.07 | 0.06 |
| 528 s9 @550 | 0.056 | 0.62 (889) | 0.44 (2 144) | 0.18 | 0.13 |
| 524 s16 @550 | 0.012 | 0.36 (1 197) | 0.15 (1 673) | 0.06 | 0.04 |
| 525 s12 @550 | 0.006 | 0.10 (1 424) | 0.017 (1 103) | 0.013 | 0.009 |
| 522 s16 @550 | 0.006 | 0.04 (9 423) | 0.04 (18 528) | 0.02 | 0.015 |
| 520 s16 @550 | 0.015 | 0.12 (3 262) | 0.08 (8 998) | 0.04 | 0.03 |
| **528 s14 @1000** | 0.010 | 0.35 (854) | **0.14 (1 253)** | 0.03 | 0.02 |
| 528 s9 @1000 | 0.031 | 0.64 (627) | 0.38 (1 387) | 0.10 | 0.06 |
| 524 s16 @1000 | 0.006 | 0.37 (411) | 0.07 (941) | 0.02 | 0.01 |

Turret (yaw toward nearest enemy, side 550):

| | tour<5° at < 50 m | tour<5° at 50-100 m | Δ\|yaw\| at 1 s from 5-30°, 50-100 m | mean hold < 5° (p90) |
|---|---|---|---|---|
| 528 s5 | 9.9 % | 5.0 % | +26° | 0.22 s (0.4) |
| 528 s9 | 11.5 % | 7.4 % | +15° | 0.26 s (0.5) |
| **528 s14** | 10.5 % | 6.1 % | +13° | **0.20 s (0.4)** |
| 522 s16 | 24.4 % | 22.2 % | +1° | 1.12 s (2.2) |
| 520 s16 | 18.5 % | 14.0 % | +3° | 1.02 s (2.5) |

## Reading

- The trigger closes at 50-100 m: aligned p_fire 0.44 → 0.18 side 550 and 0.38 → 0.14 side 1000, at 524 s16's level (0.15/0.07). It closes everywhere in the same proportions (< 50 m: 0.62 → 0.35; off-axis: 0.056 → 0.016), cadence halved (0.83 → 0.40 shot/s).
- Touches at 50-100 m fall back to save_5's level: 0.050 → 0.018 side 550, 0.018 → 0.004 side 1000. Still 2.5 to 4 times above 524 s16, but under 520/522 (0.028-0.030).
- Precision at 50-100 m regresses: aligned shot 47 → 39 % and 43 → 31 %. 520/522 are at 52-62 %.
- The turret has not learned to hold: hold under 5° 0.26 → 0.20 s (520/522: 1.0-1.1 s), tour<5° 15.6 → 13.9 % and 15.0 → 12.1 % (520/522: 33-45 %). It deviates a bit less (+15° → +13°) and μ/yaw gain is stable (−0.004/°, corr −0.13). The distance-touch peak of save_9 was not converted into aim.
- Total throughput still highest side 550 (0.100 vs 0.073-0.088), carried by point blank (0.062, 62 % of touches); side 1000 0.037 ≈ 520/522.
- Dump: gone (disarmed 2.4/1.6 %, 1st empty at 62/83 s, 37-41 % empty).
- Heads: dir_x and dir_y saturated 10-11 % of time (s9: 1 % and 9 %), canon_x 3 %; |μ| dir_x 0.73 (s9: 0.50). σ canon 0.44-0.56, no ceiling. Deterministic: 10 % of aligned & close states fire side 550, 4 % side 1000.
Established: the "trigger closed at range" lock is closing, ~100-130 k late vs 524/525 (243 k → 378 k). The 5 shells per touch delayed the closing, did not teach aim: target hold never exceeded 0.26 s. Established: the kills/tank criterion freezes D at 750 since 304 k (kill 14.4-15.9 = 0.45-0.50 per tank); under the old criterion D would have kept rising as in 524/525. Noise: end-of-window CSV rise (kill 14.4 → 15.9, r 1.4e-3 → 2.0e-3 over 40 k). Not settled: whether holding D 750 (near encounters kept) lets the turret progress, or the trigger continues toward 525 s12 (0.017 at 50-100 m); role of saturated direction heads and `alpha_dir_y` (3.9e-3).
Options (Samuel): let run to save_18-20 (test of what the criterion does when D is held); stop: trajectory = 524 shifted, the run has answered (economy delays the lock without lifting it); lead already noted 2026-11-16/17: only separate-aim-credit runs (520, 522, control) have a turret that holds the target (1.0-1.1 s, tour<5° 33-45 %).
Markers for save_18 (486 k): p_fire aligned 50-100 m side 550 ≥ 0.15 held, or < 0.05 (525 s12: 0.017); hold < 5° > 0.4 s and tour<5° > 20 % = the turret finally learns; ≤ 0.2 s = no; CSV: D restarts above 750 (kill > 16) or falls back (kill < 9.6).

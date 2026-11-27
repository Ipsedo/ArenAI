# Probe 528 save_9: trigger stays open at 50-100 m, touches follow

Run 528 (ppo_liquid, kills/tank curriculum + `shells_per_hit` 5, tree uncommitted on 68c41e20). save_9 = 243 k, D 600. Binary of 2026-11-17 08:32, no inference diff. Sides 550 and 1000, 4 episodes, stochastic, run economy (10 / 5 / 30).

| | tour<5° | tour<20° | ratio | p_al | p_off | hit/sh | h<5° | shots<5° | shots/s | touches/tank-s | 1st empty | disarmed | σ canon | \|μ\| cx | mean μ cx | n aligned & close |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **528 s9 @550** | **15.6 %** | 55.8 % | 11.4 | 0.64 | 0.056 | 17.3 % | 35 % | 33 % | **0.83** | **0.150** | 27 s | 4.6 % | 0.49-0.57 | 0.45 | +0.19 | 495 |
| **528 s9 @1000** | **15.0 %** | 58.5 % | 21 | 0.66 | 0.031 | 14.7 % | 34 % | 32 % | **0.44** | **0.067** | 37 s | 4.9 % | 0.53-0.60 | 0.47 | +0.25 | 355 |
| 528 s5 @550 | 11.4 % | 50.7 % | 4.7 | 0.58 | 0.124 | 11.7 % | 28 % | 20 % | 1.01 | 0.121 | 12.7 s | 24 % | 0.51-0.59 | 0.52 | +0.29 | 683 |
| 528 s5 @1000 | 10.9 % | 47.0 % | 7.2 | 0.64 | 0.090 | 6.8 % | 21 % | 19 % | 0.62 | 0.043 | 14.1 s | 34 % | 0.56-0.63 | 0.59 | +0.45 | 345 |
| 525 s9 @550 | 6.6 % | 37.5 % | 42 | 0.43 | 0.010 | 31.3 % | 50 % | 23 % | 0.24 | 0.076 | 43 s | 3 % | 0.58-0.59 | 0.49 | +0.33 | 731 |
| 525 s9 @1000 | 4.1 % | 27.7 % | 92 | 0.47 | 0.005 | 26.9 % | 48 % | 21 % | 0.11 | 0.030 | 77 s | 2.5 % | 0.59-0.60 | 0.50 | +0.41 | 322 |
| 524 s9 @550 | 12.4 % | — | 25 | 0.31 | 0.012 | 32.4 % | 59 % | 28 % | 0.21 | 0.068 | — | 2 % | 0.59-0.61 | 0.53 | — | — |
| 524 s9 @1000 | 14.8 % | — | 54 | 0.24 | 0.004 | 22.3 % | 43 % | 22 % | 0.07 | 0.016 | — | 1 % | — | — | — | — |
| control 430 s9 @550 | 33.7 % | — | 10 | 0.14 | — | 34.9 % | — | 61 % | — | 0.101 | — | 1 % | 0.36-0.50 | 0.20-0.22 | — | — |

CSV at same index: hit 0.157, kill 22.7, fire 0.8-0.95, skip 0.31 (525: 0.218 / 16.5 / 0.28; 524: 0.245 / 18.1 / 0.33).

## By distance band

Touches per tank-second:

| | total | < 50 m | 50-100 m | 100-200 m | > 200 m |
|---|---|---|---|---|---|
| **528 s9 @550** | **0.150** | 0.074 | **0.050** | 0.013 | 0.014 |
| 528 s5 @550 | 0.121 | 0.081 | 0.016 | 0.008 | 0.015 |
| 525 s9 @550 | 0.076 | 0.055 | 0.005 | 0.005 | 0.011 |
| 524 s9 @550 | 0.068 | 0.049 | 0.005 | 0.005 | 0.009 |
| 522 s9 @550 (separate credit) | 0.098 | 0.041 | 0.027 | 0.012 | 0.016 |
| 522 s24 @550 | 0.160 | 0.063 | 0.055 | 0.021 | 0.021 |
| **528 s9 @1000** | **0.067** | 0.035 | **0.018** | 0.002 | 0.013 |
| 525 s9 @1000 | 0.030 | 0.022 | 0.001 | 0.001 | 0.006 |
| 524 s9 @1000 | 0.016 | 0.012 | 0.001 | 0.000 | 0.003 |
| 522 s9 @1000 | 0.056 | 0.027 | 0.013 | 0.002 | 0.014 |

hit/shell all shots — aligned shots < 5°:

| | < 50 m | 50-100 m | 100-200 m | > 200 m |
|---|---|---|---|---|
| **528 s9 @550** | 56 — 78 % | **28 — 47 %** | 7 — 17 % | 4 — 9 % |
| 528 s5 @550 | 47 — 77 % | 10 — 31 % | 3 — 11 % | 4 — 8 % |
| 525 s9 @550 | 61 — 81 % | 19 — 35 % | 13 — 26 % | 13 — 28 % |
| 524 s9 @550 | 66 — 88 % | 18 — 38 % | 12 — 30 % | 13 — 34 % |
| 522 s9 @550 | 64 — 82 % | 27 — 39 % | 11 — 21 % | 10 — 16 % |
| **528 s9 @1000** | 64 — 81 % | **26 — 43 %** | 4 — 10 % | 5 — 14 % |
| 522 s9 @1000 | 67 — 83 % | 29 — 45 % | 5 — 10 % | 7 — 17 % |

Armed p_fire on target aligned < 5° (count):

| | off-axis | < 50 m | 50-100 m | 100-200 m | > 200 m |
|---|---|---|---|---|---|
| **528 s9 @550** | 0.056 | 0.62 (889) | **0.44 (2 144)** | 0.18 | 0.13 |
| 528 s5 @550 | 0.124 | 0.55 (984) | 0.38 (1 321) | 0.23 | 0.20 |
| 525 s9 @550 | 0.010 | 0.38 (966) | 0.14 (944) | 0.07 | 0.04 |
| 524 s9 @550 | 0.012 | 0.28 (1 503) | 0.07 (1 895) | 0.03 | 0.02 |
| 522 s9 @550 | 0.036 | 0.11 (3 699) | 0.14 (5 912) | 0.09 | 0.06 |
| **528 s9 @1000** | 0.031 | 0.64 (627) | **0.38 (1 387)** | 0.10 | 0.06 |
| 525 s9 @1000 | 0.005 | 0.44 (435) | 0.14 (288) | 0.06 | 0.03 |
| 524 s9 @1000 | 0.004 | 0.22 (558) | 0.04 (826) | 0.01 | 0.006 |

## Reading

- Trigger at 50-100 m: held open, even rising (0.38 → 0.44 side 550, 0.36 → 0.38 side 1000) where 525 went 0.38 → 0.14 and 524 to 0.07. Still selective: off-axis 0.124 → 0.056 and 0.090 → 0.031, ratio aligned & close 11/21, angle-only ratio 4.0/3.3 (s5: 2.2/1.8).
- Touches at 50-100 m ×3 since s5 (0.016 → 0.050 side 550, 0.009 → 0.018 side 1000), ten times 525/524 at same index, twice 522 s9, at 522 s24's level. Total throughput (0.150/0.067) highest measured at this save.
- Precision at 50-100 m: 31 → 47 % and 40 → 43 % for an aligned shot; 525/524 plateaued at 35-38 %, 522 s9 was at 39/45 %.
- Aim: starts. tour<5° 11.4 → 15.6 % and 10.9 → 15.0 % (525 regressed to 6.6/4.1 %), tour<20° 51 → 56 % and 47 → 59 %. Aligned states at 50-100 m 1 321 → 2 144 side 550. Still far from control (34 %). Mean μ canon_x +0.19/+0.25: sweep bias decreasing (s5: +0.29/+0.45).
- Dump: resorbed. Disarmed 24 → 4.6 % and 34 → 4.9 %, 1st empty at 27/37 s. 59-70 % of tank-episodes still empty at some point.
- Beyond 100 m: nothing. Aligned shot at 100-200 m: 17/10 % success, 0.013/0.002 touch/tank-s; 42-61 % of shells still go toward targets beyond 200 m for 4-5 % success, and 25 % of shots leave at ≥ 20° from any target (1 % success).
- Heads: σ canon 0.5-0.6 no ceiling, |μ| canon_x 0.45; dir_y saturated 8-9 % of time (s5: 1-2 %). Deterministic: 21 % of aligned & close states fire side 550, 9 % side 1000.
Established: the four markers set at save_5 passed (aligned p_fire 50-100 m ≥ 0.25, touches/tank-s at 50-100 m ≥ 0.025, aligned hit > 35 %, tour<5° > 15 %). The "trigger closed at range" lock of 524/525 did not close between s5 and s9; gap with them is an order of magnitude in the 50-100 m band, beyond seed noise. CSV plateau reading: kill and hit flat from 120 k to 243 k hide a shift of touches from point blank (0.081 → 0.074) toward 50-100 m (0.016 → 0.050) while D rose 400 → 600.
Confounded: 5-shell economy and kills/tank criterion. The criterion slowed nothing (D at same indices as 524/525), so the effect plausibly comes from the economy; no run separates them. Not settled: holding at D ≥ 700 (525: kill 16 → 8), aim beyond 100 m, effect of the KL regime (skip 31 %) and rising direction α.
Markers for save_13-14 (350-380 k, D ~800): touches/tank-s at 50-100 m ≥ 0.05 side 550 and rising side 1000; 100-200 m band taking off (aligned hit > 25 %); tour<5° > 20 %; p_fire off-axis < 0.03 without closing at 50-100 m; CSV kill ≥ 20 at D 800 (525: 8-9; 520/522: 13.5-20).

## Addendum: does the turret learn? (yaw toward nearest enemy, side 550)

| | tour<5° at < 50 m | tour<5° at 50-100 m | μ/yaw slope at 50-100 m (corr) | Δ\|yaw\| at 1 s from 5-30°, 50-100 m | mean hold < 5° (p90) |
|---|---|---|---|---|---|
| 528 s5 | 9.9 % | 5.0 % | −0.0019 (−0.07) | +26° | 0.22 s (0.4) |
| **528 s9** | 11.5 % | 7.4 % | −0.0038 (−0.13) | +15° | 0.26 s (0.5) |
| 525 s5 | 8.1 % | 4.2 % | −0.0017 (−0.09) | +19° | 0.23 s (0.5) |
| 525 s9 | 7.8 % | 3.3 % | −0.0035 (−0.15) | +24° | 0.19 s (0.4) |
| 522 s9 (separate credit) | 14.6 % | 11.5 % | −0.0012 (−0.07) | +8° | 0.80 s (1.9) |

Side 1000: same direction (50-100 m: 4.7 → 6.5 %, Δ +29° → +16°, hold 0.21 → 0.23 s). Beyond 100 m: slope ≈ 0 and tour<5° 3-4 % in all runs. The turret progresses a bit (deviates less from target, gain ×2 at 50-100 m) but still does not hold: 0.26 s under 5° vs 0.80 s for 522, and still moves away from the target on average. Touch gain s5 → s9 comes mostly from the trigger (firing in passing, more armed time), not a turret that settles. Script `aim.py` (session scratchpad).

# Probe chronology 522 (save_1 → save_63)

Binary `cmake-build-release` of 2026-11-15 20:07 (commit 9d843717 + empty penalty), no inference diff. 4 episodes, 32 tanks, economy 10/2/30, stochastic unless stated. Probe does not respawn and does not apply the penalty. touches/tank-s = hit/shell × shots/s. Saves 1-7, 10, 11: side 550 only.

## Side 550

| save | idx | D | p_fire 0-2° | 5-10° | 45-180° | ratio | shots/s | hit/sh | hit < 5° | shots < 5° | 1st empty p50 | %empty | disarmed | tour<5° | touches/tank-s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 27 k | 300 | 0.634 | 0.632 | 0.630 | 1.0 | 1.59 | 5.4 % | 13 % | 20 % | 2.7 s | 97 % | 57 % | 17 % | 0.086 |
| 3 | 81 k | 400 | 0.684 | 0.682 | 0.663 | 1.0 | 1.24 | 10.3 % | 27 % | 20 % | 2.6 s | 95 % | 61 % | 13 % | 0.127 |
| 5 | 135 k | 450 | 0.543 | 0.487 | 0.310 | 1.8 | 0.90 | 13.5 % | 38 % | 24 % | 3.8 s | 92 % | 55 % | 13 % | 0.122 |
| 7 | 189 k | 550 | 0.318 | 0.182 | 0.096 | 3.4 | 0.83 | 16.8 % | 33 % | 33 % | 8.1 s | 84 % | 28 % | 17 % | 0.140 |
| 9 | 243 k | 650 | 0.094 | 0.056 | 0.031 | 2.5 | 0.45 | 21.8 % | 34 % | 45 % | 25 s | 71 % | 12 % | 25 % | 0.098 |
| 10 | 270 k | 700 | 0.062 | 0.023 | 0.013 | 4.2 | 0.24 | 36.9 % | 52 % | 59 % | 56 s | 34 % | 1 % | 31 % | 0.089 |
| 11 | 297 k | 750 | 0.175 | 0.054 | 0.012 | 7.7 | 0.44 | 31.5 % | 48 % | 53 % | 24 s | 41 % | 1 % | 22 % | 0.139 |
| 16 | 432 k | 1000 | 0.036 | 0.012 | 0.005 | 7.2 | 0.16 | 46.2 % | 56 % | 71 % | 85 s | 20 % | 0.1 % | 45 % | 0.075 |
| 24 | 648 k | 1000 | 0.094 | 0.041 | 0.011 | 15.3 | 0.42 | 36.4 % | 44 % | 69 % | 38 s | 43 % | 0.9 % | 39 % | 0.154 |
| 32 | 864 k | 1000 | 0.154 | 0.039 | 0.019 | 12.4 | 0.43 | 35.2 % | 52 % | 59 % | 32 s | 38 % | 0.6 % | 28 % | 0.152 |
| 32 det | | | 0.121 | 0.040 | 0.024 | 5.6 | 0.19 | 48.0 % | 62 % | 68 % | 31 s | 14 % | 0.1 % | 34 % | 0.090 |

## Side 1000

| save | p_fire 0-2° | 5-10° | 45-180° | ratio | shots/s | hit/sh | hit < 5° | shots < 5° | 1st empty p50 | %empty | disarmed | tour<5° | touches/tank-s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 9 | 0.068 | 0.039 | 0.031 | 2.3 | 0.33 | 17.4 % | 32 % | 36 % | 37 s | 73 % | 11 % | 20 % | 0.057 |
| 16 | 0.023 | 0.005 | 0.002 | 14.1 | 0.08 | 44.1 % | 55 % | 74 % | 101 s | 12 % | 0.1 % | 37 % | 0.035 |
| 24 | 0.080 | 0.017 | 0.005 | 35.8 | 0.19 | 36.1 % | 47 % | 70 % | 45 s | 34 % | 0.7 % | 23 % | 0.069 |
| 32 | 0.075 | 0.010 | 0.004 | 56.8 | 0.13 | 36.6 % | 54 % | 62 % | 51 s | 26 % | 0.4 % | 21 % | 0.046 |
| 32 det | 0.082 | 0.024 | 0.012 | 15.7 | 0.08 | 42.2 % | 57 % | 69 % | 61 s | 15 % | 0.1 % | 28 % | 0.035 |

## References at D 1000 (550 / 1000)

| | shots/s | hit/sh | hit < 5° | ratio | 1st empty p50 | %empty | disarmed | tour<5° | touches/tank-s |
|---|---|---|---|---|---|---|---|---|---|
| 520 save_16 | 0.29 / 0.13 | 30.2 / 23.2 % | 44 / 44 % | 8.4 / 12.8 | 60 / 81 s | 38 / 30 % | 1.4 / 1.2 % | 33 / 28 % | 0.087 / 0.031 |
| 520 save_32 | 1.06 / 0.53 | 26.0 / 27.0 % | 40 / 40 % | 7.5 / 17.2 | 22 / 28 s | 42 / 48 % | 3.1 / 2.3 % | 29 / 25 % | 0.276 / 0.142 |
| 520 save_32 det | 0.81 / 0.36 | 27.9 / 29.5 % | 39 / 41 % | 6.3 / 16.0 | 21 / 36 s | 49 / 36 % | 2.7 / 1.0 % | 33 / 28 % | 0.227 / 0.107 |
| 521 save_16 | 1.06 / 0.50 | 22.2 / 22.3 % | 36 / 38 % | 5.7 / 9.5 | 9 / 20 s | 62 / 69 % | 8.4 / 6.9 % | 26 / 25 % | 0.235 / 0.112 |
| 521 save_26 | 1.25 / 0.82 | 20.6 / 19.1 % | 33 / 31 % | 4.4 / 7.9 | 5 / 9 s | 80 / 85 % | 19 / 15 % | 30 / 25 % | 0.258 / 0.157 |

## Heads (550 / 1000)

| save | σ canon_x | σ canon_y | mean μ dir_x | share \|μ dir_x\| > 1.5 | share \|μ dir_y\| > 1.5 | p_zoom |
|---|---|---|---|---|---|---|
| 9 | 0.26 / 0.28 | 0.29 / 0.31 | −0.46 / −0.47 | 0 | 0 | 0.77 / 0.76 |
| 16 | 0.25 / 0.35 | 0.28 / 0.33 | −0.87 / −0.87 | 0 | 0 | 0.60 / 0.55 |
| 24 | 0.35 / 0.61 | 0.27 / 0.49 | −0.46 / −0.36 | 0 | 25 / 38 % | 0.57 / 0.40 |
| 32 | 0.45 / 0.60 | 0.48 / 0.60 | −1.36 / −1.44 | 45 / 54 % | 0 | 0.16 / 0.09 |
| 520 save_32 | 0.36 / 0.58 | 0.26 / 0.51 | +0.32 / 0.00 | 10 / 6 % | 14 / 10 % | 0.44 / 0.25 |

## Reading

1. save_1-3, taxed dump: trigger flat at 0.63-0.68, reserve empty in 2.7 s, 57-61 % of time disarmed. The penalty is paid without effect.
2. save_5-10, descent: off-axis falls first (ratio 1 → 3.4), then everything falls together (ratio 2.5 at save_9). Trough at save_10 (0.24 shot/s). p_fire independent of shells remaining (1-2 = 6-10): the penalty acts as a tax on firing, not end-of-reserve rationing.
3. save_11, first rebound on axis: ×2.8 at 0-2°, 45-180° at floor.
4. save_16, second trough on arrival at D 1000: 0.16/0.08 shot/s, mute deterministic, but best precision of the run (hit/shell 46/44 %) and best turret (tour<5° 45/37 %).
5. save_24-32, cruise: cadence 0.42-0.43 side 550, 0.13-0.19 side 1000; hit/shell stable 35-37 %; ratio 12-15 / 36-57. Turret regresses to 28/21 % while σ canon rises; dir_x saturates and zoom abandoned at save_32.

## Vs 520 and 521

- hit/shell: +9 to +10 points over the best 520, +15 to +17 over 521 save_26.
- Aligned shots' precision: 52-54 % vs 40 % (520) and 31-33 % (521), at equal turret ⟹ gain comes from the moment of the shot, not aim.
- Touch throughput: 55 % of 520 side 550, 32 % side 1000.
- Deterministic: 522 save_32 fires 4× less than 520 save_32 det, hit/shell 42-48 % vs 28-29.5 %.
- Useful next: save_36-37 (two sides + deterministic): side-1000 cadence, share of aligned states at p > 0.5, dir_x saturation, p_zoom.

---

# Update 2026-11-16 13:40: save_40, 48, 54

Same binary and protocol. save_54 also deterministic.

## Side 550

| save | idx | p_fire 0-2° | 5-10° | 45-180° | ratio | p_off | shots/s | hit/sh | hit < 5° | shots < 5° | 1st empty p50 | %empty | disarmed | tour<5° | touches/tank-s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 864 k | 0.154 | 0.039 | 0.019 | 12.4 | 0.020 | 0.43 | 35.2 % | 52 % | 59 % | 32 s | 38 % | 0.6 % | 28 % | 0.152 |
| 40 | 1 080 k | 0.222 | 0.040 | 0.015 | 15.9 | 0.015 | 0.49 | 36.3 % | 47 % | 69 % | 16 s | 40 % | 1.6 % | 27 % | 0.178 |
| 48 | 1 296 k | 0.162 | 0.040 | 0.016 | 14.7 | 0.020 | 0.57 | 34.9 % | 45 % | 70 % | 25 s | 47 % | 0.8 % | 37 % | 0.199 |
| 54 | 1 458 k | 0.198 | 0.040 | 0.022 | 11.6 | 0.025 | 0.57 | 37.0 % | 49 % | 69 % | 24 s | 39 % | 1.2 % | 34 % | 0.211 |
| 32 det | | 0.121 | 0.040 | 0.024 | 5.6 | 0.027 | 0.19 | 48.0 % | 62 % | 68 % | 31 s | 14 % | 0.1 % | 34 % | 0.090 |
| 54 det | | 0.190 | 0.043 | 0.021 | 10.9 | 0.027 | 0.43 | 40.3 % | 54 % | 64 % | 25 s | 27 % | 0.5 % | 29 % | 0.173 |

## Side 1000

| save | p_fire 0-2° | 5-10° | 45-180° | ratio | p_off | shots/s | hit/sh | hit < 5° | shots < 5° | 1st empty p50 | %empty | disarmed | tour<5° | touches/tank-s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 0.075 | 0.010 | 0.004 | 56.8 | 0.005 | 0.13 | 36.6 % | 54 % | 62 % | 51 s | 26 % | 0.4 % | 21 % | 0.046 |
| 40 | 0.133 | 0.009 | 0.005 | 46.1 | 0.006 | 0.17 | 40.0 % | 55 % | 68 % | 44 s | 27 % | 0.6 % | 19 % | 0.068 |
| 48 | 0.151 | 0.017 | 0.008 | 35.8 | 0.009 | 0.28 | 35.9 % | 47 % | 70 % | 35 s | 43 % | 1.0 % | 25 % | 0.101 |
| 54 | 0.149 | 0.019 | 0.011 | 31.6 | 0.013 | 0.30 | 34.2 % | 47 % | 67 % | 38 s | 34 % | 0.4 % | 26 % | 0.101 |
| 32 det | 0.082 | 0.024 | 0.012 | 15.7 | 0.010 | 0.08 | 42.2 % | 57 % | 69 % | 61 s | 15 % | 0.1 % | 28 % | 0.035 |
| 54 det | 0.130 | 0.023 | 0.010 | 21.0 | 0.011 | 0.24 | 39.8 % | 54 % | 65 % | 32 s | 29 % | 0.1 % | 27 % | 0.095 |

Aligned-and-close counts: 690-830 side 550, 310-370 side 1000 (side-1000 ratio noisy).

## Heads (550 / 1000)

| save | σ canon_x | σ canon_y | mean μ dir_x | share \|μ dir_x\| > 1.5 | share \|μ dir_y\| > 1.5 | p_zoom |
|---|---|---|---|---|---|---|
| 32 | 0.45 / 0.60 | 0.48 / 0.60 | −1.36 / −1.44 | 45 / 54 % | 0 | 0.16 / 0.09 |
| 40 | 0.61 / 0.67 | 0.57 / 0.66 | −0.87 / −0.98 | 7 / 7 % | 0 | 0.29 / 0.18 |
| 48 | 0.22 / 0.62 | 0.20 / 0.59 | −0.72 / −0.92 | 11 / 16 % | 36 / 18 % | 0.59 / 0.40 |
| 54 | 0.23 / 0.48 | 0.22 / 0.46 | −0.10 / −0.07 | 10 / 8 % | 12 / 8 % | 0.51 / 0.34 |

## CSV at same index (±15 k)

| save | hit | kill | fire | ev | skip | α_fire |
|---|---|---|---|---|---|---|
| 32 | 0.414 | 17.7 | 0.27 | 0.69 | 0 | 1.1·10⁻⁴ |
| 40 | 0.359 | 21.7 | 0.28 | 0.50 | 2.3 % | 1.4·10⁻⁴ |
| 48 | 0.364 | 19.9 | 0.24 | 0.60 | 2.4 % | 6.5·10⁻⁴ |
| 54 | 0.365 | 19.4 | 0.58 | 0.53 | 3.6 % | 5.0·10⁻³ |

## Reading per axis (save_32 → save_54)

1. Shot: hit/shell flat 34-40 % on the 8 points; cadence +33 % side 550, ×2.3 side 1000 ⟹ touches/tank-s 0.152 → 0.211 and 0.046 → 0.101, monotone. Median miss 8.5-13 m, 10-15 % of shells fired at ≥ 20° (0 touches).
2. Trigger: side 550 shape unchanged (5-10° frozen at 0.040, 45-180° 0.015-0.022). Side 1000 axis doubles from save_32 to save_48 (0-2° 0.075 → 0.15) then stops, while off-axis continues (45-180° 0.004 → 0.011, p_off ×2.5) ⟹ ratio 57 → 32.
3. Dump: absent. Disarmed 0.4-1.6 %, 27-47 % of tank-episodes empty once, median 1st empty 16-44 s.
4. Aim: tour<5° 27-37 % / 19-26 %, no trend; aligned-shot precision 45-55 %.
5. Heads: dir_x desaturated from save_40, zoom back at save_48; σ canon falls to 0.22 side 550 at save_48-54 (0.45-0.61 before) with no visible effect on tour<5°.
6. Deterministic: cadence ×2.3/×3 from save_32 to save_54, hit/shell 40 %.
Order of events: axis rises first (save_32 → 48, α_fire still under 7·10⁻⁴), off-axis next (save_48 → 54, α_fire ×8). CSV sees only the last step (fire 0.24 → 0.58).
Established: touch throughput up over 4 saves at constant precision; no dump; deterministic usable; dir_x and zoom were transient. Noise: hit/shell and tour<5° between neighbouring saves; absolute side-1000 ratio. Not settled: whether the side-1000 off-axis rise is the onset of regulator release (510/514); a single interval (save_48 → 54) separates it from the axis rise. Next useful: save_58-60 both sides (p_off and 45-180° side 1000, disarmed, touches/tank-s). 520 references: save_32 0.276/0.142, save_37 0.320/0.140 at hit/shell 26 % then less.

---

# Update 2026-11-16 16:15: save_59, 63

Same binary and protocol. save_63 also deterministic.

| save (550 / 1000) | idx | p_fire 0-2° | 5-10° | 45-180° | ratio | p_off | shots/s | hit/sh | hit < 5° | 1st empty p50 | %empty | disarmed | tour<5° | touches/tank-s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 54 | 1 458 k | 0.198 / 0.149 | 0.040 / 0.019 | 0.022 / 0.011 | 11.6 / 31.6 | 0.025 / 0.013 | 0.57 / 0.30 | 37.0 / 34.2 % | 49 / 47 % | 24 / 38 s | 39 / 34 % | 1.2 / 0.4 % | 34 / 26 % | 0.211 / 0.101 |
| 59 | 1 593 k | 0.185 / 0.166 | 0.033 / 0.019 | 0.015 / 0.007 | 15.5 / 31.3 | 0.018 / 0.008 | 0.48 / 0.29 | 38.3 / 38.6 % | 51 / 52 % | 22 / 19 s | 33 / 29 % | 0.8 / 0.4 % | 31 / 25 % | 0.182 / 0.111 |
| 63 | 1 701 k | 0.108 / 0.099 | 0.022 / 0.015 | 0.010 / 0.005 | 16.2 / 34.9 | 0.011 / 0.007 | 0.32 / 0.22 | 42.1 / 41.1 % | 53 / 53 % | 47 / 45 s | 26 / 23 % | 0.4 / 0.7 % | 31 / 23 % | 0.133 / 0.092 |
| 54 det | | 0.190 / 0.130 | 0.043 / 0.023 | 0.021 / 0.010 | 10.9 / 21.0 | 0.027 / 0.011 | 0.43 / 0.24 | 40.3 / 39.8 % | 54 / 54 % | 25 / 32 s | 27 / 29 % | 0.5 / 0.1 % | 29 / 27 % | 0.173 / 0.095 |
| 63 det | | 0.110 / 0.102 | 0.024 / 0.013 | 0.012 / 0.008 | 14.6 / 31.0 | 0.013 / 0.008 | 0.26 / 0.16 | 46.5 / 46.8 % | 57 / 57 % | 45 / 42 s | 17 / 12 % | 0.3 / 0.1 % | 32 / 23 % | 0.121 / 0.076 |

Aligned-and-close counts: 850-1 040 side 550, 510-740 side 1000.
Heads (550 / 1000): σ canon_x 0.37 / 0.50 (save_59), 0.61 / 0.66 (save_63); dir_x not saturated (7-18 %), dir_y saturated 21-28 % at save_63; p_zoom 0.42-0.44 / 0.33.
CSV at same index: save_59 hit 0.39, kill 21.8, fire 0.33, α_fire 1.3·10⁻²; save_63 hit 0.38, kill 20, fire 0.21, α_fire 2.8·10⁻².

Reading:
1. Trigger: side-1000 off-axis falls back (p_off 0.013 → 0.007), then whole curve ÷ ≈ 2 at save_63, shape unchanged (ratio 16 / 31-35 over 3 saves). α_fire ×6 over the interval, no release.
2. Shot: cadence 0.57 → 0.32 and 0.30 → 0.22; hit/shell 37/34 → 42/41 %. Touch throughput recedes (0.211 → 0.133 side 550), less side 1000 (0.101 → 0.092, peak 0.111 at save_59).
3. Dump absent, disarmed ≤ 0.8 %, 23-33 % of tank-episodes empty once.
4. Aim flat (tour<5° 31 % / 23-25 %, aligned-shot hit 51-53 %).
5. Deterministic: save_63 fires at 73-81 % of stochastic, hit/shell 46.5 %, but touch throughput 0.121/0.076 vs 0.173/0.095 at save_54.
Established: the side 32 → 54 off-axis rise was an oscillation; since save_24 aim is flat and only the trigger moves, along a cadence/precision trade-off (max touch throughput at save_54-59, max precision at save_63). Not settled: save_63 transient trough or onset of closing; whether α_fire acts higher up.

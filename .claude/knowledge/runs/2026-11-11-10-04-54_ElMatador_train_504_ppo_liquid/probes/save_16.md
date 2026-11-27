# Probe 504 save_16 — 432 k, D 950

- `build/` binary 10:06 (9789cbb3, SoftClamp); diff vs run code limited to the SAC branch of `probe/main.cpp` ⟹ identical liquid inference, launched with `--force`.
- Sides 550 and 1000, 4 episodes, stochastic, economy 10 / +2 / 20 (config.json).

## Summary, with save_5 and save_9

| save (index) | side | ratio | p_al | p_off | p_arm | hit/shell | empty (s) | disarmed | shots/s | turret<5° | σx p50 | σ@ceil | \|μ\|cx | μsat |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 (135 k) | 550 | 3.5 | 0.53 | 0.15 | 0.20 | 8.4 % | 5.9 | 46 % | 0.86 | 13.8 % | 0.48 | 0 | 0.16 | 0 |
| 9 (243 k) | 550 | 6.1 | 0.19 | 0.031 | 0.054 | **16.1 %** | **22.2** | 16 % | 0.43 | 12.5 % | 0.62 | 0 | 0.30 | 0 |
| 16 (432 k) | 550 | 4.4 | 0.37 | 0.084 | 0.118 | 11.2 % | 9.1 | 29 % | 0.74 | 14.5 % | 0.67 | **27 %** | **0.55** | 4.2 % |
| 5 | 1000 | 5.0 | 0.61 | 0.12 | 0.15 | 3.9 % | 5.7 | 47 % | 0.69 | 12.7 % | 0.50 | 0 | 0.15 | 0 |
| 9 | 1000 | 7.0 | 0.18 | 0.025 | 0.033 | 7.6 % | 41.7 | 15 % | 0.28 | 10.1 % | 0.62 | 0 | 0.30 | 0 |
| 16 | 1000 | 4.7 | 0.29 | 0.062 | 0.078 | 6.1 % | 11.6 | 29 % | 0.52 | 13.0 % | 0.69 | **44 %** | **0.51** | 2.9 % |

## Reading

- **Trigger**: still selective (ratio 4.4-4.7, n aligned&near 1046 / 411), but level doubled since save_9 (p_arm 0.054 → 0.118): p_off rises faster than p_al, ratio falls. Deterministic: trigger stays almost mute (aligned 8 % / 3 %).
- **Dump partly back**: reserve empty 22 → 9 s (side 550), 42 → 12 s (side 1000), 29 % of decisions disarmed (save_9: 15 %). Still far from save_5's dump (6 s, 46 %).
- **Fire**: hit/shell 16 → 11 % (550), 7.6 → 6.1 % (1000). Shells fired <5° still hit 22 % / 12 %; drop comes mostly from off-axis volume (29-34 % of shots at ≥ 20°).
- **Aim**: turret <5° stable (12-15 %) since save_5 — no progress.
- **Heads**: σ_canon_x climbs toward SoftClamp's upper bound (p50 0.67-0.69, 27-44 % ≥ 0.69; save_9: 0 %), |μ_canon_x| 0.30 → 0.51-0.55 with 3-4 % saturated states. Continuous α at floor 0.001 in CSV: σ rise does not come from α.

## Comparison to landmarks

- 500 save_20 (~540 k, D 1000, side 550): hit/shell 22 %, ratio 4.7; 500 save_27 (729 k): 14 %. 502 save_20: 22 %, ratio 16.9; save_22: 8.8 %. 504 save_16 (11 %, ratio 4.4) is below these peaks but at the level of the « degraded cruise » saves of 500-502; single seed, to revisit at save_20.
- 504 reaches D 950 at 429 k, vs 461-566 k for 500-502: real curriculum lead.

## Conclusions

- **Established**: between save_9 and save_16, σ_canon_x reached the SoftClamp upper bound and |μ_canon_x| almost doubled; simultaneously less selective trigger, dump partly back, hit/shell −5 pts. Same canon_x drift as 502 (save_20 → save_22), slower and without massive μ saturation.
- **Undecided**: causality σ ↑ → trigger ↓ (temporal coincidence, as for 503). Whether SoftClamp near ceiling (saturated sigmoid) refreezes σ like the old hard clamp.
- **Next**: probe save_20 (~540 k, D 1000) for the planned test; save_12-13 to date the switch between 243 k and 432 k.

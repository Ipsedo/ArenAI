# Run 529 — probe chronology (side 550)

Run 529 = 528 + canon σ ceiling 0.25 / σ init 0.20 / canon entropy target −1. Stochastic probes, 4 episodes, `build/` binary of 2026-11-17 21:47 without inference diff. Saves 9, 14, 20 probed side 550 only (and 10-13 afterwards). Dwell, sweep (9.2° × |tanh μ|²) and noise recomputed by the same script on all rows.

| save | idx | D | dwell < 5° | passes ≥ 1 s | sweep | noise | \|μ\| cx | tour<5° | tour<20° | p_al | p_off | p_arm | shots/s | hit/sh | h<5° | shots<5° | 1st empty | disarmed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 135 k | 300 | 0.47 s | 10.6 % | 0.9° | 1.1° | 0.27 | 7.1 % | 37 % | 0.58 | 0.194 | 0.233 | 0.88 | 6.6 % | 24 % | 16 % | 6 s | 48 % |
| 9 | 243 k | 500 | **0.53 s** | **14.3 %** | 1.0° | 1.0° | 0.26 | 14.0 % | 49 % | 0.63 | 0.093 | 0.146 | 0.85 | 13.6 % | 30 % | 28 % | 13 s | 26 % |
| 14 | 378 k | 650 | 0.23 s | 1.1 % | 1.9° | 1.2° | 0.41 | 14.5 % | 57 % | 0.38 | 0.023 | 0.054 | 0.49 | 19.2 % | 32 % | 34 % | 39 s | 3.9 % |
| 20 | 540 k | 700 | 0.17 s | 0.2 % | 3.0° | 1.2° | 0.62 | 13.9 % | 60 % | 0.43 | 0.022 | 0.054 | 0.49 | 20.2 % | 34 % | 30 % | 56 s | 3.2 % |
| 27 | 729 k | 650 | 0.17 s | 0.2 % | 3.3° | 1.2° | 0.65 | 10.7 % | 54 % | 0.20 | 0.010 | 0.018 | 0.17 | 23.5 % | 42 % | 19 % | 61 s | 0.9 % |
| 528 s9 | 243 k | 600 | 0.28 s | 4.0 % | 2.3° | 2.6° | 0.45 | 15.6 % | 56 % | 0.64 | 0.056 | — | 0.83 | 17.3 % | 35 % | 33 % | 27 s | 4.6 % |
| 528 s14 | 378 k | 750 | 0.20 s | 0.8 % | 2.7° | 2.4° | 0.52 | 13.9 % | 56 % | 0.34 | 0.016 | — | 0.40 | 24.0 % | 41 % | 36 % | 62 s | 2.4 % |

Median canon σ 0.23-0.25 at all saves (ceiling held).

## Three phases

1. 0-243 k: hold rising. Sweep 0.9-1.0°/decision, dwell 0.47 → 0.53 s, passes ≥ 1 s 14 % (528 s9: 4 %), tour<5° 7 → 14 %, trigger open on target (0.63) and closing off-axis (0.19 → 0.09). save_9 is the run's best hold point, above 528 at the same save. Dump still present (disarmed 26 %).
2. 243-378 k: switch. In the same window: trigger closes (p_off 0.09 → 0.02, p_al 0.63 → 0.38, shots 0.85 → 0.49/s), dump resorbs (disarmed 26 → 4 %), sweep doubles (1.0 → 1.9°) and dwell ÷2.3 (0.53 → 0.23 s). tour<5° does not move (14 %). D goes 500 → 650.
3. 378-729 k: drift. Trigger stable to 540 k then second closing (p_al 0.43 → 0.20, shots 0.49 → 0.17/s); sweep keeps rising (1.9 → 3.0 → 3.3°), dwell at floor 0.17 s, tour<5° 14 → 11 %.

Established: at σ 0.24, hold progresses first (to 243 k) then undoes itself; the σ ceiling has a real effect at the start, does not keep it. The sweep's return coincides with the trigger closing and end of dump (243-378 k), not with the second closing of 540-729 k, which comes when the sweep is already at 3°. tour<5° stays 14 % from 243 to 540 k while dwell is divided by 3: same total time near the target, in much shorter passes.
Not settled: order within the 243-378 k window (does the trigger close before the sweep restarts, or the reverse) and the curriculum's share (D 500 → 650 in the same window).

## Window 243-378 k, save by save (side 550)

| save | idx | D | dwell < 5° | passes ≥ 1 s | sweep | \|μ\| cx | tour<5° | p_al | p_off | p_arm | shots/s | hit/sh | h<5° | 1st empty | disarmed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 9 | 243 k | 500 | 0.53 s | 14.3 % | 1.0° | 0.26 | 14.0 % | 0.63 | 0.093 | 0.146 | 0.85 | 13.6 % | 30 % | 13 s | 26 % |
| 10 | 270 k | 534 | 0.41 s | 8.3 % | 1.1° | 0.27 | 9.4 % | 0.50 | 0.069 | 0.094 | 0.56 | 11.1 % | 28 % | 13 s | 29 % |
| 11 | 297 k | 607 | 0.34 s | 4.4 % | 1.4° | 0.36 | 10.9 % | 0.57 | 0.033 | 0.068 | 0.57 | 18.5 % | 39 % | 26 s | 9.2 % |
| 12 | 324 k | 650 | 0.22 s | 1.0 % | 1.9° | 0.44 | 9.7 % | 0.34 | 0.021 | 0.049 | 0.45 | 22.4 % | 40 % | 51 s | 4.0 % |
| 13 | 351 k | 650 | 0.25 s | 1.4 % | 1.6° | 0.35 | 8.7 % | 0.50 | 0.030 | 0.066 | 0.56 | 17.2 % | 30 % | 25 s | 9.6 % |
| 14 | 378 k | 650 | 0.23 s | 1.1 % | 1.9° | 0.41 | 14.5 % | 0.38 | 0.023 | 0.054 | 0.49 | 19.2 % | 32 % | 39 s | 3.9 % |

- The trigger moves first, by one save. At save_10 the cadence has already fallen (0.85 → 0.56/s, p_arm 0.146 → 0.094) while the sweep is unchanged (1.0 → 1.1°, |μ| 0.26 → 0.27). Dwell already drops (0.53 → 0.41 s) without the command having changed.
- The sweep starts at save_11-12 (1.4° then 1.9°), together with off-axis closing (p_off 0.069 → 0.033 → 0.021) and end of dump (disarmed 29 → 9 → 4 %).
- The switch is complete at 324 k, on arrival at D 650; from save_12 to save_14 nothing moves beyond noise (save_13 rises a little everywhere).
- One probe = 4 episodes; the one-save gap between trigger and sweep is of the order of noise between neighbouring saves (save_12 / save_13). The trigger's lead is indicated, not established.

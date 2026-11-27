# Probes 513 save_16 / save_17 (+ deterministic) and pitch diagnosis

Binary b3092240, no inference diff. Sides 550 / 1000, 4 episodes, economy 5/30.

| save (index, D) | mode | ratio | p_arm | hit/sh | shots/s | empty_s | %disarmed | turret<5° | hits/min alive |
|---|---|---|---|---|---|---|---|---|---|
| s14 (378 k, 900) | stoch | 6.1 / 10.0 | 0.16 / 0.08 | 19.4 / 16.4 % | 1.27 / 0.71 | 21.7 / 27.3 | 7.6 / 7.3 % | 25.8 / 22.1 % | 16.2 / 7.7 |
| s16 (432 k, 1000) | stoch | 7.5 / 9.3 | 0.09 / 0.06 | 19.0 / 15.4 % | 0.76 / 0.52 | 24.8 / 29.6 | 6.8 / 9.5 % | 22.7 / 22.0 % | n/a |
| s17 (459 k, 1000) | stoch | 4.0 / 7.0 | 0.34 / 0.16 | 12.0 / 10.4 % | 2.02 / 1.22 | 5.7 / 7.1 | 18.6 / 14.1 % | 25.3 / 21.5 % | 15.9 / 8.2 |
| s18 (486 k, 1000) | stoch | 7.3 / 15.4 | 0.14 / 0.06 | 17.6 / 15.1 % | 1.09 / 0.55 | 9.2 / 23.2 | 12.5 / 8.7 % | 22.8 / 20.9 % | 12.7 / 5.4 |
| s17 | deterministic | 4.9 / 5.9 | 0.26 / 0.14 | 13.6 / 13.0 % | 1.60 / 0.75 | 7.7 / 11.4 | 10.8 / 6.5 % | 22.0 / 23.4 % | n/a |

## Reading

- **Trigger relapse between save_16 and save_17, in 27 k steps**: p_arm ×2-4, cadence ×2.5, 1st empty reserve from 25-30 s to 6-7 s. The relapse also shows in deterministic, so it is the policy not the draw. Lethality (hits/min) does not move: agent fires more for the same number of hits.
- CSV: cadence oscillates since 360 k (0.7 ↔ 1.7 /s). Joint kl peaks at 0.13-0.14 around 360 k and 450-480 k, but comes from the canon group (skip_canon 0.26-0.34). The « rest » group, which contains the trigger, is never stopped (skip 0).
- **Pitch**: on shots at <5° toward the nearest target, between 50 and 200 m, median error is about 1.3-1.9° in yaw as in pitch (p90 3-4°), in both 513 and 510. Ballistic drop (0.2-0.9°) is small vs this dispersion: no range-specific pitch defect. In deterministic (μ alone, no σ), save_17's dispersion is identical (1.6-1.9°): it is in μ, not in exploration noise. Consistent with the memory note on aim diagnosis « μ = white noise ».

## save_18

save_17's relapse was transient. save_18 recovers a trigger close to save_16's, with the best ratio measured at 1000 (15.4). Reserve at 550 stays midway (9.2 s vs 25 s), and hits/min are at their lowest (12.7/5.4). μ starts to saturate (2-4 % of states with |μ| > 1.5). Regime oscillating: save_14 good, save_16 good, save_17 trough, save_18 good. An isolated save no longer suffices to judge: read trend over 3-4 saves.

## Next

- save_20-22: does the oscillation damp? Watch μsat.

# Probe 529 save_5: sweep gone, dwell doubled, criterion not reached

Run 529 (ppo_liquid, 528 + canon σ ceiling 0.25 / σ init 0.20 / canon entropy target −1, tree uncommitted on 68c41e20). save_5 = 135 k, D 300 (528 s5: D 400; 529 has ~40 k CSV delay). Binary `build/arenai_agent/arenai_agent_probe` of 2026-11-17 21:47, no inference diff. Sides 550 and 1000, 4 episodes, stochastic, run economy (10 / 5 / 30).

| | tour<5° | tour<20° | dwell < 5° | passes ≥ 1 s | sweep | bias | σ canon | \|μ\| cx | ratio | p_al | p_off | hit/sh | h<5° | shots<5° | shots/s | 1st empty | disarmed | n aligned & close |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **529 s5 @550** | 7.1 % | 37.2 % | **0.47 s** | 10.6 % | **0.9°** | +0.8° | 0.24 | 0.27 | 3.0 | 0.58 | 0.194 | 6.6 % | 24 % | 16 % | 0.88 | 6.0 s | 48 % | 339 |
| 528 s5 @550 | 11.4 % | 50.7 % | 0.23 s | 2.1 % | 2.6° | +2.5° | 0.51-0.59 | 0.52 | 4.7 | 0.58 | 0.124 | 11.7 % | 28 % | 20 % | 1.01 | 12.7 s | 24 % | 683 |
| **529 s5 @1000** | 3.6 % | 26.3 % | **0.41 s** | 8.7 % | **0.9°** | +0.9° | 0.24 | 0.27 | 3.2 | 0.59 | 0.181 | 2.7 % | 11 % | 10 % | 0.66 | 7.1 s | 58 % | 121 |
| 528 s5 @1000 | 10.9 % | 47.0 % | 0.21 s | 2.1 % | 2.9° | +3.9° | 0.56-0.63 | 0.59 | 7.2 | 0.64 | 0.090 | 6.8 % | 21 % | 19 % | 0.62 | 14.1 s | 34 % | 345 |

Dwell = mean duration of passes (yaw toward nearest enemy < 5°, enemy < 100 m, armed on entry), recomputed identically on both runs (528 s5: 0.23 s here, 0.22 s in analysis_aim_lever). Sweep = mean of 9.2° × |tanh μ|², bias = mean of signed 9.2° × μ|μ|.

- σ canon: ceiling holds. p50 0.24, p90 0.247 on both axes, μ not saturated.
- Sweep: gone. Mean command 0.9°/decision vs 2.6-2.9° (528 s5) and 3-4° later. A slow rotation bias remains always in the same direction (+0.8°/decision ≈ 8°/s).
- Dwell ×2 (0.47/0.41 s vs 0.23/0.21 s), passes ≥ 1 s 9-11 % vs 2 %. Stays under threshold 0.6 s and under the immobile turret at σ 0.25 in simulation (~1 s): the command does not follow the target, it traverses it more slowly.
- The turret finds targets less. tour<20° 37 % vs 51 % (26 vs 47 side 1000), tour<5° 7.1 % vs 11.4 %, 583 passes vs 2 015. Fewer, longer passes: total time on target falls (0.47 × 583 = 274 s vs 0.23 × 2 015 = 463 s).
- Trigger: same aligned level (0.58), more open off-axis (0.19 vs 0.12), ratio 3.0 vs 4.7; aligned & close counts low (339/121), ratio noisy.
- Dump: back to 525's level. 1st empty 6-7 s vs 12.7-14.1 s, disarmed 48/58 % vs 24/34 %, 94-98 % of tank-episodes empty.
- Precision down: hit/shell 6.6/2.7 % vs 11.7/6.8 %; aligned shots 24/11 % vs 28/21 %; 43-54 % of shots leave at ≥ 20°.
Criterion set before the run (side 550): dwell > 0.6 s **and** tour<5° > 20 %: **not reached** (0.47 s, 7.1 %). The planned refutation case (dwell 0.2-0.3 s with sweep 3-4°/decision) did not occur either: the sweep stopped and dwell doubled.
Established: capping σ removes the sweep and lengthens dwell as hypothesised; at 135 k this translates neither into aim (tour<5° lower) nor touches (hit/shell ÷1.8 to 2.5), and target search is degraded. Confounded: the ~40 k CSV delay (529 s5 ≈ 528 near 95 k in kill/hit); part of the precision deficit and dump may be delay, not ceiling effect. Not settled: whether hold is learned afterwards (in 520/522 hold was that of a nearly immobile turret; here the turret still rotates slowly).
Markers for save_9 (243 k): side-550 dwell > 0.6 s and rising, or stable 0.4-0.5 s; tour<20° back toward 50 % (search redone without noise) or staying 35-40 %; rotation bias cancels or persists; disarmed and 1st empty vs 528 s9 (4.6 %) at equal D.

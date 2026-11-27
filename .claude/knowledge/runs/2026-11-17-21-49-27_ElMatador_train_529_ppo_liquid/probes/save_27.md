# Probe 529 save_27: the sweep returned through μ, at σ 0.24

Run 529 (ppo_liquid, 528 + canon σ ceiling 0.25 / σ init 0.20 / canon entropy target −1, tree uncommitted on 68c41e20). save_27 = 729 k, D 650 (down from 700). Binary `build/arenai_agent/arenai_agent_probe` of 2026-11-17 21:47, no inference diff. Sides 550 and 1000, 4 episodes, stochastic, run economy (10 / 5 / 30).

| | tour<5° | tour<20° | dwell < 5° | passes ≥ 1 s | sweep | bias | noise | σ canon | \|μ\| cx | ratio | p_al | p_off | p_arm | hit/sh | h<5° | shots<5° | shots/s | 1st empty | disarmed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **529 s27 @550** | 10.7 % | 53.9 % | **0.17 s** | 0.2 % | **3.3°** | +0.2° | 1.2° | 0.24 | 0.65 | 20 | 0.20 | 0.010 | 0.018 | 23.5 % | 42 % | 19 % | 0.17 | 61 s | 0.9 % |
| 529 s5 @550 | 7.1 % | 37.2 % | 0.47 s | 10.6 % | 0.9° | +0.8° | 1.1° | 0.24 | 0.27 | 3.0 | 0.58 | 0.194 | 0.233 | 6.6 % | 24 % | 16 % | 0.88 | 6.0 s | 48 % |
| 528 s14 @550 | 13.9 % | 56.3 % | 0.20 s | 0.8 % | 2.7° | +2.3° | 2.4° | 0.44-0.53 | 0.52 | 21 | 0.34 | 0.016 | — | 24.0 % | 41 % | 36 % | 0.40 | 62 s | 2.4 % |
| **529 s27 @1000** | 8.5 % | 44.3 % | **0.17 s** | 0.2 % | **3.1°** | +0.7° | 1.2° | 0.25 | 0.62 | 64 | 0.24 | 0.004 | 0.006 | 20.1 % | 40 % | 12 % | 0.06 | 54 s | 0.6 % |
| 529 s5 @1000 | 3.6 % | 26.3 % | 0.41 s | 8.7 % | 0.9° | +0.9° | 1.1° | 0.24 | 0.27 | 3.2 | 0.59 | 0.181 | 0.200 | 2.7 % | 11 % | 10 % | 0.66 | 7.1 s | 58 % |
| 528 s14 @1000 | 12.1 % | 54.3 % | 0.19 s | 0.6 % | 2.6° | +2.7° | 2.6° | 0.48-0.56 | 0.51 | 39 | 0.39 | 0.010 | — | 19.6 % | 36 % | 28 % | 0.19 | 83 s | 1.6 % |

Dwell, sweep (mean of 9.2° × |tanh μ|²), bias (signed mean) and noise (std of the action around the command, °/decision) recomputed identically on the three probes.

- σ canon: still at the ceiling (p50 0.24, p90 0.249), exploration noise 1.2°/decision, half of 528's.
- The sweep returned, carried by the command. |μ| canon_x 0.27 → 0.65, sweep 0.9 → 3.3°/decision, above 528 s14 (2.7°). Bias ≈ 0: back-and-forth, no more rotation in one direction. 4-5 % of canon μ saturated.
- Dwell: 0.47 → 0.17 s, under 528 s14 (0.20 s) and under the untrained network (0.37 s); 0.2 % of passes last ≥ 1 s. 5 461 passes vs 583 at s5: many very short passes.
- Turret: search is back (tour<20° 37 → 54 %), tour<5° 10.7 %, 528 s14's level (13.9 %), far from 520/522 (33-45 %).
- Trigger: almost closed. Mean armed p_fire 0.018 (0.006 side 1000), aligned & close 0.20 vs 0.34 for 528 s14; aligned < 5° all distances 0.030. Shots 0.17/s and 0.06/s. Deterministic 2.4 % of aligned & close states fire.
- Precision per shell: 528 s14's (hit/shell 23.5/20.1 %, aligned shots 42/40 %) but only 19/12 % of shots leave aligned (528 s14: 36/28 %).
- Dump: resorbed (disarmed < 1 %, 9-16 % of tank-episodes empty).
Criterion of analysis_aim_lever: refutation case (« dwell 0.2-0.3 s with a sweep of 3-4°/decision = the bottleneck is the sweep itself »): **reached at save_27**, with σ at 0.24. The ceiling removed the sweep at the start (s5); the policy relearned it through μ.
Established: at canon σ 0.24 held all run, the turret ends up sweeping faster than in 528 and holds the target less; the trigger closes to the same final state (rare, precise per shell, few aligned shots). σ high is not the cause of the sweep nor of non-holding. Not settled: when the sweep returned (saves 9-26 not probed) and whether it precedes or follows the trigger closing (CSV fire falling from 300 k). Useful probes: saves 9, 14, 20 side 550 (dwell, sweep, p_al) to date the sweep's return relative to the fall of `fire`.

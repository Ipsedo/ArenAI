# Probe 525 save_12: trigger closes, turret still sweeps

Run 525 (pure events, single advantage, flat CNN, target_kl 0.10; tree uncommitted on 68c41e20). save_12 = 324 k, D 750. Binary of 2026-11-17 08:32, no inference diff. Sides 550 and 1000, 4 episodes, stochastic; side 550 also replayed deterministic.

| | tour<5° | tour<20° | ratio | p_al | p_off | hit/sh | h<5° | shots<5° | shots/s | touches/tank-s | 1st empty | disarmed | σ canon | \|μ\| cx | mean μ cx |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 525 s12 @550 | 6.4 % | 41 % | 21 | 0.12 | 0.006 | 33.6 % | 50 % | 13 % | 0.118 | 0.040 | 84 s | 1.5 % | 0.59-0.62 | 0.82 | +0.35 |
| 525 s12 @1000 | 5.8 % | — | 50 | 0.14 | 0.003 | 22.7 % | 45 % | 15 % | 0.052 | 0.012 | 103 s | 0.7 % | 0.61-0.63 | 0.83 | — |
| 525 s12 @550 det | 6.7 % | 41 % | 16 | 0.11 | 0.007 | 39.4 % | — | — | 0.069 | 0.027 | 98 s | 0.6 % | — | 0.84 | +0.32 |
| 525 s9 @550 | 6.6 % | 38 % | 42 | 0.43 | 0.010 | 31.3 % | 50 % | 23 % | 0.242 | 0.076 | 43 s | 3 % | 0.58-0.59 | 0.49 | +0.33 |
| 525 s9 @1000 | 4.1 % | 28 % | 92 | 0.47 | 0.005 | 26.9 % | 48 % | 21 % | 0.112 | 0.030 | 77 s | 2.5 % | 0.59-0.60 | 0.50 | +0.41 |
| 524 s16 @550 | 10.1 % | — | 31 | 0.37 | 0.012 | 28.0 % | 45 % | 30 % | 0.26 | 0.072 | — | 4 % | 0.51-0.56 | 0.43 | — |
| control 430 s16 @550 | 40.3 % | — | 5.7 | 0.06 | — | 47.3 % | — | 67 % | — | 0.093 | — | 0.1 % | 0.30-0.42 | 0.20 | 0.00 |

CSV at 320-340 k (D 780-800): hit 0.27-0.28, kill 7.9-8.6, fire 0.15-0.18 (524 same index: 0.26-0.28 / 8-11 / 0.15-0.18; 522: 0.31-0.35 / 15-17).

- Trigger: closes. Aligned & close p_fire 0.43 → 0.12 (550) and 0.47 → 0.14 (1000), cadence ÷2, empty at 84-103 s. Off-axis stays at floor. Deterministic 1 % of aligned states fire.
- Aim: unchanged, 6 %. Deterministic ≡ stochastic (6.7/6.4 %): exploration noise is not what prevents target holding, it is the mean command.
- Turret command: median |μ| canon_x 0.49 → 0.82, saturated 17 % of time, mean +0.35: all-or-nothing command biased to one side. dir_x mean −0.59.
- Touch throughput: 0.076 → 0.040 (550) and 0.030 → 0.012 (1000), under 524 s16 (0.072/0.020) side 1000 and far from control (0.093/0.060).

Dwell on target (side 550, see analysis.md):

| | entries < 5° per tank-minute | mean dwell | dwells ≥ 1 s |
|---|---|---|---|
| 525 s5 / s9 / s12 | 25 / 16 / 19 | 0.28 / 0.25 / 0.20 s | 3 / 2 / 0 % |
| 525 s12 deterministic | 20 | 0.20 s | 0 % |
| 524 s16 | 25 | 0.24 s | 1 % |
| control s9 / s16 | 21 / 24 | 0.97 / 0.99 s | 23 / 24 % |
| 522 s24 | 28 | 0.84 s | 20 % |

Conclusion. Established: 525 follows 524's trajectory (kill 8 at D 800, CSV hit 0.27) and departs downward at save_12 (trigger closing, touch throughput ÷2). The turret passes over targets as often as the control's but does not stay. Pure events with single advantage does not make target holding learned in 324 k; Φ in the reward changed nothing (524 ≡ 525). Not settled: D 1000 not reached; no sign of reversal over 3 saves.

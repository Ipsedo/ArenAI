# Run 529 — ppo_liquid, canon σ ceiling 0.25

Launched 2026-11-17 21:49, running; read at 85 k (1.14 h, 74 k steps/h, 2.38 M tank-steps/h). Code: uncommitted tree on 68c41e20 (binary `cmake-build-release` of 2026-11-17 21:49).

## What the run tests

528 + option 1 of `train_528_ppo_liquid/levier_visee.md` (analysis_aim_lever): σ ceiling 0.25 on canon_x/canon_y, initial canon σ 0.20, canon entropy target −1 (`continuous_target_entropy` [0, 0, −1, −1]). The three are confounded. Everything else identical to 528 (economy 5 shells/touch, kills/tank curriculum 0.3 / 0.5, target_kl 0.10).

## CSV (D 250 throughout), means per 10 k slice, kill / hit

| index | 529 | 528 | 526 | 525 | 524 |
|---|---|---|---|---|---|
| 10 k | 9.4 / 0.045 | 8.5 / 0.051 | 8.7 / 0.053 | 2.9 / 0.033 | 6.6 / 0.041 |
| 20 k | 9.7 / 0.049 | 10.4 / 0.066 | 12.3 / 0.072 | 5.1 / 0.052 | 8.5 / 0.054 |
| 30 k | 9.5 / 0.049 | 13.5 / 0.081 | 13.0 / 0.075 | 7.6 / 0.065 | 11.3 / 0.073 |
| 40 k | 8.8 / 0.050 | 16.1 / 0.097 | 15.1 / 0.082 | 10.5 / 0.083 | 13.5 / 0.095 |
| 50 k | 8.6 / 0.057 | 18.6 / 0.117 | 17.5 / 0.092 | 12.5 / 0.099 | 16.0 / 0.122 |
| 60 k | 10.2 / 0.071 | 18.4 / 0.126 | 19.0 / 0.100 | 13.3 / 0.111 | 16.1 / 0.133 |
| 70 k | 12.0 / 0.087 | 18.0 / 0.130 | 19.6 / 0.104 | 12.6 / 0.115 | 16.7 / 0.137 |
| 80 k | 14.3 / 0.100 | 17.9 / 0.131 | 20.3 / 0.104 | 12.8 / 0.116 | 17.3 / 0.137 |
| 90 k | 15.8 / 0.107 | 18.4 / 0.136 | 21.4 / 0.108 | 12.9 / 0.118 | 19.5 / 0.141 |

- 0-50 k: flat (kill ~9, hit 0.05, r < 0). The other runs climb from 10 k.
- 50-85 k: climb at 528's slope (kill +1.7 / 10 k, hit 0.057 → 0.107), no flattening on the last row (kill 16.1, hit 0.108, r 1.8e-3).
- Delay ≈ 40-45 k: 529 at 85 k ≡ 528 at 40 k ≡ 524 at 42 k.
- D still 250 at 85 k; 524/525/526/528 pass D 300 at 47.6 k and D 350 at 67.4 k. The kills/tank criterion (0.5 = 16 kills) was just reached: D 300 expected within ~30 k.
- fire 0.73-0.95 then 1.0-1.2/s (528: 1.1-1.3); kl 0.015 → 0.058, skip 0 → 0.09 (≡ 528 at equal index); ev 0.55-0.67 except a trough at 29 k (−3.8).
- Hc 0.09 (45 k) → −0.10 (85 k): expected, two dims of four capped at σ 0.25. All α at floor 1e-4 from 40 k.

## Diagnosis

Established: at the CSV the start is slower than 524/526/528, not better; only the first slice (10 k) is the highest of the series, compatible with initial canon σ 0.20 dispersing less the shots of an untrained network.
Noise or not: a 50 k plateau at D 250 exists in none of the four previous runs (same curriculum seed), so not a simple seed gap; cause not established (reduced turret exploration = hypothesis).
Not settled: target holding, which the CSV does not measure. `hit` per shot 0.107 stays under 528 at D 250 (0.117 at 50 k).
Markers: refutation criterion set before the run: probe save_5 (~135 k), dwell < 5° > 0.6 s and tour<5° > 20 %; otherwise the bottleneck is the sweep, not σ. Actual canon σ and mean command in °/decision at probe (expected σ ≤ 0.25, sweep < 3-4°/decision). D milestones: 529 lost synchronisation with other runs, compare at equal D.

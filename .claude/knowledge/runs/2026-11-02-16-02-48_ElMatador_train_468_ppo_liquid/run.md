# train_468_ppo_liquid

| | |
|---|---|
| machine | ElMatador |
| user | samuel |
| launch | 2026-11-02 16:02:48 |
| commit at launch | `ec0cd93e` — * add probe main => CNN is working (targeting enemies) * renew aim reward * remove fire cost |
| commits ≤ 90 min after | none |
| diff at launch | unknown (identity recorded after the run ended) |
| raw data | `outputs/train_468_ppo_liquid` on ElMatador |

## Established facts


- 2026-11-02: launched 16:02, commit `ec0cd93e` = 467's code with two changes: aim tolerance 10 m + 6° → 2.5 m + 3°, and fire_cost 0.01 removed (jolt_enemy_tank.cpp tol = 2.5 m + 3°·d; fire cost and `nb_uncharged_shells` deleted; model tests aligned: empty-reserve death −0.5, shot without cost = 0). config.json identical to 467 (checked by diff).
- 2026-11-02: at 235 k steps (3 h 20) vs 467 at equal index: miss (mean shell/enemy distance) unchanged ~21 m in both runs; with the new tolerance (≈ 5 m at 50 m firing range) the exponential equals exp(-4) ≈ 0.02: r_aim 0.0005 vs 0.0026 at D=250 — the aim signal is extinguished, not tightened. hit 5.8 % flat (467: 7.5 % at D=250, 11 % at D=350-400); kill 0.2 (467: 0.9 → 2.9); fire 0.6/s (467: 0.9-1.45); total r −0.002/−0.003 (467: −0.001). D: 250 until 97 k, 300 since, never passed; 467 was at 450 at 175 k and 500 at 207 k. Hc 0.195 → 0.10, kl 0.008 → 0.013, skip 0: no KL regime, the reward has nothing left to say.
- 2026-11-02: the 10 m + 6° floor had been set 2026-10-29 precisely because 2° = tail of the exponential (see leverage-per-episode note); the 467 probe led to tightening it so the shot premium disappears, but at 21 m miss the slope is zero. Do not read 468 as a 300 plateau; tolerance must be of the order of current miss (≈ 20 m) to carry a gradient, then tighten (annealing). Any comparison to 467 must be at equal D.
- 2026-11-02 evening: Samuel chose **Cauchy** 1/(1+(miss/tol)²) with the same tolerance 2.5 m + 3° (implemented in compute_aim_reward, model tests OK). At d=50 m, miss 21 m: value 0.056 (467: 0.25, 468: 0.017), slope ×1.5 vs 468; proposal exp(-miss/log10(1+d)) discarded (tol ≈ 2 m constant, even tighter). Next run (469) to be compared to 467 and 468 at equal D.

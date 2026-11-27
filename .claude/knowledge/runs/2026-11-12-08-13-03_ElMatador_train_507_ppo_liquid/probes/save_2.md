# Probe 507 save_2 (54 k, D 350) vs 504 save_2

`build/` binary of 12/11 08:01, code 25a75a34 (no newer source). For 504, the only inference diff is the addition of SAC classes, no effect on the liquid actor (`--force`). 4 episodes per side, stochastic.

| run | side | p_arm | ratio | hit/sh | h<5° | empty_s | shots/s | turret<5° | σ cx/cy | \|μ\|cx | corr(yaw, μ_cx) within ±40° |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 507 | 550 | 0.55 | 1.13 | 5.0 % | 14.7 % | 3.0 | 0.88 | 9.0 % | 0.28/0.29 | 0.11 | −0.37 |
| 504 | 550 | 0.79 | 1.03 | 3.8 % | 11.0 % | 2.4 | 1.19 | 10.0 % | 0.28/0.28 | 0.10 | −0.31 |
| 507 | 1000 | 0.54 | 1.12 | 2.2 % | 7.7 % | 2.9 | 0.78 | 6.0 % | 0.29/0.30 | 0.12 | −0.22 |
| 504 | 1000 | 0.79 | 1.04 | 1.5 % | 4.9 % | 2.3 | 1.18 | 9.9 % | 0.28/0.28 | 0.09 | −0.19 |

- hit/shell ×1.3-1.5 and hits of aligned shots +35-55 %, but turret not better placed (turret<5° 9 / 6 % vs 10 %) and its target tracking barely stronger (−0.37 vs −0.31, within noise).
- Trigger dumps less: p_arm 0.55 vs 0.79, 0.8-0.9 shots/s vs 1.2, reserve empty at 3.0 s vs 2.4. No selectivity yet (ratio 1.1).
- CSV lead (hit 11 % vs 8 %, kills ×1.7) thus comes mostly from better time-placed shots (less dump), not a better aligned turret. Out of field, nothing tracks the target (corr ≈ 0) in both runs.
- Turret verdict: save_5 (135 k) and save_9 (243 k) vs 504 (turret<5° 13.8 % and 12.5 %).

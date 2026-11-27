# Run 524 — ppo_liquid, single advantage + flat CNN, target_kl 0.10

Launched 2026-11-16 22:54; analysis 2026-11-17 07:00, index 608 k (8.1 h), D 1000 since 461 k. Throughput 75 k steps/h (2.4 M tank-steps/h), like 523 (76 k), under 522 (87 k).
**Effective code: 5c338159 + `target_kl` 0.10 uncommitted at launch (= f4702f90).** Training binary dated 22:54:41; f4702f90 (23:07) only changes the `target_kl` default, already in `config.json`. 68c41e20 (23:28, `ground_height` with second ray) postdates the binary: **not** in the run. The script's automatic estimate (68c41e20) is wrong.

## What the run tests

Sole change vs 523: `target_kl` 0.05 → 0.10 (stop threshold 0.15 on joint KL). Vs 522, three changes remain confounded: flat CNN (cfb76a2e), single advantage (5c338159), single KL stop at 0.10. Reward, economy (10/2/30), empty penalty −D/1000 unchanged since 522.

## CSV

D milestones: 520/522's shifted by exactly one step (300 @ 48 k instead of 21 k, then same indices: 1000 @ 461 k instead of 429 k). Only the first step missed.
At equal index (window means; vid/s = reserve emptyings per tank-second, reward residual):

| index | run | D | hit | kill | fire | r_hit | vid/s | miss | ev | skip | α_fire |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 100-130 k | 520 | 400-450 | 0.13 | 24.3 | 1.68 | 9.8e-3 | — | 40 | 0.44 | 0 | 1e-4 |
| | 522 | 400-450 | 0.15 | 22.7 | 1.25 | 9.1e-3 | 0.095 | 37 | 0.36 | 0 | 1e-4 |
| | 524 | 350-400 | 0.15 | 19.9 | 1.04 | 7.6e-3 | 0.078 | 31 | 0.60 | 0.25 | 1e-4 |
| 150-180 k | 522 | 500 | 0.15 | 22.8 | 1.13 | 8.3e-3 | 0.078 | 45 | 0.49 | 0 | 1e-4 |
| | 524 | 450 | 0.16 | 23.2 | 0.84 | 7.5e-3 | 0.072 | 38 | 0.64 | 0.33 | 1e-4 |
| 280-320 k | 522 | 700-800 | 0.31 | 17.0 | 0.35 | 3.6e-3 | 0.010 | 40 | 0.72 | 0 | 1e-4 |
| | 524 | 650-750 | 0.24 | 13.4 | 0.33 | 3.3e-3 | 0.015 | 36 | 0.74 | 0.20 | 1.2e-4 |
| 380-420 k | 522 | 900-950 | 0.36 | 14.8 | 0.21 | 2.8e-3 | 0.004 | 40 | 0.78 | 0 | 1.2e-4 |
| | 524 | 850-900 | 0.25 | 9.5 | 0.24 | 2.0e-3 | 0.007 | 34 | 0.69 | 0.24 | 3.8e-3 |
| 470-520 k | 520 | 1000 | 0.23 | 15.8 | 0.32 | 3.1e-3 | — | 52 | 0.76 | 0 | 1e-4 |
| | 522 | 1000 | 0.35 | 16.9 | 0.24 | 3.4e-3 | 0.007 | 43 | 0.74 | 0 | 1.3e-4 |
| | 524 | 1000 | 0.19 | 8.7 | 0.26 | 2.0e-3 | 0.013 | 46 | 0.77 | 0.23 | 1.9e-2 |
| 575-612 k | 520 | 1000 | 0.24 | 18.5 | 0.51 | 5.1e-3 | — | 50 | 0.69 | 0 | 1e-4 |
| | 522 | 1000 | 0.39 | 17.0 | 0.16 | 3.9e-3 | 0.006 | 39 | 0.77 | 0 | 1.3e-4 |
| | 524 | 1000 | 0.18 | 8.5 | 0.18 | 1.8e-3 | 0.011 | 47 | 0.78 | 0.26 | 2.4e-2 |

Phases of 524:
1. 0-180 k (D 250-450): at 522's level. kill peak 23.2 at 150-180 k, hit 0.16, ev 0.60-0.64 (522: 0.36-0.49). Hc below 0 from 70 k (min −0.52 at 133 k).
2. 180-340 k (D 500-800): cadence falls as in 522, precision less. fire 0.84 → 0.18, hit 0.16 → 0.26 (522: 0.31-0.36), kill 23 → 8 (522: 17), r_hit 7.5e-3 → 1.7e-3.
3. 320-460 k: α leave the floor. α_canon_x first (1e-4 → 3e-3 at 340-360 k), then α_fire (5e-4 at 340 k → 2.9e-2 at 460 k, ×290), α_canon_y (6.6e-3 at 440 k). In 522 α_fire took off only at 1 040 k.
4. 460 k → (D 1000): relaxation. hit 0.255 → 0.17-0.20, emptyings 0.006 → 0.010-0.013/s (one per tank every 80-100 s, twice 522), miss 39 → 46-49 m, kill 6.3-9.9, r ≈ 0 (5e-5 last slice). No recovery in 150 k at D 1000.

## Diagnosis

Established:
- 524 under 520 and 522 at equal D since D 500: at D 1000 kill ÷2 (8 vs 16-18), r_hit ÷2, hit 0.18 vs 0.35-0.40 (522) and 0.23-0.24 (520). Gap beyond seed noise (reserve ×2 between copies, milestone ±30 k) and lasting 280 k.
- KL regime persists at target_kl 0.10: skip 0.20-0.34 from 100 k to end, mean kl 0.05; 520 and 522 were at skip 0. A quarter of actor minibatches cut.
- The trigger entropy regulator acts: α_fire ×290 in 120 k, and from 460 k hit falls while emptyings double. In 522 the same α_fire (3e-2) had no hold; here starting precision is lower.
- Critic not at fault: ev 0.70-0.82, level of 520/522.
Not settled by CSV: which of flat CNN / single advantage / single KL stop carries the gap (confounded; the other PC's run, cfb76a2e alone, separates the CNN); whether the kill drop comes from aim, trigger or cadence: see probes/timeline.md.
Options/markers: probes at saves 5, 9, 16, 22 done. Stop/continue markers at D 1000: kill > 12 and hit > 0.25 held over 60 k = 520-type recovery (trough then rise on axis); α_fire > 3e-2 with emptyings > 0.015/s = installed relaxation. Compare to other PC's run at same index when its CSV is available.

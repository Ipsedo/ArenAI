# Probe 510 save_3 (81 k, D 400) vs 509 and 508 save_3

Run 510 = 509 + per-group KL stop (GroupKlStop, commit 7422d098), identical config (p_init fire 0.5). `build/` binary of 12/11 23:40, commit 1e171bbf: only diff = trainers (KL stop, training only) and the sigma_gradients tool ⟹ no inference effect, launched in `--force`. 4 episodes per side, stochastic, 32 tanks. 509 save_3: same index, same D, same binary.

| run | side | p_arm | ratio | hit/sh | h<5° | shots<5° | empty_s | %disarmed | shots/s | turret<5° | σ cx/cy | \|μ\|cx |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 509 s3 | 550 | 0.77 | 1.02 | 8.1 % | 17.8 % | 25 % | 2.4 | 64 % | 1.39 | 18.1 % | 0.09/0.13 | 0.067 |
| **510 s3** | 550 | 0.77 | 1.05 | **10.4 %** | **24.2 %** | 25 % | 2.5 | 54 % | 1.78 | 20.8 % | **0.19/0.24** | 0.11 |
| 508 s3 | 550 | 0.56 | **1.23** | 8.0 % | 20.6 % | 17 % | 2.9 | 62 % | 1.26 | 12.6 % | 0.28/0.27 | 0.12 |
| 509 s3 | 1000 | 0.76 | 1.03 | 4.3 % | 11.9 % | 19 % | 2.3 | 70 % | 1.22 | 12.9 % | 0.09/0.13 | 0.058 |
| **510 s3** | 1000 | 0.75 | 1.05 | **5.2 %** | **14.0 %** | 22 % | 2.3 | 66 % | 1.41 | 16.6 % | **0.19/0.24** | 0.11 |
| 508 s3 | 1000 | 0.55 | **1.24** | 3.3 % | 10.0 % | 14 % | 3.0 | 65 % | 1.13 | 9.9 % | 0.29/0.28 | 0.11 |

508 save_3: probed 13/11 with the same binary in `--force` (diff = Φ and probe geometry; the stabiliser was already running uncommitted in 508).

Aligned & <40 m counts: 880 (550) and 389 (1000) ⟹ ratio noisy at 1000. No μ saturation, σ never at ceiling.

CSV (510 / 509, same indices): skip 0 all run vs 0.37-0.46; skip_canon 0.07 → 0.17; joint kl 0.027-0.031 like 509; hit 13.5 / 13.8 %, kill 23.5 / 23.7; Hc −0.23 / −0.51 at 82 k. D milestones identical at index in 508, 509 and 510: non-discriminating.

## Reading

- **Trigger**: flat and in dump, like 509 at same index. p_fire 0.75-0.80 over all angle slices (gap 0.04), no selectivity in deterministic (> 0.5 everywhere), reserve empty at 2.3-2.5 s, 93-99 % of tank-episodes empty. Removing the joint skip changed nothing at this stage, consistent with 509 save_5's measurement (trigger KL 0.0002-0.0018 per step, never limiting).
- **σ canon**: 0.19/0.24 vs 0.09/0.13 in 509: 509 save_3's σ canon trough did not occur. Canon is now stopped alone (skip_canon ~15 %) instead of dragging the whole actor into 40 % cuts.
- **Aligned fire**: h<5° 24/14 % vs 18/12 %, hit/shell +28 % / +21 %. Same trend as 509 save_5's precision lead, slightly earlier. Single seed.
- **Turret**: 20.8/16.6 % vs 18.1/12.9 %, slightly better, within known seed noise.

## Established / noise

- Established: per-group stop removes the joint skip (0 vs 40 %) and canon keeps a σ ~2× wider at 81 k.
- Established: trigger does not benefit yet (flat, dump identical to 509).
- Possible noise: precision and turret gains (one seed, 4 episodes).
- Established (two sides): at 81 k, 508's trigger already sorts (ratio 1.23-1.24, p_arm 0.55, reserve 2.9-3.0 s) while 509 and 510 are still flat (1.02-1.05, p_arm 0.77). Trigger lag of factorised-credit runs exists from save_3 and lifting the skip does not close it: its cause is elsewhere than the joint skip.
- 508 is behind 510 on aim: turret<5° 12.6/9.9 % vs 20.8/16.6 %, h<5° 20.6/10.0 % vs 24.2/14.0 %.

## Next

- 510 save_5 (135 k) vs 509 save_5 and 508 save_5: does the trigger leave the dump (ratio > 1.5, reserve > 4 s)?

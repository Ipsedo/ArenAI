# Run 537 — ppo_liquid, coloured noise + actor batch 2400, without a·|a| (start)

Launched 2026-11-19 07:31, running, read at 94 k (D 350), 82 k steps/h. Code: b83da585 + uncommitted working tree (training binary 2026-11-19 07:31): `actor_batch_size` 512 → 2400 as default, a·|a| removed from `torch_types`. Flat 7-block CNN, actor lr 1e-4, ρ 0.9.

## What the run tests

534 + two **confounded** changes: actor batch 2400 (16 planned actor steps per rollout instead of 76) and linear turret command (a·|a| removed).

## CSV

D milestones: 300 at 48 k, 350 at 67 k (534: 48 k / 67 k; 528: same ±5 k).

| index | run | kl | clip | skip | kill | hit | fire | Hc |
|---|---|---|---|---|---|---|---|---|
| 1.8 k | 534 / 537 | 0.026 / 0.006 | 0.31 / 0.11 | 0.10 / 0 | – | – | 0.77 / 0.56 | −0.06 / −0.06 |
| 20 k | 534 / 537 | 0.061 / 0.031 | 0.40 / 0.31 | 0.28 / 0 | 6.7 / 3.5 | 0.053 / 0.055 | 1.19 / 1.27 | −0.06 / −0.19 |
| 45 k | 534 / 537 | 0.074 / 0.056 | 0.39 / 0.36 | 0.49 / 0.01 | 17.3 / 16.6 | 0.092 / 0.110 | 1.92 / 1.74 | −0.17 / −0.33 |
| 66 k | 534 / 537 | 0.068 / 0.069 | 0.38 / 0.36 | 0.53 / 0.05 | 19.0 / 19.7 | 0.104 / 0.115 | 1.61 / 1.11 | −0.26 / −0.34 |
| 80 k | 534 / 537 | 0.069 / 0.066 | 0.39 / 0.38 | 0.58 / 0.08 | 20.5 / 19.4 | 0.107 / 0.111 | 1.91 / 1.35 | −0.31 / −0.42 |
| 90 k | 534 / 537 | 0.070 / 0.080 | 0.40 / 0.39 | 0.65 / 0.12 | 20.2 / 18.8 | 0.109 / 0.112 | 1.27 / 1.35 | −0.37 / −0.49 |
| 80 k | 526 (white, batch 2400) | 0.011 | 0.15 | 0 | 21.0 | 0.107 | 1.80 | −0.22 |

- Slower start than 534 (kill 0.6 at 10 k vs 2.6; 3.5 at 20 k vs 6.7), caught up at 45 k.
- kill and hit flat since 56 k (kill 18.3 → 19.7 → 18.9; hit 0.115 → 0.111).
- kl is not healthy: from 0.006 it reaches 534's at 66 k (0.069) and exceeds it at 90 k (0.080). Skip takes off: 0.05 at 66 k, 0.10 at 85 k, 0.153 at 94 k (+1 pt / 2 k over last 10 k). Batch 2400 shifted the KL regime by about 60 k, did not remove it (526, white noise: kl 0.011 at same index).
- Hc falls faster than 534 (−0.49 at 90 k), α at floor except `alpha_dir_y` (0.0021).

## Diagnosis

- Established: CSV level ≡ 534 at equal D (within seed noise); skip ÷5 at equal index but accelerating.
- Probes (probes/save_3.md): the coloured-noise sign on the turret (533/534/535 s3) is not reproduced; the trigger becomes selective earlier.
- Not settled: which of the two changes explains the aim gap.
Options (no decision): (1) let run to s5 (135 k): see whether corrective rises once the curriculum advances and where skip goes; (2) isolate: 537 with `--actor_batch_size 512` (= 534 without a·|a|), judged at s3; (3) isolate the other way: put a·|a| back with batch 2400.
Markers: skip at 135 k; beyond 0.75 the rollout is no longer seen whole; s5: corrective 5-40° vs 0.57 (537 s3), 0.62-0.64 (534 s3), 0.59-0.60 (535 s5).

# Run 528 — ppo_liquid

Launched 2026-11-17 16:46, read at 17:17, **running**. Index 40 k (45 rows), D 250, 79 k steps/h (2.54 M tank-steps/h), save_1 written 17:08.
Code: uncommitted tree on `68c41e20`, binary `cmake-build-release` compiled 16:46:51, just before launch. Command line without economy option and `config.json` gives `shells_per_hit` = 5 ⟹ default 5 is compiled, with the kills/tank curriculum (`curriculum_ratio_low/high` = 0.3 / 0.5). High confidence.

## What the run tests

| | 525 | 527 | 528 |
|---|---|---|---|
| curriculum criterion | touches/shots 0.05 / 0.08 | kills/tank 0.3 / 0.5 | kills/tank 0.3 / 0.5 |
| `shells_per_hit` | 2 | 2 | **5** |
| `actor_batch_size` | 512 | 512 | 512 |

Only difference with 527: 5 shells refunded per touch (break-even of a shot 50 % → 20 %). This is the config planned on 2026-11-17: the two levers together. 527 (stopped at 30 k, D 250) serves only as seed replica of the start.

## CSV (D 250 throughout), means over 4 k of index

| index | run | kill | hit | fire/s | r_hit | r | Hc | kl | clip | ev |
|---|---|---|---|---|---|---|---|---|---|---|
| 20 k | 528 | 11.3 | 0.070 | 1.33 | 3.5e-3 | 3.4e-4 | 0.24 | 0.024 | 0.27 | 0.53 |
| 30 k | 528 | 14.4 | 0.085 | 1.17 | 4.7e-3 | 1.1e-3 | 0.25 | 0.034 | 0.31 | 0.32 |
| 40 k | **528** | **17.4** | **0.105** | **1.31** | **5.6e-3** | **1.7e-3** | 0.25 | 0.040 | 0.33 | 0.73 |
| 40 k | 524 | 14.8 | 0.105 | 0.94 | 4.4e-3 | 8.3e-4 | 0.29 | 0.044 | 0.34 | 0.66 |
| 40 k | 525 | 11.4 | 0.089 | 0.83 | 3.9e-3 | 4.8e-4 | 0.27 | 0.045 | 0.35 | 0.40 |
| 40 k | 526 | 15.9 | 0.085 | 1.51 | 5.1e-3 | 4.9e-4 | −0.08 | 0.010 | 0.13 | 0.38 |
| 30 k | 527 | 12.3 | 0.093 | 0.87 | 3.9e-3 | 6.9e-4 | 0.37 | 0.028 | 0.27 | 0.57 |

- Steady, uneventful climb: kill 8 → 18.5, hit 0.058 → 0.112, r > 0 from 13 k.
- **Higher firing cadence** than 2-shell/batch-512 runs: 1.2-1.3/s vs 0.8-0.95 (524/525/527) from 15 k. Expected economy effect (more shells in reserve) and the only gap beyond noise; hit per shot identical to 524 (0.105).
- r_hit and r highest of the five runs at 40 k (r ×2 vs 524), via firing volume.
- kl 0.04 (target 0.10), skip ~0, clip 0.33, Hc stable 0.25, α in free decay (1.4e-4, floor 1e-4): nothing pathological. ev 0.5-0.75 with two isolated troughs (27 k, 30 k).
Kills per episode (rebuilt from 16-episode moving mean): 8, 17, 11, 4, 4, 10, 17, 14, 17, 15, 16, 23, 20, 22, 19, 16, 19, 20, 19, 22, 16, 23. The last 11 all ≥ 16, mean 19.9 = 0.62 kill/tank.

## Diagnosis

Established: config correct: economy 5 + kills/tank curriculum, nothing else differs from 527. D still 250 is not a block: the criterion decides every 8 probe episodes, i.e. ~16 episodes at `boundary_proba` 0.5 (~29 k index). First window covered start-up episodes (mean ~13 = 0.4/tank, dead band). Second falls at ~55-60 k (≈ 17:30) and the episodes filling it are at 0.6/tank, above threshold 0.5. Structural consequence of the criterion: at most one D step per ~29 k index at start. D 1000 needs 15 steps ⟹ not before ~450 k in the best case (510: 428 k under old criterion).
Noise: kill 17.4 vs 14.8 (524) and 15.9 (526) at 40 k: seed-noise size (525 was at 11.4 at almost equal code).
Not settled: all that motivates the run: trigger open at 50-100 m, kills held at D ≥ 600 (525: kill 16 → 8, fire 0.34 → 0.16), possible dump return with 5 shells/touch (513-514). CSV `fire` is blind to the dump: higher cadence may be aiming or spraying.
Options (Samuel): let it run (nothing justifies touching); if D still 250 at 90 k with CSV kill ≥ 16, look at the criterion (probe episodes would be under the mean), not the run.
Markers: D 300 expected ~55-60 k, then a step every ~29 k while kill/tank > 0.5. Probe save_5 (~135 k) by distance band; CSV at D 600-800: kill and fire must hold where 525 collapsed.

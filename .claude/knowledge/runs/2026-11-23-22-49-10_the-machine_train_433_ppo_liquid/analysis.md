# train_433_ppo_liquid — analysis (2026-11-24, run in progress)

- Launched 23/11 22:49, last row 24/11 13:41, index 404 k, 27.1 k steps/h (868 k tank-steps/h), 32 tanks.
- Effective code: f7435cc2 + staged uncommitted changes (`vision.cpp/.h`, `constants.h`, `*_hyperparams.h`) — high confidence: sources of 23/11 21:51, training binary 22:49, `actor_repr.txt` shows 4 blocks and an LTC input 448 = 64 + 3 × 128.

## What the run tests

4-block CNN (instead of 7) + spatial softmax (x, y, amplitude per channel, temperature per channel) + presence head on the map read by the policy. Intended reference: 552 (other PC, db9a825a, 7 blocks flattened, head at block 4). Vs local 432: 5 commits + architecture ⟹ confounded, 432 is not a reference.

## CSV

D milestones: 300 @ 48 k, 400 @ 114 k, 500 @ 181 k, 600 @ 233 k, 700 @ 281 k, 800 @ 328 k, 850 @ 366 k (552: D 1000 @ 699 k, intermediate milestones unknown here).

| D | index | hit | kill | fire | miss | ev | Hc | Hd | skip |
|---|---|---|---|---|---|---|---|---|---|
| 350 | 67-113 k | 0.142 | 24.1 | 2.04 | 34 | 0.38 | −0.31 | 0.63 | 0 |
| 500 | 181-207 k | 0.107 | 23.2 | 1.67 | 42 | 0.38 | −0.80 | 0.57 | 0.007 |
| 650 | 267-280 k | 0.085 | 17.4 | 1.33 | 43 | 0.25 | −0.68 | 0.53 | 0.05 |
| 750 | 305-327 k | 0.115 | 20.2 | 1.32 | 46 | 0.49 | −0.69 | 0.43 | 0.03 |
| 850 | 366-404 k | 0.159 | 20.9 | 1.13 | 41 | 0.47 | −0.43 | 0.32 | 0.07 |

- hit recedes from 0.14 (D 350) to 0.085 (D 650) then rises to 0.16 (D 850) while fire drops from 2.0 to 1.1: the hit rise comes from the trigger (fewer off-axis shots), not from aim (probe save_14: turret<5° 0.13-0.19).
- α_canon_y rises from 4.5e-4 to 1.1e-3 (pitch Hc under target); other α at floor 1e-4.
- 552 marker at D 1000: hit 0.38, fire 0.42, kill 19.3, miss 46 m. 433 is not there.

## Diagnosis (detail: `probes/save_14.md`)

- Established: target position present at the policy input (0.87° / 1.51° / 4.18° at 60-100 / 100-200 / 200-400 m), at least as good as frozen 552 (1.26 / 2.19 / 5.34° with 2× more readout data).
- Established: not used beyond 100 m; gesture credit ρ ≈ 0; sweep with dominant sign 84-95 %.
- In the noise / undecidable: curriculum speed and result vs 552 (no 552 CSV here, not at D 1000).

## Markers

- D 1000: hit ≥ 0.38 and kill ≥ 19.3 to equal 552; hits/tank-s > 100 m in the probe.
- Probe at D 1000: corr(μ canon_x, yaw) at 100-400 m must become negative; oracle/policy ratio must drop below ×1.9 (550) and ×3.1 (1000).

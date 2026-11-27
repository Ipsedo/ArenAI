# Run 541 — canon credit on the critic's H head (running)

`train_541_ppo_liquid`, ppo_liquid, launched 2026-11-20 00:52, running; read at 98 k (D 350). Throughput 81 k steps/h (2.59 M tank-steps/h). Code: uncommitted tree above b83da585, binary `cmake-build-release` of 2026-11-20 00:52. Sources modified after launch (code review): working tree no longer exactly the run's code.

## What the run tests

Config identical to 539 (current image, actor batch 2400, single advantage). Only difference: the canon (canon_x, canon_y) is credited at one step on normalised γ·σ(H(s′)) − σ(H(s)), H = `to_shot_value`, linear head of the critic trained by auxiliary BCE ×0.1 on labelled shots; other actions keep GAE; group KL stop. New columns: `skip_canon`, `shot_value`.

## CSV (98 k)

D milestones identical to 538/539: D 300 at 48 k, D 350 at 67 k.

| index | D | kill 538 / 539 / 541 | hit | fire | Hc | kl |
|---|---|---|---|---|---|---|
| 18 k | 250 | 11.4 / 15.0 / 14.7 | 0.085 / 0.088 / 0.089 | 1.05 / 0.95 / 1.15 | −0.13 / −0.15 / −0.27 | 0.007 / 0.008 / 0.013 |
| 45 k | 250 | 19.2 / 19.4 / 20.6 | 0.112 / 0.110 / 0.109 | 1.94 / 1.76 / 1.02 | −0.39 / −0.31 / −0.65 | 0.010 / 0.009 / 0.018 |
| 63 k | 300 | 22.6 / 23.6 / 19.5 | 0.116 / 0.122 / 0.101 | 2.46 / 1.43 / 0.58 | −0.55 / −0.41 / −0.92 | 0.011 / 0.012 / 0.016 |
| 81 k | 350 | 22.8 / 24.4 / 19.1 | 0.109 / 0.120 / 0.098 | 1.94 / 1.06 / 0.82 | −0.42 / −0.51 / −0.79 | 0.015 / 0.014 / 0.023 |

- Identical to 539 up to 45 k, then plateau: kill 19-21 and hit 0.098-0.102 from 45 k to 98 k, where 538/539 rise to 23-24 and 0.12. `fire` 0.5-1.0/s vs 1.1-2.5.
- Hc falls twice as fast (−0.8 vs −0.45 at 80 k): σ canon tightens (probe: 0.12 → 0.07). kl 0.02 vs 0.012, `skip` 0, `skip_canon` ≤ 0.2 %.
- `shot_value` (in-sample BCE) 0.16 → 0.06-0.16, noisy, no trend after 30 k. `ev` unstable (0.1-0.6, trough −1.9 at 57 k); α of dir_x / dir_y rise (0.001 → 0.0025).

## Diagnosis (98 k)

- Established (probes s1-s3, probes/save_3.md): the turret does not steer toward the target (corrective 0.51-0.55, gain 5× weaker than 539), it calms (command ÷2); same signature as 540. The trigger becomes selective on angle (ratio 1.2).
- The CSV kill plateau coincides with the cadence drop and aligned time divided by 2.
- Noise: kill gap at a single index (one run per arm). Not settled: effect at D ≥ 400.
Markers: save_5 (135 k) vs 539 s5; kill at D 400-450 vs 539 (25-26).

## Update at 703 k (2026-11-20 afternoon)

D milestones identical to 539 up to D 800 (328 k), then D 850 at 366 k and D 900 at 638 k.

| phase | index | D | kill | hit | fire | Hc | kl |
|---|---|---|---|---|---|---|---|
| lag on 539 | 45-150 k | 250-450 | 18-21 (539: 24-26) | 0.10 | 0.5-1.2 | −0.2 to −0.8 | 0.012-0.022 |
| high 1 | 165-345 k | 450-800 | 24-27 (539: 25 → 17) | 0.12 → 0.17 | 0.7-2.0 | −0.6 to −1.0 | 0.03-0.06 |
| low 1 | 360-480 k | 850 → 700 | 22 → 4-6 | 0.13-0.21 | 0.04-0.3 | −0.04 to −0.3 | 0.012-0.027 |
| high 2 | 495-600 k | 700 → 850 | 25-26 | 0.18-0.19 | 0.5-1.4 | −0.7 to −1.3 | 0.05-0.09 |
| low 2 | 615-703 k | 900 → 800 | 23 → 9-11 | 0.19-0.30 | 0.02-0.4 | −0.2 to −0.6 | 0.012-0.02 |

- The morning diagnosis (aim dead at 81 k) no longer holds after 135 k: see probes/timeline.md. Aim and selective trigger appear between s3 and s9 and hold at s20 and s26.
- The run oscillates: firing cadence collapses twice, the curriculum lowers D, competence returns. Cause not established. `skip_canon` 6-10 % and kl 0.05-0.09 in high phase 2.

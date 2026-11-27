# Run 526 — ppo_liquid, actor batch 2400

Launched 2026-11-17 13:42, **running**; read at 14:47, index 93 k (D 350), 86 k steps/h (2.75 M tank-steps/h). Code: uncommitted tree on 68c41e20, identical to 525 (binary rebuilt 13:42:03, only file touched since 525: `liquid_ppo_hyperparams.h` at 13:40). High confidence.

## What the run tests

One change vs 525: `actor_batch_size` 512 → 2400 (whole rollout per actor step; `actor_gpu_batch_size` stays 512). This is the "lower per-step noise" option of the 525 audit. Identical: pure events (Φ removed), single advantage, flat CNN, actor epochs 4, target_kl 0.10.

## CSV

D milestones identical to 524 and 525 (300 @ 48 k, 350 @ 67 k): at this stage equal-index comparison = equal-D comparison.

| D | run | kill | hit | fire/s | r_hit | s | miss | Hc | kl | clip | skip |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 250 | **526** | 13.0 | 0.073 | 1.35 | 4.3e-3 | 58 | 24 | −0.02 | 0.007 | 0.10 | 0 |
| 250 | 525 | 7.4 | 0.064 | 0.90 | 2.8e-3 | 56 | 24 | +0.21 | 0.032 | 0.30 | 0.003 |
| 250 | 524 | 10.9 | 0.074 | 0.96 | 3.3e-3 | 53 | 24 | +0.24 | 0.030 | 0.28 | 0.003 |
| 300 | **526** | 19.1 | 0.101 | 1.73 | 6.5e-3 | 54 | 29 | −0.18 | 0.010 | 0.13 | 0 |
| 300 | 525 | 12.9 | 0.112 | 0.83 | 4.8e-3 | 57 | 25 | +0.19 | 0.053 | 0.34 | 0.03 |
| 300 | 524 | 16.3 | 0.133 | 1.06 | 6.2e-3 | 59 | 25 | +0.22 | 0.053 | 0.35 | 0.02 |
| 350 | **526** (67-93 k) | 20.9 | 0.107 | 1.71 | 8.0e-3 | 49 | 35 | −0.18 | 0.012 | 0.14 | 0 |
| 350 | 525 (67-113 k) | 13.2 | 0.118 | 0.89 | 4.8e-3 | 60 | 27 | −0.03 | 0.055 | 0.35 | 0.14 |
| 350 | 524 (67-113 k) | 19.7 | 0.143 | 1.16 | 7.1e-3 | 53 | 30 | −0.05 | 0.057 | 0.33 | 0.14 |

Last slice (84-93 k): kill 21.7, hit 0.109, fire 1.63, ev 0.46, kl 0.014, clip 0.15. `alpha_canon_x` 2.3e-3 and `alpha_canon_y` 2.1e-3 rising continuously since 20 k, `alpha_fire` and `alpha_zoom` falling (3.8e-4).

## Diagnosis

Established: optimisation regime changed as expected: kl ÷5 (0.012 vs 0.055), clip ÷2.5, skip null (525: 14 % at D 350); kill +60 % vs 525 at D 250-350 but via **firing volume** (fire ×1.9), not precision: hit ≤ 525 and < 524, miss larger (35 m vs 27 m), shorter life (49 s vs 60 s); continuous σ tightens earlier (Hc −0.18 from 50 k; 525: > 0 until 67 k) and the regulator reacts on canon (α_canon ×2.3, at floor in 524/525, 0.006-0.015 in 522/control).
Noise: lead over 524 (kill 20.9 vs 19.7 at D 350).
Not settled: aim and trigger. CSV does not say if firing volume is a dump nor whether the turret holds the target; 525 was also fine until D 550 and collapsed at D ≥ 600.
Options/markers: probe save_5 (135 k) vs 525 s5 (tour<5° 11.8/10.6 %, ratio 3.6, empty 6.1 s) and control (29 %); add dwell < 5° (525: 0.2 s, control: 1.0 s). CSV at D 600-800: 525 fell to kill 16 → 8 with fire 0.34 → 0.16; D 600 expected ~16:30 if milestones follow 525's. Watch Hc (−0.18 stable from 50 k) and α_canon.

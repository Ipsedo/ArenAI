# Run 504 — ppo_liquid — analysis at 320 k (2026-11-11 15:16)

## Header

- Launched 11/11 10:04 (mtime `config.json`), **running**, last row 15:16, index 320 k, D 750.
- Throughput 61.7 k steps/h (1.97 M tank-steps/h, 32 tanks), save_every ≈ 27 k (save_11 ≈ 297 k).
- Effective code: binary compiled 09:57 on uncommitted tree = content of **f50a43bd** (SoftClamp log σ + `critic_epochs = 8`), high confidence (memory note + config.json). f9342001 touches only SAC, out of scope. Current binary (10:31) postdates launch, no effect on the run.

## What the run tests

Base = 502 (batch 512, lr 1e-4 / 3e-4, target Hc 0). Two confounded but separable-trace changes: SoftClamp (σ/μ never saturated) and `critic_epochs` 4 → 8 (ev at D = 1000). Config diff vs 503 (batch 9600 → 512, lr) is only a return to 502's config.

## CSV reading — at equal D with 500-502

D schedule identical to 500-502 since D 400 (D 750 at 305 k like 501; 502: 328 k).

| D | run | hit | kill | ev | Hc | kl | skip | miss |
|---|---|---|---|---|---|---|---|---|
| 550 | 504 | **15.2 %** | 9.6 | **0.68** | 0.34 | 0.010 | 0 | 28 |
| | 500-502 | 12.7-14.0 % | 9.6-11.2 | 0.43-0.47 | 0.33-0.43 | 0.013-0.015 | 0.01-0.03 | 30-33 |
| 650 | 504 | **16.0 %** | 9.9 | **0.53** | 0.41 | 0.010 | 0 | 32 |
| | 500-502 | 10.6-13.8 % | 7.6-11.9 | 0.41-0.45 | 0.09-0.39 | 0.015 | 0.01-0.05 | 34-40 |
| 750 | 504 | **13.0 %** | 9.1 | **0.50** | 0.29 | 0.015 | 0.005 | 40 |
| | 500-502 | 10.0-13.2 % | 7.0-8.8 | 0.44-0.49 | −0.05-0.32 | 0.015-0.030 | 0.03-0.11 | 36-42 |

504 trajectory: continuous rise of CSV hit to 17.3 % (266 k, D 600, lineage record), then decline 16 → 13 % between D 650 and D 750 (270-320 k). Same window: miss 30 → 41 m, ev 0.67 → 0.50 (noisy points, one −0.12 at 314 k), kl 0.010 → 0.016-0.021, clip 0.14 → 0.22, skip 0 → 0.01, alpha_fire 0.001 → 0.006; continuous α at floor 0.001.

## Diagnosis

- **Established**: SoftClamp does its job (probes save_5/save_9: 0 % of σ at ceiling, |μ| > 1.5 = 0 %). No KL regime (skip ≤ 0.01 vs 0.03-0.11 for 500-502 at same D).
- **Within noise / undecided**: lead over 500-502 at equal D (ev +0.1-0.2, hit +1-3 pts). ev is in-sample and 8 critic epochs inflate it mechanically; hit is clearer but single seed.
- **Decline 650 → 750**: also present in 500 and 502 at same D (miss grows with spawn distance, target farther); not yet a 504-specific signal. kl/clip rising and alpha_fire leaving the floor reproduce the start of the drift seen in 500-502 near D 750-850, at lower level.

## Options and landmarks

- **Let run** until D = 1000: 500-502 reach it at 539-595 k ⟹ for 504 ≈ 520-580 k, i.e. **18:30-19:30** at current throughput (save_19-21).
- Early-stop signals: skip > 0.05 held, kl > 0.025, Hc < 0, ev < 0.3 over several rows, kill < 6.
- Test that counts: ev and hit at first D ≥ 950 step vs 500-502 (ev 0.36-0.41, hit 10 %).
- Redo probe at same D as landmarks: save_13 (~351 k, D ~800) then save_20 (~540 k), side 550 + 1000; read p_fire ratio, reserve empty, hit/shell vs 500 save_20 (22 %, ratio 4.7).

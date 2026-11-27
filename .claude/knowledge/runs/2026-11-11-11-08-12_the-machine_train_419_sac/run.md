# train_419_sac

| | |
|---|---|
| machine | the-machine |
| user | samuel |
| launch | 2026-11-11 11:08:12 |
| commit at launch | `f9342001` — returns for SAC computed on multiple steps (10 nexts for the moment) |
| commits ≤ 90 min after | none |
| diff at launch | unknown (identity recorded after the run ended) |
| raw data | `outputs/train_419_sac` on the-machine |

## Established facts

_SAC run 419 (2026-11-11) = 417 + n_step 10, only config diff; at 17k steps: Hc at target (−3.4) from 12k (415: 36k), q loss ×8, Q ≈ 0 (−0.09 bias gone), Hd 0.15, hit 0.053 = random floor; too early, judge = ∂Q/∂a probe on save_1 (27k)_

Run 419 launched 2026-11-11 10:52, config identical to 417 except `n_step` 10. Read at 17k steps (1 h 23).

Means 9k–17k (415 / 417 / 419): hit 0.075 / 0.059 / 0.053, fire 0.68 / 0.49 / 0.57 shells/s,
miss 21.2 / 21.8 / 22.7 m, q loss 0.0036 / 0.0030 / 0.023, Hd 0.23 / 0.30 / 0.15. D = 250 everywhere.
Hc reaches −3.3 (≈ σ target 0.1 over 4 axes) from 12k; Q goes from −0.09 (415/417) to ≈ +0.01.

**Why:** q loss ×8 expected (target = sum of 10 noisy rewards); at 17k hit is at the random floor
like 417, so nothing about aim is measurable yet.

**How to apply:** do not conclude on CSV metrics before ~50k; the decisive test is the probe
∂Q/∂a_canon (qgrad.py) on save_1: P(right direction) armed ≫ 0.5 (417: 0.47–0.50).
See run 417, sac-agent-2026-11.

2026-11-11, ∂Q/∂a probe on save_1 (27k, D=250, 2 episodes), 419 vs 417 save_1 at the same stage:
P(∂Q/∂a right direction) armed 0.51 vs 0.52, all slices 0.49–0.56 vs 0.50–0.54, corr ≈ 0 in both;
|∂Q/∂a| ×3 (2.5e-3 vs 8e-4); P(μ right direction) on |yaw| 10–30° 0.54 vs 0.64; armed 30 % vs 15 % of
decisions. → at 27k, n-step makes the critic more sensitive to the action without giving it a direction.
Even the potential term (action-dependent from s_{t+1}) does not appear in ∂Q/∂a.

2026-11-11, twins (probe: columns dq1/dq2 per critic, 1 episode D=250, save_1): sign agreement
Q1/Q2 on ∂Q/∂a_canon 0.48–0.55 and corr(Q1, Q2) 0.03–0.16 in 419 as in 417 (even armed, |yaw|<10).
→ Q's dependence on the action is fitting noise specific to each critic, not a shared
signal. Expected order of magnitude of the armed potential alone |yaw|<10: ~1e-2 to 1e-1, measured 8e-3 of
random sign. 1st-layer weights: action columns ≈ init, no atrophy. Probe pitfall: a single
autograd::grad with several outputs returns the gradient of their sum.

2026-11-11, feature probe (probe --features exports for SAC: actor [vision 192 | sensors 64],
then Q1, Q2; vision = SpatialSoftmax [x 64 | y 64 | presence 64]), 1 episode D=250, split by tank,
target = nearest tank. Ridge yaw (|yaw|<30°, <60 m): R² 0.02 (419 s1), 0.14–0.16 (417 s1),
−0.11 (417 s5); MLP worse (overfits). Yaw sign at |yaw|<10°: 0.52–0.57 everywhere. Detection
« tank < 40 m in the field (|yaw|<15°, |pitch|<10°) » AUC 0.74–0.86. Control: sensors → reserve
R² 0.67, CSV/bin alignment exact. → SAC encoders (actor AND critics) vaguely detect the
presence of a tank but do not encode its bearing, and it does not improve with training.
Limits: no positive control (no local PPO run), small n, target = nearest ≠ most salient.
Distinct from cnn-perception-validated (PPO CNN): here SAC CNN trained by TD alone.

# Run 420 (folder `train_20_sac`, wrong name) — SAC, shared state encoder

- Launched 2026-11-11 14:21, read at 15:44: **running**, index 41k, D = 250 since the start.
- Throughput 29.8k steps/h (955k tank-steps/h, 32 tanks) — comparable to 419.
- Effective commit: **aed53757** (shared encoder), high confidence: training binary compiled
  at 14:21 and containing `StateEncoder`; `actor_repr.txt` shows `SacActor { encoder: StateEncoder }`
  and `critic_repr.txt` `TwinQNetwork { encoder: StateEncoder, q1, q2 }`, same SpatialSoftmax
  temperature (0.3145) on actor and critic side.

## What this run tests

`config.json` identical to 419. Only code change: a single `StateEncoder` (vision + sensors),
carried by the twins and trained by the TD loss; the actor reads the detached encoding and optimises only
its head. Hypothesis: in 417/419 the vision features did not encode the target bearing (R² ≤ 0.16)
and ∂Q/∂a_canon had no direction; a shared encoder does not change the signal that trains it
(TD alone), it only removes the actor's own encoder. No confounded change.

## CSV reading (means per index slice, D = 250 everywhere)

| index | run | hit | kill | fire | miss | s | q | Q | Hc |
|---|---|---|---|---|---|---|---|---|---|
| 5–10k | 419 | 0.050 | 0.31 | 0.57 | 23.4 | 47 | 0.015 | 0.000 | −2.2 |
| | 420 | **0.012** | 0 | 0.43 | 25.4 | 29 | 0.005 | −0.003 | −0.7 |
| 10–20k | 419 | 0.054 | 0.37 | 0.58 | 22.5 | 61 | 0.025 | 0.009 | −3.5 |
| | 420 | 0.025 | 0.10 | 0.52 | 24.1 | 28 | 0.016 | 0.028 | −2.7 |
| 20–30k | 419 | 0.062 | 0.47 | 0.60 | 20.8 | 59 | 0.042 | 0.009 | −3.6 |
| | 420 | 0.042 | 0.26 | 0.58 | 22.3 | 47 | 0.038 | 0.072 | −4.0 |
| 30–42k | 419 | 0.070 | 0.66 | 0.69 | 19.3 | 71 | 0.057 | 0.026 | −3.7 |
| | 420 | 0.065 | 0.65 | 0.62 | 20.4 | 71 | 0.077 | 0.114 | −5.0 |

(419 stops at 33k; 415: hit 0.091 at 30–42k and D = 300, 417: 0.073.)

- **Slow start**: hit ≈ 0.004–0.01 until 8k (419: 0.03–0.05 from 3k), kill = 0 until 14k.
  Part of it precedes `learning_starts` (5k) and thus belongs to init/seed.
- **Catch-up**: steeper hit slope, 0.072 at 39k and 0.077 at 41k (run record, isolated);
  miss 26.9 → 19.8 m; s 20 → 72 s. Curriculum did not move (high threshold 0.08).
- **Hc overshoots the target**: −5.6 at 30k, min −5.97 at 31k (target ≈ −3.5 for σ 0.1 × 4 axes, which 419
  holds at −3.4/−3.7). αc fell to ~1.8e-4 then rise again (3.7e-4 at 41k): the α loop
  corrects late. The probe shows it is not the canon: median canon σ 0.48/0.43 (H ≈ −0.36/−0.30,
  σ ≫ target 0.1). The deficit comes from the direction axes, whose μ is saturated (|μ| > 1.5: 25 and 29 %
  of decisions): the post-tanh entropy target is reached by saturation, as in 415/416.
- Q rises to +0.11 (419: +0.01–0.03) and q ×1.3; consistent with a higher r at the end of the window,
  no negative drift as in 415.
- αd rises continuously (0.0037 at 41k), as in 419.

## Diagnosis

Established:
- The code is indeed the shared-encoder one, with no other change.
- At save_1 (27k), **∂Q/∂a_canon still has no direction** (see `probes/save_1.md`):
  P(right direction) 0.50–0.52 at all angles, armed 0.52, twins disagree in sign (0.45–0.49).
  Strictly identical to 417/419: the shared encoder did not make the critic sensitive to continuous
  aim.
- The trigger (discrete head, exact expectation over Q(s, ·) with K outputs) is **selective**:
  p_fire 0.50 under 5° vs 0.24 beyond 45°, monotone decrease with angle.

In the noise / undecided:
- Slow start then catch-up: single seed, the gap at 41k (0.065 vs 0.070 at 30–42k) is nil.
- Unknown whether the trigger's selectivity is new: no 417/419 probe was
  run with the current script (`ratio` metric).
- Hc under target: effect of the shared encoder (the actor follows features that move under it)
  or mere slowness of α; not attributable.

## Options (Samuel's decision)

1. Probe 419 save_1 with the same script (side 250) to know whether the selective trigger is due to
   the shared encoder or was already there (≈ 7 min with training in parallel).
2. Let it run to ~80–100k and probe save_2/save_3: does ∂Q/∂a take a direction later, or does only
   the discrete channel learn.
3. As in 419, test the shared encoder's features (`--features`): is the bearing encoded
   now that the encoder is single?

## Markers

- D passes 300 (hit > 0.08 held): 415 had it at ~20k, 420 is not there at 41k.
- Hc: should rise back toward −3.5 as αc rises; if it stays < −5, look at direction σ.
- Probe save_2: trigger `ratio` and armed P(∂Q/∂a right direction).

# train_420_sac

| | |
|---|---|
| machine | the-machine |
| user | samuel |
| launch | 2026-11-11 14:21:42 |
| commit at launch | `f9342001` — returns for SAC computed on multiple steps (10 nexts for the moment) |
| commits ≤ 90 min after | `f9df436a`, `aed53757` |
| diff at launch | unknown (identity recorded after the run ended) |
| raw data | `outputs/train_420_sac` on the-machine |

## Established facts

_SAC run 420 (folder outputs/train_20_sac, wrong name) = 419 + shared state encoder (aed53757): slow start then hit 0.077 at 41k (D=250); probe save_1: ∂Q/∂a_canon still without direction (0.50–0.52) but selective trigger (ratio 2.2/2.4); Hc < target by direction-μ saturation_

Run 420 launched 2026-11-11 14:21 in `outputs/train_20_sac` (Samuel mistyped the name; do not rename
while it runs). Config identical to 419; code = aed53757 (StateEncoder shared by the twins,
actor on detached encoding), confirmed by the checkpoint reprs.

CSV at 41k (D=250): hit ≈ 0.01 until 8k (419: 0.05), then catch-up to 0.065 over 30–42k (419: 0.070),
isolated record 0.077; Hc −5 to −6 (target −3.5); Q +0.11.

Probe save_1 (27k, sides 250/550, 4 episodes): armed ∂Q/∂a_canon P(right direction) 0.50–0.52, twins
disagree in sign (0.45–0.49), as 417/419 → the shared encoder does not lift the continuous-aim lock.
Selective trigger (p_fire 0.50 under 5° → 0.24 beyond 45°, ratio 2.2/2.4); dump unchanged
(empty reserve at 3.6 s, 73 % disarmed); canon σ 0.48 ≫ target 0.1, direction μ saturated 25–29 %.

**Why:** selectivity has no reference (no 417/419 probe with the probe-checkpoint script),
so it is not attributable to the shared encoder.

**How to apply:** before crediting the shared encoder, probe 419 save_1 with the same script; track
`ratio` and P(∂Q/∂a) on save_2+. See run 419, sac-agent-2026-11,
run 416 (post-tanh saturation).

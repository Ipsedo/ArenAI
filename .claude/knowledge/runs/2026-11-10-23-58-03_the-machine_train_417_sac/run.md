# train_417_sac

| | |
|---|---|
| machine | the-machine |
| user | samuel |
| launch | 2026-11-10 23:58:03 |
| commit at launch | `39ae97f9` — update probe for actor (no liquid) |
| commits ≤ 90 min after | `50a5e256`, `ff5fc4ab`, `40a6f467`, `11779aaf`, `3de97377` |
| diff at launch | unknown (identity recorded after the run ended) |
| raw data | `outputs/train_417_sac` on the-machine |

## Established facts

_SAC run 417 (2026-11-11, = 415 + stratified replay 0.7 + σ ceiling 0.7, target_sigma 0.1, buffer 600k) — worse than 415 and 416: curriculum 2× slower (D=300 at 48k, 350 at 114k), hit peak 0.103, continuous regression at D=350 (0.10 → 0.072, kill 3.8); probe: unconditional fire, angle at fire 16.6° → 28.6°, corr(μ, error) ≈ 0_

Run 417 (relaunched 2026-11-10 23:58, WIP binary that became 50a5e256; state layout unchanged, verified),
analysed at 153k steps / 10 h, still running. Config diff vs 415: `armed_fraction` 0.7 + buffer
600k (vs 800k); σ ceiling 0.7 (e52e76ff) present, absent from 415. Same `target_sigma` 0.1 as 415.

CSV, per-stage means (415 / 416 / 417):
- D=250: end at 20k / 20k / **47k**.
- D=300: hit 0.090 / 0.104 / 0.088, kill 4.3 / 5.6 / 3.3; 417 stays there 65k steps.
- D=350: hit 0.117 / 0.103 / 0.086, kill 11.9 / 8.0 / 3.8, fire 1.54 / 0.79 / 0.67 shells/s.
Hit peak 0.103 at 100k, then monotone decline 0.100 → 0.072 and miss 22.7 → 28.4 m over 40k steps at fixed
D (not a mere stage jump). Hc −3.6 from 36k (as 415), Q ≈ −0.2, q loss 0.027
(×1.5 vs 415), αd0 rises to 6e-3 at 100k then falls back.

Probe (D=350, 2 episodes, sampled) save_3 (81k) / save_5 (135k):
- p(fire | can_fire) 0.86 / 0.88, nearly independent of angle; dry 84 / 88 % of decisions.
- canon σ 0.53 / 0.59 (415: 0.42–0.50; ceiling rarely reached), |μ| pitch 0.69 → 0.98, pitch saturation
  15 → 28 %: entropy still satisfied by saturation.
- median angle at fire 16.6° → 28.6°, share of shells > 30° 24 → 48 %; hit at equal angle ~stable
  (< 2°: 0.19/0.17 vs 0.31 in 415) → the regression = alignment degrading.
- corr(μ canon, yaw/pitch error) −0.05 / −0.02: still no aim.

**Why:** attribution impossible: 416 (stratified + ceiling) left D=250 at 20k like 415, so
stratification does not explain 417's own slowness; the 50a5e256 refactor re-read = equivalent;
no fixed seed (no manual_seed) → 415/416/417 gaps possibly seed noise. Only robust
fact: corr(μ, error) ≈ 0 in the 3 runs, settings move hit only by ±0.02.

**How to apply:** stop attributing a gap between single SAC runs to a hyperparameter. Proposed check
(minutes, no run): on probe states, corr(∂Q/∂a_canon, yaw/pitch error) → does the critic
see the value of aiming? (requires extending the probe, to validate with Samuel).
See run 415, run 416, sac-agent-2026-11.

2026-11-11, critic gradient (extended probe: columns `dq_canon_x/y` = ∂E[min Q]/∂a_canon at the mode,
save_5, D=350, uncommitted): Δyaw = +7.4°·act per decision (corr 0.42); **P(∂Q/∂a in the direction
that reduces |yaw|) = 0.47–0.50 in all slices**, including armed + can_fire + |yaw| < 5°; corr ≈ 0.
|∂Q/∂a| ~6e-4 (1.4e-3 armed). → the critic does not see the value of aiming: lock = credit, not actor.
Pitch not interpretable: Δpitch does not respond to act_canon_y (corr 0.00), `pitch` geometry to check.
No factor-3 bug on « seconds before empty reserve »: 1 probe step = 3 frames = 0.1 s,
cooldown 5 frames → 1 shot / 2 decisions, 10 shells in ≥ 2 s (physical minimum).

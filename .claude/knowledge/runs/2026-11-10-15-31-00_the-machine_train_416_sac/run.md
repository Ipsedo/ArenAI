# train_416_sac

| | |
|---|---|
| machine | the-machine |
| user | samuel |
| launch | 2026-11-10 15:31:00 |
| commit at launch | `39ae97f9` — update probe for actor (no liquid) |
| commits ≤ 90 min after | none |
| diff at launch | unknown (identity recorded after the run ended) |
| raw data | `outputs/train_416_sac` on the-machine |

## Established facts

_SAC run 416 (2026-11-10, 415 + target_sigma 0.3 + 70 % armed stratified replay + σ ceiling ≤ 0.7) — at 69k: faster at D=300 (hit 0.104) but lower peak (0.114) and regression from D=350 (0.103; 0.096 at D=400), kill 8 vs 12. Probe save_2: canon σ stuck at ceiling 0.7 on 94 % of decisions, action noise ×1.7, median angle at fire 19.6° vs 11.8°; p(fire|can_fire) 0.88, reserve emptied in 2 s, dry 84 % of the time; corr(μ, error) ≈ 0_

Run 416 (launched 2026-11-10 15:31, analysed at 69k steps / 4.6 h, still running). Three simultaneous
changes vs 415: `target_sigma` 0.1 → 0.3 (Hc target −3.58 → +0.52 over 4 axes), stratified replay
`--armed_fraction 0.7` (uncommitted), and the `LOG_SIGMA_MAX` ceiling −0.3567 (σ ≤ 0.7,
commit e52e76ff 09:25, after 415's save_0 at 08:28 → 415 ran without it).

CSV, comparison at equal D (per-stage means):
- D=300: hit 0.104 vs 0.090, kill 5.6 vs 4.3 — 416 learns faster.
- D=350: hit 0.103 vs 0.117, kill 8.0 vs 11.9, fire 0.79 vs 1.54 shells/s, miss 25.9 vs 28.6.
- D=400 (3 rows): hit 0.096 vs 0.111, kill 7.8 vs 13.2.
Hit peak 0.114 at 37k (D=300) then continuous decline; 415 peaked at 0.121 at 60k (D=350).
Hc stuck at its target 0.52, αc stable 5e-4; αd0 rises 2× faster than 415 (5.2e-3 at 46k) then
falls back. Q starts at −0.47 (armed states at random fire) and ends −0.08 vs −0.33, but measured
on a stratified batch: not comparable.

Probe save_2 (54k, D=350, 2 episodes, sampled) vs 415 save_2:
- **canon σ = 0.70 (the ceiling) on 94 % of decisions** (415: 10 %, σ 0.38/0.52), |μ| 0.43 vs
  0.79–0.97. The σ 0.3 entropy target is reached with pre-tanh σ at max and moderate μ; action
  noise (std of act − tanh μ) 0.48 vs 0.26–0.30. Median angle at fire 19.6° vs 11.8°, 75 % of
  shells at > 10° (415: 58 %). Hit per shell at equal angle identical to 415 (0.21/0.19/0.15/0.07/0.02)
  → the hit loss comes entirely from the noisier alignment.
- **p(fire | can_fire) 0.88** (415 save_2: 0.43, save_3: 0.81), reserve emptied in 2 s, dry
  84 % of decisions (415: 53 %), lives 20 % longer → 0.77 vs 1.42 shells/s of life. The cadence
  drop is NOT a more stochastic fire head (CSV hypothesis refuted): it is fewer
  hits → fewer refills, and a longer dry life.
- corr(μ canon, yaw/pitch error) −0.06 / −0.01: still no aim (as 415).

Structural observation: in both runs the actor satisfies the post-tanh entropy target by
saturation, never by a small σ — 415 by |μ| → 1 (tanh squashes), 416 by σ at the ceiling. The σ 0.3
target therefore leaves a much larger pre-tanh noise than intended.

**Why:** 416 does not improve on 415; the dominant factor is `target_sigma` 0.3 (via σ saturating at
the ceiling), not stratification, whose effect remains unisolated.

**How to apply:** options not decided, to validate with Samuel: return to `target_sigma` 0.1
keeping the stratified replay (isolates stratification); or define the entropy target pre-tanh
(gaussian alone) so it can no longer be bypassed by saturation. The probe
`arenai_agent/probe/main.cpp` was repaired (SacActor → Actor, uncommitted) and compiled in
cmake-build-release. See run 415, sac-agent-2026-11.

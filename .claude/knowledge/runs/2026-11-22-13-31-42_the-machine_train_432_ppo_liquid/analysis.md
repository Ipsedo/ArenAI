# train_432_ppo_liquid — analysis at 122k (running)

- Launched 2026-11-22 13:31, analysed at index 122k (D=400), ~48k steps/h, 32 tanks.
- Code: working tree uncommitted at launch, on top of `dcca274e`. Sources have mtime 14:16
  (commit `7bccaa04` + pre-commit), content assumed identical at launch: medium confidence.
- Probe: `build/` binary of 13:28, no inference diff reported by the script.

## What this run tests

Vs the previous lineage (431 ppg, 430 liquid):

- a single optimisation group for the policy: no more separate canon / rest log-prob, no more
  aim advantage (`aim_credit` removed), no more per-group KL stop (`skip_canon` gone);
- continuous entropy target −1 on the 4 actions (0 before);
- value bootstrap at the time limit (`dcca274e`).

These three changes are confounded. No recent liquid reference run (430 = 4k steps).

## CSV

| slice (index) | D | hit | kill | fire | miss (m) | ev | Hc | kl | clip |
|---|---|---|---|---|---|---|---|---|---|
| 0-10k | 250 | 0.038 | 2.8 | 0.78 | 22 | 0.45 | 0.10 | 0.004 | 0.07 |
| 30-41k | 250 | 0.135 | 19.9 | 1.44 | 28 | 0.56 | −0.20 | 0.012 | 0.15 |
| 51-61k | 300 | 0.154 | 24.3 | 1.59 | 34 | 0.47 | −0.23 | 0.013 | 0.14 |
| 92-102k | 350 | 0.137 | 25.5 | 1.49 | 43 | 0.51 | −0.35 | 0.019 | 0.17 |
| 122k | 400 | 0.129 | 24.7 | 1.76 | 45 | – | – | – | – |

- `hit` ×4 between 10k and 60k, then slowly recedes as D rises; `miss` follows D.
- `skip` ≈ 0, `kl` 0.02 under the 0.05 target: no KL regime.
- D milestones: 300 @ 48k, 350 @ 67k, 400 between 102k and 120k. 431 (ppg) at D=350: hit 0.154,
  kill 27.5, miss 47; gap within noise.

## Probes (4 episodes, stochastic)

| save | index | side | turret<5° | hit/shell | h<5° | shots<5° | trigger ratio | %empty | canon σx |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 350 | 0.053 | 0.029 | 0.122 | 0.089 | 1.00 | 1.00 | 0.299 |
| 1 | 27k | 350 | 0.154 | 0.122 | 0.329 | 0.199 | 1.07 | 0.92 | 0.186 |
| 3 | 81k | 350 | 0.188 | 0.135 | 0.368 | 0.242 | 1.27 | 0.83 | 0.089 |
| 3 | 81k | 550 | 0.128 | 0.100 | 0.313 | 0.202 | 1.35 | 0.93 | 0.168 |

Bearing servo, measured outside the standard probe: regression of μ_canon_x on the signed error
towards the nearest target (`yaw`), converted into fraction of error closed per decision.
Measured convention: a canon_x action of +1 shifts the error by +12.7° per decision, the right
direction is therefore a negative correlation.

| error | save_0 corr | save_1 corr / closure | save_3 corr / closure |
|---|---|---|---|
| 0-5° | +0.04 | −0.19 / 13 % | −0.19 / 14 % |
| 5-10° | +0.06 | −0.30 / 9 % | −0.24 / 8 % |
| 10-20° | +0.08 | −0.30 / 5 % | −0.26 / 5 % |
| 20-45° | +0.10 | −0.20 / 1.4 % | −0.15 / 1.4 % |
| 45-180° | +0.05 | −0.05 / 0.1 % | −0.06 / 0.1 % |

## Diagnosis

Established:

- Bearing aim is learned: feedback loop absent at init, present at 27k, and the turret spends
  3.5× more time within 5° of a target (0.053 → 0.188).
- It is acquired before 27k and its gain no longer progresses between 27k and 81k. It acts only under
  ~45° of error and closes only 5 to 14 % of the error per decision.
- The turret<5° gain between save_1 and save_3 (0.154 → 0.188) coincides with the canon σ drop
  (0.186 → 0.089), not with a more directive μ.
- Trigger: weak but nascent selectivity (1.07 → 1.27), still coupled to zoom
  (corr 0.99); 83 % of tanks empty the reserve, in 3.4 s median.

Undecided:

- Pitch: corr(μ_canon_y, pitch error) between −0.02 and −0.09, and the canon_y action does not move
  the probe's `pitch` column (slope 0.2°): the measurement is not valid as is.
- Own effect of the single group: confounded with the −1 entropy target and the bootstrap.

## Update save_6 (162k, D=450, side 350)

| save | turret<5° | hit/shell | h<5° | trigger ratio | p_al / p_off | %empty | empty_s | canon σx p10/p50/p90 |
|---|---|---|---|---|---|---|---|---|
| 3 | 0.188 | 0.135 | 0.368 | 1.27 | 0.75 / 0.59 | 0.83 | 3.4 | 0.040 / 0.089 / 0.328 |
| 6 | 0.175 | 0.162 | 0.405 | 2.43 | 0.78 / 0.32 | 0.70 | 6.9 | 0.021 / 0.246 / 0.476 |

Bearing servo at save_6: corr −0.24 / −0.33 / −0.24 / −0.13 and closure
14.5 % / 8.8 % / 3.4 % / 0.8 % per decision over 0-5° / 5-10° / 10-20° / 20-45°.

- The aim gain stays flat for the 3rd consecutive probe (27k, 81k, 162k); turret<5° no longer
  rises (0.175).
- What progresses between 81k and 162k is the trigger: off-axis extinction (p_off 0.59 → 0.32),
  ratio ×2, reserve emptied twice later, partial decoupling from zoom (corr 0.99 → 0.77).
  The hit/shell gain (0.135 → 0.162) comes from there.
- Canon σ becomes state-dependent: median risen to 0.25, spread from 0.02 to 0.48.

## Update save_13 (351k, D=800)

CSV: `fire` 1.40 → 0.35 shots/s between 250k and 360k, `kill` 25.8 → 15.6, `r_hit` 0.0095 → 0.0035,
`hit` plateaus at 0.21-0.22. Clear switch at 260k (D 600 → 650).

| save | side | fire/s | p_al (<5° & <40 m) | p_fire <5° alone | p_off | hit/shell | h<5° | hits/s | turret<5° | median target dist. | canon σx |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 | 350 | 1.77 | 0.78 | 0.56 | 0.32 | 0.162 | 0.405 | 0.287 | 0.175 | 147 m | 0.246 |
| 13 | 350 | 0.57 | 0.57 | 0.33 | 0.030 | 0.245 | 0.497 | 0.139 | 0.077 | 294 m | 0.478 |
| 13 | 800 | 0.30 | 0.70 | 0.24 | 0.016 | 0.221 | 0.503 | 0.067 | 0.050 | 415 m | 0.491 |

- Off-axis extinguished whatever the reserve (0.012-0.035 from 0 to 50 %): no longer an
  end-of-reserve economy.
- At equal side (350), hits/s halved, tanks 2× farther, turret<5° halved,
  canon σ doubled: aim regresses.
- Empty-reserve penalty: −D/1000, treated as terminal (`cuts` includes `truncated`, no
  bootstrap). Median positive reward of a hit 0.30, mean 0.48. At D=800 the penalty is 0.8,
  i.e. 0.08 per shell of the initial stock, vs ~0.01 expectation for an off-axis shell (2 %
  hit): off-axis is deficit-making from D=250 (0.025 per shell).
- A shell at <5° hits 1 time in 2: expectation +0.24 and neutral reserve (2 refunded per hit). The
  penalty does not justify p_fire 0.24-0.33 in these states.
- Precedent: 428 (launched before `ee1ca624`, fixed penalty) closed its trigger the same way
  (fire 0.9 → 0.3, kill 22 → 15).

## Markers

- Re-probe save_6 to save_8 side 350 with the same regression: does the closure gain take off from
  5 % at 10-20°?
- `Hc` falls (−0.35) while the μ gain is flat: watch whether σ keeps tightening
  without μ following.

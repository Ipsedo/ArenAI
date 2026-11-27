# Run 550 — single advantage without Φ, empty penalty 0.25

`train_550_ppo_liquid`, ppo_liquid, launched 2026-11-22 21:49, **running**; read at 585 k (2026-11-23 04:26), 88 k steps/h. Code: `86edf2dc` + `65899e8a` (parameter clean-up, binary of 11-22 21:49 launched before the 22:26 commit). High confidence: `config.json` ≡ 549. Probes: saves 8, 12, 17, 21 side 550, save 21 side 900, binary `build/` of 11-22 13:14 with `--force` (the diff only touches Φ and probe exports, network identical). The probe does not apply the empty penalty.

## What the run tests

Against 549 (`dcca274e`), three **confounded** changes: (1) a single policy optimisation group: no separate aim credit, no group KL stop; (2) aim potential Φ₂ removed: pure event reward; (3) constant empty-reserve penalty 0.25 (instead of D/1000). It is the committed version of the other PC's Φ-less twin (432), which had the D/1000 penalty.

## CSV

Milestones: D 300 at 48 k, 550 at 208 k, 800 at 328 k, 900 at 404 k, **blocked at 900 for 181 k steps**.

| index | D | kill | hit | fire/s | Hc | Hd | kl | skip | ev |
|---|---|---|---|---|---|---|---|---|---|
| 200-225 k | 550 | 25.9 | 0.14 | 1.42 | −0.26 | 0.57 | 0.019 | 0.04 | 0.56 |
| 250-275 k | 650 | 22.2 | 0.20 | 0.52 | +0.14 | 0.42 | 0.010 | 0.02 | 0.63 |
| 375-400 k | 850 | 22.1 | 0.25 | 0.51 | −0.11 | 0.34 | 0.012 | 0.01 | 0.71 |
| 450-475 k | 900 | 13.4 | 0.29 | 0.26 | −0.01 | 0.33 | 0.007 | 0.00 | 0.47 |
| 550-585 k | 900 | 17.5 | 0.27 | 0.32 | −0.05 | 0.34 | 0.010 | 0.00 | 0.69 |

At equal index (550-600 k): 549 is at D 1000, kill 26-27, fire 0.9; 546 at D 1000, kill 21-23, fire 0.25-0.37. 550 is behind both, one D step lower.

## Probes

| save (index, D) | side | aligned close / off-axis p_fire | p_fire < 2° at 0-40 / 40-100 / 100-200 / > 200 m | hit/shell | disarmed | empty (median) | shots/s | touches/tank-s | tour < 5° | σ canon_x | mean tanh μ canon_x |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 (216 k, 550) | 550 | 0.71 / 0.21 | 0.71 / 0.67 / 0.35 / 0.29 | 0.15 | 26 % | 7.8 s | 1.35 | 0.21 | 17.6 % | 0.31 | +0.16 |
| 12 (324 k, 750) | 550 | 0.54 / 0.03 | 0.61 / 0.30 / 0.05 / 0.04 | 0.25 | 5 % | 38 s | 0.36 | 0.09 | 18.2 % | 0.47 | +0.28 |
| 17 (459 k, 900) | 550 | 0.32 / 0.01 | 0.35 / 0.29 / 0.08 / 0.02 | 0.34 | 2 % | 57 s | 0.28 | 0.10 | 15.7 % | 0.60 | +0.53 |
| 21 (567 k, 900) | 550 | 0.45 / 0.03 | 0.47 / 0.26 / 0.07 / 0.06 | 0.29 | 8 % | 37 s | 0.49 | 0.14 | 19.3 % | 0.58 | +0.42 |
| 21 | 900 | 0.54 / 0.02 | 0.50 / 0.28 / 0.04 / 0.02 | 0.26 | 5 % | 44 s | 0.24 | 0.06 | 17.5 % | 0.60 | +0.55 |

Turret command on a visible target at 5-30° (`probes/commande_tourelle.py`, plant 12°/unit):

| save | < 60 m directed | μ if target right / left | < 5° one second later | 60-100 m: median \|yaw\| after 1 s |
|---|---|---|---|---|
| 8 | 1.7°/dec | −0.07 / +0.30 | 31 % | 14 → 23° |
| 12 | 0.6 | +0.09 / +0.36 | 15 % | 17 → 46° |
| 17 | 1.4 | +0.05 / +0.56 | 19 % | 17 → 88° |
| 21 (550) | 2.4 | −0.06 / +0.53 | 26 % | 15 → 66° |
| 21 (900) | 2.7 | −0.07 / +0.61 | 30 % | 15 → 88° |

## Diagnosis

Established:
- KL regime: settled. kl 0.007-0.02 for a 0.05 target, skip < 4 % then ~0, clip 0.10 (549: kl 0.03-0.15, clip 0.2-0.45).
- Dump: settled. Disarmed 26 % → 2-8 %, 1st empty 7.8 s → 37-57 s, from save 12.
- Trigger selective in angle, closed in distance. Off-axis 0.01-0.03; aligned < 2°: 0.5 at < 40 m, 0.3 at 40-100 m, 0.02-0.07 beyond. Closing between 225 and 275 k (D 550-650, fire 1.4 → 0.5), same shape as 539 and 432. Shots at > 100 m hit 7-10 %, under the break-even (~9 %): the closing at range is consistent with real precision.
- Turret in continuous sweep, which sets in. Mean tanh μ canon_x +0.16 → +0.55, μ > 0 on 88-90 % of decisions at saves 17-21 (~6°/decision). The record "directed" command at < 60 m (2.4-2.7°/dec vs 0.3-1.1 in all previous runs) is a one-sided brake: μ ≈ 0 when the target is to the right, +0.6 when to the left. Beyond 60 m the turret crosses the target (15° → 66-88° in 1 s). Yaw still explains only 4-8 % of var(μ).
- σ canon_x rises to 0.60 (SoftClamp ceiling 0.7) while the six α are at floor 1e-4 since 285-376 k: not the entropy bonus, the policy gradient. The −1 target is inert (Hc ≈ 0).
- Elevation unlearned: corr site/μ canon_y +0.23 → −0.18 (549 with Φ₂: +0.57 to +0.67).
- Touch volume halved side 550 between save 8 and save 12 (0.21 → 0.09 touch/tank-s), back to 0.14 at save 21. Curriculum stays in the criterion's dead band (kills/tank between 0.3 and 0.5 at the boundary).
In the noise: tour < 5° 16-19 % at all saves; "< 5° in 1 s" 15-31 %, within the 19-33 % band of all runs; the kill rebound 13.4 → 17.5 (475 → 585 k): a single oscillation, of the order of 546's.
Not settled: whether sweep + reflex shot at < 100 m is a fixed point or a phase (touches/tank-s and < 60 m command rise from save 12 to save 21, but so does the rotation bias); the share of each of the three confounded changes (432, penalty D/1000, had the same shape up to 334 k: the 0.25 penalty does not seem at fault).

## Reading by problem

| problem | state in 550 |
|---|---|
| KL regime / skip | gone |
| reserve dump | gone |
| σ canon collapsed at start | absent (0.31 at 216 k), inverted: σ drifts toward ceiling |
| trigger closing at range | present, identical to 539/432 |
| trigger oscillating | fire 0.15-0.47 per 10 k slice at D 900, as 546 |
| aim holding | not learned: sweep, no pursuit; elevation lost without Φ₂ |
| plateau | D 900 not crossed, 181 k steps |

Overall: without Φ the single advantage correctly learns **when to fire** (selectivity, shell economy) but not **how to bring the cannon**; it compensates with a sweep that passes the target in front of a reflex short-range shot. Same bottleneck as "critic blind to the aiming gesture", unchanged by bootstrapped GAE, −1 target and batch 2400.
Options (no decision): let run to ~800 k and probe saves 25 and 29 (fixed point vs phase); stop (run already answered on KL, dump and selectivity; turret shows no sign of pursuit); diagnosis without new run: offline GAE on `save_21_side900_rewards.csv` by sign of turret action, to measure whether the sweep receives positive advantage by itself (acquiring out-of-field targets) or only for lack of credit for stopping.
Markers: fixed point: mean tanh μ canon_x ≥ +0.5, |yaw| at 60-100 m still > 60° after 1 s, p_fire < 2° at 100-200 m ≤ 0.08; phase: brake on both sides (μ of opposite sign by side of target), "< 5° in 1 s" > 33 % at < 60 m, touches/tank-s side 550 > 0.21 (save 8's level); CSV: passage to D 950 (kills/tank > 0.5 at boundary), fire > 0.5/s held at hit ≥ 0.25.

## Origin of the sweep (2026-11-23, save 21 side 900, `probes/balayage.py`)

GAE recomputed offline (γ 0.998, λ 0.95, return std 1.525 read in `return_scaler.pt`), normalised advantage.
The sweep is target search, not a credit defect.
- Camera is on the cannon: horizontal field 79° (40° zoomed, p_zoom 0.56-0.80), i.e. 11-22 % of the horizon. Only other cue: direction of last shot received.
- An enemy at < 100 m is out of field 68 % of the time it exists (29.5 % of decisions out of field, 13.7 % in view); 57 % of the time no enemy at < 100 m.
- Rotation: median 10.9°/decision (≈ 110°/s, 60 % of max speed), > 4°/dec 82 % of time, same direction in 97 % of tank-episodes. At save 8: 3.6°/dec, 46 % of time.
- 96 % of entries into view of a target < 100 m occur with the turret rotating > 4°/dec (7 entries per tank-minute).
- Out of field, the slope of advantage on the turret action is null: −0.02 to +0.05 (±0.01) at ≥ 60 m. Nothing pushes to turn nor to stop: the policy is at a stationary point.
In view the signal is right but exists only at short range:

| target in view | slope dA/da, target behind rotation direction | median \|Δturret\| | hit of aligned shot < 2° |
|---|---|---|---|
| < 60 m | −1.35 ± 0.08 | 5.3°/dec (3.4 aligned) | 0.69-0.95 |
| 60-100 m | −0.22 ± 0.02 | 9.3 | 0.50 |
| 100-200 m | +0.03 ± 0.01 | 11.4 | 0.15-0.40 (n 144) |

- The turret does brake when the close target has just been passed (μ ≈ 0) and continues when it is ahead (μ +0.6): a one-way scanner, not a broken brake.
- Passage of a < 100 m target through the field: median 0.6 s, 31 % ≤ 0.5 s. 40 % of touches leave with turret > 4°/dec (hit 22 % vs 29 % at rest).
- Beyond 100 m, even an aligned shot < 2° hits 15-40 % (shell 250 m/s: drop 1.8 m and 0.6 s flight at 150 m, elevation not learned): stopping on a far target earns nothing, hence no advantage teaches it.
Causal chain: narrow field tied to cannon ⟹ searching = turning; far shot unprofitable ⟹ only targets < 100 m are worth a stop; 0.6 s pass ⟹ reflex shot. The three hold together: it is an optimum of the problem as posed, not an optimiser failure.
Reserve: control 548 (target position given as input, Φ₂ regime) had not improved the directed command; information alone is therefore not proved sufficient for pursuit.

## Oracles and deterministic mode (2026-11-23, save 21 side 900, `probes/oracle_hausse.py`)

Probe `--oracle --oracle_fov 79`: servo on the nearest enemy in field, elevation corrected for drop, no lead on moving target.

| control | touches/tank-s | shots/tank-s | hit < 40 m | 40-70 | 70-100 | 100-150 | 150-200 | > 200 |
|---|---|---|---|---|---|---|---|---|
| policy | 0.063 | 0.24 | 0.80 | 0.58 | 0.19 | 0.08 | 0.12 | 0.07 |
| oracle elevation (yaw + policy trigger) | 0.073 | 0.30 | 0.88 | 0.65 | 0.31 | 0.10 | 0.03 | 0.06 |
| oracle yaw (elevation + policy trigger) | 0.146 | 0.52 | 0.99 | 0.84 | 0.63 | 0.38 | 0.22 | 0.08 |
| full oracle | 0.629 | 2.22 | 0.93 | 0.80 | 0.62 | 0.54 | 0.28 | 0.15 |

- Elevation is not the bottleneck: +16 % touches. "Ballistic fire control" lead dismissed.
- Yaw is: ×2.3 with elevation and policy trigger unchanged; the trigger opens by itself at range as soon as yaw is held (0.38 hit at 100-150 m).
- Physical ceiling: perfect aim without lead = 0.28 at 150-200 m, 0.15 beyond (median miss 50-100 m). Beyond 150 m firing is of little profit even without learning defect.
Deterministic mode (μ alone): median dwell < 5° 0.2 s (p90 0.5 s) at < 60 m, identical to stochastic; one second after entry, < 5° in 37 % of cases (40 % stochastic). Touches 0.046/tank-s. **Target holding does not exist in μ**: σ canon_x 0.6 is not the cause. The turret's angular velocity is in proprioception: the holding defect does not come from an unobserved rotation.

## Turret command chain, read in code and measured (2026-11-23)

Scripts `probes/reponse.py`, `probes/reponse_cellule.py`, `probes/boucle.py`. Save 21 side 900.
"0.6 s pass": duration during which a target at < 100 m stays in the camera field (±39.5°) while the turret sweeps. Median 0.6 s = 6 decisions.
Correction: the "slope dA/da −1.35 at < 60 m" of the previous section is an artefact. It regresses the advantage on the raw action, which depends on the state; on exploration noise alone (what the policy gradient sees), ρ(advantage, noise toward target) = +0.001 at < 60 m, −0.004 at 60-100 m (λ 0.95); +0.02 at λ 0.5; +0.01 to +0.03 at λ 0. **The gradient contains no information on the turret gesture**, as in all previous measures (≤ 0.03).
Chain stage by stage:
- Action → turret (`turret.cpp`, `gun_laying.cpp`): velocity command on a world-stabilised direction, 180°/s per unit. Measured: Δyaw = 9.3·a(t) + 9.0·a(t−1), R² 0.98; exogenous disturbance 0.4°/decision.
- Actor (`liquid_actor.cpp`, `liquid_cell.cpp`): μ, σ, fire heads plugged on cell state only, no direct input → output path. Cell τ ≡ initialisation (0.10-2.0 s, median 0.43) from save 0 to save 21. Step response simulated on weights (random inputs): μ canon_x reaches 48 % / 76 % / 89 % of the change in 1 / 2 / 3 decisions; p_fire 39 / 67 / 82 %.
- Real response (deterministic, 463 entries sweeping a target < 60 m): μ stays at +0.57 until the decision where the target passes the center, falls to +0.43 then +0.35 on the next two (target already at +14° then +25°), rises to +0.43. p_fire at passage: 0.04.
- σ canon_x 0.6. Holding simulation with measured plant and best gain:

| observation → action delay | σ 0.05 | σ 0.15 | σ 0.30 | σ 0.60 |
|---|---|---|---|---|
| 0 decision: time at < 2° / < 5° | 88 / 100 % | 55 / 94 % | 31 / 69 % | 19 / 46 % |
| 1 | 78 / 100 | 35 / 75 | 19 / 45 | 11 / 28 |
| 2 | 67 / 99 | 27 / 63 | 16 / 37 | 9 / 22 |

Reading: no bug in the chain. Three facts add up: (1) the event gradient does not carry the gesture (ρ ≈ 0); (2) at σ 0.6 holding is impossible even for a perfect regulator, so its value never appears in the data; (3) the cell adds ~1.5 decisions of delay, which costs 20-30 points of holding at σ 0.15.

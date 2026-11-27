# What a turret gesture changes, and what the events say about it

Runs: 528, 529, 533-538 (537 s3/s4, 538 s3/s8/s12, 528 s9, 529 s3/s9).
Date: 2026-11-19. Frozen policies, no training code modified. Exports `arenai_agent_sigma_gradients` (12 episodes): 537 s3 and s4 (side 350, coloured noise ρ 0.9, σ canon 0.17), and those of 2026-11-18 for 528 s9 (white noise, σ 0.57), 529 s9 and s3 (white noise, σ 0.23). Scripts and data: `outputs/comparisons/2026-11-19_bruit_avantage_visee/` (`lagprofile.py`, `drift.py`, `command.py`, `counterfactual.py`, `detrend.py`, `hhead.py`, `hwhy.py`, `hlag.py`; outputs `outputs/comparisons/2026-11-19_bruit_avantage_visee/data/`, e.g. `data/lag_*.txt`, `data/detrend_538_s8_side*.txt`, `data/hhead_538_s12*.txt`, `data/frames_538_s12.png`, `data/hfix_538_s12_*`). Probes 533-537 for part 4.

"Gesture" = normalised exploration noise of canon_x, signed toward the nearest target. ρ = per-row correlation; standard error by bootstrap over tank-episodes.

## 1. Yaw moves because the turret commands itself, not because the target drifts

Enriched export (positions, cannon and chassis headings; `drift.py`, `command.py`), 537 s3, 6 episodes, same nearest target < 100 m, |yaw| < 40°. The turret responds over two decisions (−8.8° then −8.6° per action unit, R² 0.92 of cannon heading motion).

Yaw variation, variance share:

| component | median per decision | at 0.1 s | at 1 s | at 0.1 s without the 10 % extremes |
|---|---|---|---|---|
| turret command (actions t and t−1) | 2.3° | 51 % | 61 % | 89 % |
| own translation (parallax) | 0.18° | 20 % | 16 % | 6 % |
| target displacement | 0.11° | 20 % | 11 % | 0 % |
| stabiliser, inertia | 0.13° | 4 % | 3 % | 2 % |
| unreconstructed | 0.24° | 5 % | 8 % | 2 % |

Target bearing drifts 0.18° per decision (median), 1.6° in 1 s. The turret moves 2.2° per decision. **A first version of this report said the opposite ("85 % of the movement does not come from the command"): the regression used only the current decision's action and ignored the previous decision's. That was wrong.**

Composition of the command (° of cannon heading per decision, median / p90):

| | per decision | cumulative over 1 s |
|---|---|---|
| share of μ | 2.7° / 6.5° | 16.4° / 44.6° |
| share of exploration noise | 1.5° / 4.2° | 13.2° / 36.3° |
| μ + noise (corr −0.48 per decision, −0.65 over 1 s) | 2.2° / 5.9° | 8.9° / 33.3° |
| bearing drift | 0.18° / 0.75° | 1.6° / 7.4° |

What μ commands as a function of yaw: 0.1° toward the target at 0-2°, 0.2° at 2-5°, 0.5° at 5-10°, 0.8° at 10-40°, nothing beyond 40°; equilibrium point shifted to +4°. The norm of μ's command is 3.2° per decision whatever the yaw. What explains tanh(μ): yaw 4 %, noise of the 5 past decisions 18 %, previous μ brings the cumulative to 44 %, bearing drift and chassis rotation 0 %. The rest (55 %) changes from one decision to the next with no link to geometry.

## 1 bis. Replay: the target-linked part of μ, without the rest

`counterfactual.py`: yaw replayed in 1D on the real segments (same nearest target, median duration 2.6 s), with the real bearing drift and, as the only command, the mean setpoint of μ as a function of yaw. Drift assumed unchanged (approximation).

| | yaw < 5° | yaw < 2° | dwell < 5° |
|---|---|---|---|
| real | 6.8 % | 2.7 % | 0.31 s |
| no command | 3.5 % | 1.4 % | 1.90 s |
| yaw-linked part of μ alone | 15.2 % | 4.2 % | 2.77 s |
| same, setpoint ×2 | 22.9 % | 3.9 % | 3.00 s |

Median |yaw| toward the nearest at < 100 m: 81° (the target is out of field most of the time, and the setpoint is zero beyond 40°).

## 2. The effect of a gesture fades in under a second

ρ(gesture, future state), yaw 1-40°, target < 100 m:

| | 528 s9 | 529 s9 | 529 s3 | 537 s3 | 537 s4 |
|---|---|---|---|---|---|
| drop of \|yaw\| at 0.1 s | 0.142 | 0.075 | 0.066 | 0.047 | 0.038 |
| at 0.5 s | 0.051 | 0.031 | 0.040 | 0.027 | 0.019 |
| at 1 s | 0.025 | 0.012 | 0.011 | 0.010 | 0.003 |
| aligned < 5° at 1 s | +0.014 | +0.008 | +0.016 | −0.015 | −0.022 |

Standard errors 0.003-0.008. In the 1-10° band the effect is null from 0.5 s everywhere.

## 3. Events contain almost nothing about the gesture, whatever the slicing

ρ(credit signal, gesture), yaw 1-40°, < 100 m:

| signal | 528 s9 | 529 s9 | 529 s3 | 537 s3 | 537 s4 |
|---|---|---|---|---|---|
| run advantage (λ 0.95) | 0.006 | 0.012 | – | 0.002 | −0.004 |
| 30 s return without critic | 0.003 | 0.009 | 0.007 | −0.016 | −0.005 |
| sum of rewards 0-1.2 s | 0.012 | 0.008 | 0.017 | −0.009 | 0.003 |
| sum 0-0.4 s | 0.013 | 0.010 | 0.017 | −0.006 | 0.008 |
| optimal linear filter per lag slice (out of sample) | 0.013 | 0.003 | 0.017 | −0.008 | −0.006 |
| standard error | 0.004 | 0.003 | 0.005 | 0.009 | 0.009 |

- Best band: 10-40° in 528 s9, 0.024-0.027. Band 1-10°: indistinguishable from 0 everywhere.
- Measured ceiling of credit from events alone: ρ ≈ 0.01-0.03, i.e. 6 to 30 rollouts for a gradient to come out of noise (t = 2), if the policy does not move meanwhile.
- References of 2026-11-12 and 2026-11-18: same-step Φ 0.10-0.13; geometric critic at λ 0.3 0.076. All signals that made the turret learn are state differences, not events.
- Coloured noise: no signal gain, and the standard error doubles at equal row count (plan effect 6-10: 6 to 10 times fewer independent samples).

Variance composition of the return seen from these states (537 s3 ≡ 528 s9): hits 81-85 %, kills 11-13 %, damage received + death 5-7 %; by lag: 76-80 % comes from rewards beyond 2 s, 10 % from 0-1.2 s. The noise is not that of the other tanks: it is the tank's own future hits, unrelated to the present gesture.

## 4. The policy cancels its own coloured noise

corr(μ canon_x at t, noise at t−1) and corr of 1 s sums (probes, side 250):

| | at 0.1 s | 1 s sums |
|---|---|---|
| 533 s0 (untrained) | −0.01 | −0.01 |
| 533 s1 / s3 | −0.25 / −0.25 | −0.31 / −0.40 |
| 534 s3 | −0.29 | −0.49 |
| 535 s5 | −0.28 | −0.39 |
| 537 s1 / s3 | −0.39 / −0.43 | −0.51 / −0.59 |
| 528 s9, 529 s9 (white noise, exports) | −0.03 / −0.01 | – |

- Learned from 27 k in all coloured-noise runs; 10 to 28 % of μ's variance is explained by the past second's noise.
- The past second's noise explains nothing of the current yaw (R² 0.000).
- The "corrector" of probes 533/534 is not an artefact of this compensation: 0.624 → 0.620 once the part of μ linked to past noise is removed.

## 5. Same decomposition in white noise (538 s3, same code, 12 episodes)

| | 537 s3 (coloured) | 538 s3 (white) |
|---|---|---|
| bearing drift, median per decision | 0.18° | 0.16° |
| turret movement (μ + noise), median per decision | 2.2° | 2.7° |
| share of μ / share of noise | 2.7° / 1.5° | 2.6° / 0.9° |
| corr(μ command, noise command) per decision / over 1 s | −0.48 / −0.65 | −0.04 / −0.13 |
| cumulative μ over 1 s, median / p90 | 16° / 45° | 13° / 55° |
| autocorr of μ at 0.1 s | 0.62 | 0.87 |
| μ explained by: yaw / past noise / + previous μ | 4 % / 18 % / 44 % | 7 % / 2 % / 77 % |
| μ toward target at 2-5° / 5-10° / 10-20° (° per decision) | 0.24 / 0.54 / 0.76 | 0.26 / 0.50 / 0.70 |
| mean μ setpoint at yaw ±1° (° per decision) | −0.60 / −0.30 | −1.23 / −0.93 |
| replay, yaw-linked part of μ alone: yaw < 5° / dwell | 15.2 % / 2.77 s | 6.2 % / 0.87 s |
| real: yaw < 5° / dwell | 6.8 % / 0.31 s | 6.7 % / 0.33 s |
| ρ(events, gesture): sum 0-0.4 s / optimal filter | −0.006 / −0.008 | +0.017 / +0.024 |
| ρ(run advantage, gesture) | +0.002 | +0.009 |

- In white noise the turret moves as much (2.7° per decision for a target drifting 0.16°), but differently: μ is smooth and carries a constant bias of −0.9 to −2.4° per decision at any yaw, i.e. a one-direction sweep of about 13° per second, on which the same corrector of 0.3 to 0.7° per decision is superposed.
- In coloured noise μ is choppy and partly serves to cancel the noise; the sweep bias is weaker (−0.3 to −0.8°).
- In both cases dwell on target is 0.3 s and the nearest enemy is 81° off-axis at the median.

## 6. Inference-time intervention: sweep removed (538 s8, 216 k, 6 episodes per arm)

Probe `--detrend_canon_x_steps N`: the sliding mean (N decisions) of μ canon_x is subtracted before the action draw, everything else is the policy. Script `detrend.py`, outputs `data/detrend_538_s8_side*.txt`.

| side 250 / 550 | policy | removal 1.5 s | removal 0.5 s |
|---|---|---|---|
| kills per episode | 25.0 / 22.5 | 25.3 / 19.0 | 23.0 / 19.0 |
| hits per episode | 491 / 432 | 446 / 381 | 394 / 370 |
| hit/shell | 19.3 / 16.7 % | 18.6 / 15.2 % | 17.3 / 14.3 % |
| dwell < 5° | 0.31 / 0.30 s | 0.39 / 0.39 s | 0.49 / 0.50 s |
| dwells ≥ 1 s | 2.1 / 2.8 % | 5.6 / 6.0 % | 11.4 / 10.8 % |
| passes < 5° per tank-episode | 9.4 / 16.2 | 7.4 / 12.3 | 4.9 / 8.1 |
| yaw < 5° (target < 100 m) | 6.4 / 7.9 % | 6.1 / 6.5 % | 4.1 / 4.9 % |
| median \|yaw\| (target < 100 m) | 82° / 75° | 89° / 93° | 89° / 92° |

Std of kills between episodes: 1.3 to 3.0 (standard error ≈ 1 per arm).

- Removing the sweep lengthens dwell (0.3 → 0.5 s) but halves the passes: aligned time drops, hits drop 9 to 20 %, kills rise in no arm.
- The sweep is thus the policy's acquisition means, not a defect to remove. The "penalty on μ canon" lead is not supported.
- Camera field is ±39.5°: the zero setpoint beyond 40° corresponds to a target out of image. What is missing is the gain on a visible target (0.5 to 0.8° per decision at 5-40°, out of 9.2° possible).
- Limits: policy trained with its sweep (trigger and corrector adapted to it); the filter also removes the slow part of a sustained correction; one checkpoint, 6 episodes.

## 7. Shot value learned from shot outcomes (lead B), offline on 538 s12

Export `arenai_agent_sigma_gradients` (columns `fired_shells`, `landed_shells`, `nb_hits`, option `--frames_fired`), side 550: 36 episodes for fitting (36 263 labelled shots, 23.6 % hits), 6 never-seen episodes for test (all images, 5 705 shots). H(image) = share of shells fired at that decision that hit; the run's 7-block CNN, from scratch, no proprio nor memory. Script `hhead.py`, outputs `data/hhead_538_s12*.txt`. Criterion fixed before the measurement: ρ(γH(s′) − H(s), gesture) ≥ 0.08 at one step, yaw 1-40°, target < 100 m.

| | at 0.1 s | at 0.2 s | at 0.3 s | AUC hit / miss (test) |
|---|---|---|---|---|
| H read from geometry (upper bound) | +0.147 | +0.146 | +0.095 | 0.865 |
| H learned from image, 36 episodes | −0.001 | +0.056 | +0.052 | 0.749 |
| same, 18 episodes | +0.002 | +0.057 | +0.051 | 0.709 |
| same, 9 episodes | −0.002 | +0.055 | +0.050 | 0.684 |
| state reference, −\|yaw\| | +0.197 | +0.216 | +0.143 | – |

Standard errors 0.004 to 0.008. Band 100-300 m: image H ≈ 0, bound 0.03.

- The label carries the signal: the geometric bound (0.15) exceeds Φ (0.10-0.13 on 12 and 16/11).
- The image returns only a part: null at one step, 0.056 at two steps, below the criterion. Hit / miss separation improves with data (0.68 → 0.75), not the gesture credit.
- Image H by yaw (< 100 m): 0.26 at 0-2°, 0.22 at 5-10°, 0.15 at 10-20°, 0.07 at 20-40°; actual shot outcome: 0.54 / 0.39 / 0.30 / 0.16.
- Overfitting from the 6th pass at 36 episodes (measurement taken at the 5th).
- Unexplained: why the effect is null at one step in the image and entire in geometry.
- Limits: 36 episodes = one eighth of what a run sees up to D 650; one seed; frozen policy; the precedent of 532 (positive offline, negative online).

## 8. Why the image returns less than geometry (538 s12, `hwhy.py`, `hlag.py`)

Real images: `outputs/comparisons/2026-11-19_bruit_avantage_visee/data/frames_538_s12.png` (rows: aligned hit, aligned miss, yaw 10-20°, aligned at 80-100 m). Every part of every tank, the terrain and the decor objects have a random colour per episode; no reticle; target ~10 px wide at 80-100 m.

**The image of a decision is that of the previous decision.** `VulkanOffscreenRenderer::draw_and_get_frame` submits the current render then reads the other slot (`FRAME_SLOTS` = 2), i.e. the render of the previous call; training renders once per decision (`render_frame` after the `action_repeat` frames). Present since the Vulkan migration (0dc62306, 18/07). Measurement: the variation of image H over one step tracks the geometry variation of the previous step (corr 0.148) more than that of the same step (0.102), then 0.01-0.02 elsewhere.

| test | result |
|---|---|
| already-seen colours (same episodes, later) versus never seen | AUC 0.730 versus 0.688 (geometry 0.864 / 0.863) |
| centre 64×128 full resolution | AUC 0.767, ρ at 0.2 s 0.060 |
| centre 64×64 | AUC 0.737, ρ at 0.2 s 0.065 |
| vertical band 128×64 | AUC 0.722, ρ at 0.2 s 0.053 |
| geometry read 0 / 1 / 2 decisions before the shot | AUC 0.863 / 0.856 / 0.850 |
| volume: 9 / 18 / 36 episodes of shots | AUC 0.684 / 0.709 / 0.749 |

- The lag explains the null one-step credit (the effect of a gesture appears in the image at 0.2 s), not the level: geometry delayed by one decision recognises hits almost as well.
- Neither colours, framing nor resolution close the 0.75 / 0.86 gap; only volume reduces it (+0.03 to +0.04 per doubling).
- Consequence of the lag for any value read from the image: the one-step difference does not contain the action just taken. Consistent with the critic's ρ ≈ 0 at λ 0 (16 and 18/11).
- Not measured: what the lag costs in hits and kills; the effect of its correction on training throughput; whether the game (rendered every frame) has the same lag in decisions.

## 9. Image lag fixed: measurement of B redone (538 s12, same protocol and volumes)

Fix (validated by Samuel): `draw_and_get_frame` reads the slot just submitted. Three view tests expected a first black image, assertions removed; view 58, core 28, agent 2 442 tests passed in `build/`. New exports `data/hfix_538_s12_*`, outputs `data/hhead_538_s12_fix.txt`, `data/hwhy_lag_fix.txt`. The policy is 538's, trained with the lag, replayed without.

| | lagged image (§7) | up-to-date image |
|---|---|---|
| ρ(γH(s′) − H(s), gesture), H learned from image, at 0.1 s | −0.001 ± 0.004 | **+0.100 ± 0.008** |
| same at 0.2 s / 0.3 s | +0.056 / +0.052 | +0.071 / +0.037 |
| same at 0.1 s, yaw 1-10° / 10-40° | +0.006 / −0.008 | +0.139 / +0.058 |
| same at 0.1 s, target at 100-300 m | −0.008 | +0.011 |
| geometric bound at 0.1 s | +0.147 | +0.170 |
| AUC hit / miss, image (5th pass) / geometry | 0.749 / 0.865 | 0.797 / 0.893 |
| labelled shots fit / test | 36 263 / 5 705 | 40 522 / 6 830 |
| image H ↔ geometry variation of same step / previous step | 0.102 / 0.148 | 0.253 / −0.046 |
| export duration of 36 episodes | 23 min 30 | 25 min 10 |

- The criterion fixed before the measurement (0.08 at one step) is reached: 0.100, at Φ's level (0.10-0.13).
- The lag is gone: image variation tracks the same step's geometry.
- Nothing beyond 100 m, as for Φ out of field.
- Limits: one seed; offline, frozen policy; precedent of 532; the effect of the fix alone on a training (critic, trigger, kills) is not measured; CI reference images not regenerated.

## Reading (Claude's)

- The turret learned a weak but real corrector (0.2 to 0.8° per decision toward the target). It is drowned in a command 4 to 15 times larger, unrelated to the target, part of which serves to cancel the coloured noise. Dwell on target (0.31 s) is set by this agitation, not by target movement (1.90 s with the turret still).
- An exploration gesture weighs 1.5° in a proper movement of 2.2° per decision, hence its weak trace on the state (part 2) and its near-null trace on events (part 3).
- No tuning of event credit (λ, γ, window, per-component head, batch) can exceed ρ ≈ 0.02: consistent with the failures of 526, 530, 531, 532.
- Coloured noise was meant to make the gesture last; the policy cancels it and the effect at 1 s did not move (0.010 versus 0.012).
- Two distinct limits appear: μ agitation (dwell) and the absence of setpoint beyond 40° (acquisition).

## Not established

- Where the part of μ unrelated to geometry (55 %) comes from: not measured. Untested hypothesis: random walk of the μ head under a gradient at ρ ≈ 0.01 (growing `‖mu.weight‖`, measured 2026-11-19 on 523-535, see `2026-11-18_skip-kl-rise-507-535.md`).
- What a strong gain on a visible target would bring, sweep kept out of field: not measured.
- Whether noise compensation explains the KL regime of coloured-noise runs.
- None of this is tested by intervention in training; the 1D replay is not a game simulation.

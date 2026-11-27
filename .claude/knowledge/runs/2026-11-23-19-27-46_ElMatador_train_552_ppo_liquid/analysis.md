# Run 552 — presence head on the CNNs (ppo_liquid)

## Header

`outputs/train_552_ppo_liquid`, ppo_liquid, launched 2026-11-23 19:27 (mtime `config.json`), read at 23:03: **running**, index 278 k, D 600 (650 reached at 233 k, back to 600 around 265 k). Throughput 77.3 k steps/h = 2.47 M tank-steps/h (550: 84.7 k, 551: 85.7 k) ⟹ the head costs ≈ −9 %. Effective code: `db9a825a` (presence head) **without the `pres` metric** — binary `cmake-build-release` compiled 19:27 = launch, commit 20:41, `pres` column absent from CSV. The `actor_presence_optim.pt` / `critic_presence_optim.pt` files of the saves prove presence learning runs. High confidence.

## What the run tests

Single change vs 550 (and 551 without the direct branch): auxiliary presence head, 8×16 map after block 4, on the actor and critic CNNs, separate Adam ($\eta = 3\cdot10^{-4}$), one pass per rollout after the PPO updates. Loss:
$$\mathcal{L}_{\text{pres}} = -\frac{1}{|B|\,HW}\sum_{b\in B}\sum_{ij}\Big[y_{bij}\log\sigma(z_{bij}) + (1-y_{bij})\log(1-\sigma(z_{bij}))\Big]$$
$y_{bij}\in\{0,1\}$ = cell covered by a living visible enemy, $z_{bij}$ = logit of the 1×1 convolution on block-4 features. The gradient crosses blocks 1-4, shared with PPO. Reward, economy, curriculum unchanged (diff `train_environment.cpp` = probe geometry only). Not confounded. Hypothesis (analysis 2026-11-23, `outputs/comparisons/2026-11-23_origine_balayage`): half of the < 60 m passes are ignored because the CNN does not extract the target; the head must raise the share of engaged passes (reference 551 s20 side 900: 43 % net < 60 m, 20 % at 60-100 m).

## CSV at equal D

D milestones identical to 551 up to 650 (21 / 48 / 67 / 114 / 145 / 181 / 208 / 233 k).

| D | kill 552 / 551 / 550 | fire 552 / 551 / 550 | hit 552 / 551 / 550 | miss (m) 552 / 551 / 550 |
|---|---|---|---|---|
| 400 | 22.3 / 23.1 / 25.0 | 1.38 / 1.61 / 1.57 | 0.108 / 0.109 / 0.141 | 38 / 40 / 43 |
| 500 | 18.7 / 24.3 / 24.8 | 0.71 / 1.55 / 1.54 | 0.137 / 0.123 / 0.138 | 37 / 54 / 51 |
| 600 | 19.1 / 26.0 / 23.7 | 0.47 / 1.20 / 0.69 | 0.203 / 0.152 / 0.178 | 40 / 62 / 52 |
| 650 | 17.0 / 25.0 / 20.7 | 0.47 / 1.02 / 0.41 | 0.194 / 0.166 / 0.215 | 37 / 59 / 49 |

- The trigger rarefies from D 450-500 (fire 1.4 → 0.7 between 114 k and 180 k), vs D 600-900 for 550/551. Kill −20 to −25 % at D 500-650, outside the 550/551 gap.
- End of reading: kill 22 (225 k) → 14 (273 k), D 650 → 600. Not an extinction (fire 0.4-0.7, hit 0.18-0.20 stable).
- Healthy optimisation: kl 0.009-0.015, skip < 3 %, clip 0.11-0.15, ev 0.5-0.8 (isolated negative spikes at 208 k, 221 k, 227 k: single rows). The six α at floor 1e-4 from ~230 k except canon_x (1.2e-4) and fire (2.1e-4); Hc −0.33.

## Probes (binary `build/` 20:13, `--force`: same network, diff = probe code)

Probe save_10 (270 k, D 600), 4 episodes, sides 550 and 900; saves 3/6/8 side 550, 2 episodes. GPU shared with training. Script `probes/site_et_reaction.py` (reaction independent of sweep sign: `reaction.py` assumes μ > 0 and is wrong here).

| | median canon elev. | median target site (< 100 m, ±20°) | share \|site\| < 6° | net < 60 m | net < 60 m, \|site\| < 6° | tour<5° |
|---|---|---|---|---|---|---|
| 552 s3 | −7.6° | +4.0° | 0.61 | 0.14 | 0.13 (n 52) | 0.133 |
| 552 s6 | −17.7° | +13.3° | 0.20 | 0.16 | 0.55 (n 82) | 0.015 |
| 552 s8 | −9.3° | +6.0° | 0.47 | 0.14 | 0.23 (n 143) | 0.065 |
| 552 s10 side 550 | −15.1° | +10.6° | 0.34 | 0.08 | 0.29 (n 115) | 0.032 |
| 552 s10 side 900 | −16.7° | +12.9° | 0.28 | 0.06 | 0.17 (n 90) | 0.021 |
| 550 s8 | −3.4° | +1.3° | 0.70 | — | — | — |
| 550 s12 | −0.4° | −1.2° | 0.70 | 0.19 | 0.31 (n 200) | — |
| 550 s17 | −4.5° | +1.6° | 0.69 | — | — | — |
| 550 s21 side 900 | −1.4° | +0.2° | 0.76 | 0.46 | 0.61 (n 231) | 0.16-0.19 |
| 551 s20 side 900 | −2.6° | −0.1° | 0.76 | 0.43 | 0.54 (n 379) | — |

Other save_10 facts: reversed sweep (tanh μ canon_x −0.6, same sign 98-99 % of the time); selective trigger (p_fire 0.66-0.74 aligned < 5° & < 40 m, 0.005-0.009 off-axis); hit/shell 0.25, median miss 10-11 m; disarmed 3-4 %; σ canon_x 0.32. Vertical field $\pm 22.5°$ (fovy $45°$), zoomed $\pm\arctan(\tan 22.5° / 2) = \pm 11.7°$; with $p_{\text{zoom}} \approx 0.7$ and a median site of $+11$ to $+13°$ the target is out of the zoomed image about once in two.

## Diagnosis

Established: 552's canon is held low (median elevation −8 to −18° from s3 to s10) vs −0.4 to −4.5° on 550: the near target is above the axis, often out of the zoomed image. Share of passes at good site (|site| < 6°) 0.20-0.47 vs 0.69-0.76. Direct consequence: tour<5° 1.5-6.5 % (550 s21: 16-19 %), few engaged passes, fire and kill lower at CSV, shorter miss (near shots only).
In the noise / not settled: at good site, 552 s10's < 60 m engagement (0.29, n 115) ≡ 550 s12 (0.31) at neighbouring index: **no trace of a head gain** on engagement, but small n and 2-episode probes for s3-s8 (0.13 → 0.55 → 0.23, noise). Cause of the low canon: not established; without Φ elevation has no gesture gradient (550 memory: "elevation unlearned"); 550 s17 already had p10 −21.7°. One seed: seed drift or head effect, undecidable here. That the head learned the map is not measured (no `pres` column).
Options (no decision): (1) let run, probe save_13 and save_16 side 550 (2 ep.) + `probes/site_et_reaction.py`; marker: median elevation > −5° and share |site| < 6° ≥ 0.6 ⟹ low canon was transient, then compare engagement to 550 s21 / 551 s20 (0.54-0.61 at good site); stays < −8° ⟹ the run cannot answer the head question. (2) Measure the head itself on save_10: probe `--presence --sightings`, detection AUC by distance slice vs offline bench (0.91 / 0.97 / 0.96 at < 60 / 60-100 / 100-200 m); cost ~5 min; says if the map is learned online, not whether PPO uses it. (3) Second seed of the same code (G4: tactical training, to validate): canon median < −8° on both seeds at s6-s10 ⟹ head effect; otherwise seed.

## Learned presence map: save_10 (measure of 2026-11-23, 23:20)

Probe save_10 side 900, 1 episode policy zoom + 1 episode forced zoom, all decisions (`probes/data/pres_s10_side900_{policy,on}.*`, ~10 GB): images, exact training target map (`--presence`), projections (`--sightings`). Prediction recomputed in Python from `actor.pt` / `critic.pt` weights (blocks 1-4 + 1×1 conv), script `probes/carte_presence.py`. New episodes, never seen by training. Reserve: the `--sightings` call invokes `camera->fov()`, which advances the zoom transition.
Share of visible enemies detected on their footprint at 1 % false positives per cell:

| distance | actor, policy zoom | critic, policy zoom | actor, forced zoom | critic, forced zoom | bench 551 s20, forced zoom |
|---|---|---|---|---|---|
| < 60 m | 0.98 | 0.95 | 1.00 | 0.97 | 0.91 |
| 60-100 m | 0.98 | 0.96 | 1.00 | 0.98 | 0.97 |
| 100-200 m | 0.92 | 0.91 | 1.00 | 0.99 | 0.96 |
| 200-400 m | 0.88 | 0.83 | 0.96 | 0.94 | 0.82 |
| > 400 m | 0.74 | 0.65 | 0.77 | 0.63 | 0.31 |
| cell AUC | 0.993 | 0.986 | 0.993 | 0.989 | 0.972 |

Established: the head learns the map online, at the bench's level or above; at < 100 m blocks 1-4 of both CNNs contain the enemy in 95-100 % of images where it is visible. Consequence for the hypothesis: at good site, 552 s10's < 60 m engagement (0.29) stays 550 s12's (0.31). The information is present at block 4; not measured is whether it survives blocks 5-7 + flatten up to the policy. Evaluator: redo `decodage.py` (probe `--features`, actor CNN output) on 552 s10 and compare engaged / ignored AUC to 551 s20 (0.945 / 0.745). Reduced gap ⟹ final perception repaired, bottleneck downstream (heads, credit); gap kept ⟹ reading after block 4 loses the target.

## Target decoding at CNN output: save_10 (2026-11-23)

Probe save_10 side 900, 6 episodes, `--features` (`probes/data/feat_s10_side900.*`), script `probes/decodage.py` = `comparisons/2026-11-23_origine_balayage/decodage.py` with two adaptations: head feature width read from file (128 here, 256 on 551 with the direct branch), reaction independent of μ canon_x sign; cross-validation thresholds lowered (≥ 10 positives in training, ≥ 2 in test) because counts are small. Replayed on 551 s20 data the adapted script gives 0.941 / 0.744 (report: 0.945 / 0.745); the 551 figures of the table come from this replay — the low canon leaves only 2 620 control decisions within ±8° of the horizon (551: 22 902).
AUC "target present" vs "no target", actor CNN output (blocks 1-7 flattened):

| target in free view | engaged 552 (551) | ignored 552 (551) | engaged vs ignored 552 (551) |
|---|---|---|---|
| < 60 m | 0.978 (0.941), n 50 | **0.942 (0.744)**, n 79 | 0.874 (0.909) |
| 60-100 m | — (n 5) | **0.902 (0.719)**, n 187 | — |
| 100-200 m | — (n 0) | **0.807 (0.621)**, n 219 | — |

Detector learned on engaged passes < 60 m, applied to other tanks, at 5 % false positives: 96 % of engaged, **65 % of ignored** (551: 15 %), 20 % of masked targets (551: 12 %). Engaged / ignored geometry: median distance 34.8 / 45.2 m, |site| 1.9 / 3.3°, yaw 5.5 / 5.4°.
Verdict (rule set before the measure): the engaged − ignored AUC gap goes from 0.20 to 0.04 ⟹ in ignored passes the target is now present at CNN output. The ignored half is no longer a perception defect: the bottleneck is downstream (heads, cell, credit). Engagement at good site did not move (0.29 vs 0.31). Head features (liquid cell output) carry the same information: ignored vs no target 0.909 at < 60 m (551: 0.741): the target reaches μ's input; missing is μ's response, coherent with the known credit bottleneck (without Φ, $\rho(A, \text{noise toward target}) \approx 0$), not established here.
Reserves: small counts (50 / 79 at < 60 m); ignored are 10 m farther than engaged; "engaged vs ignored" stays separable (0.874), so the representation still differs between classes, perhaps by distance. One checkpoint, one seed.

## Night 23→24/11: 1.11 M, D 1000 (read 2026-11-24 09:10)

Run still going, save_41. Plateau at D 650 from 233 k to 404 k, then D 700 → 1000 from 404 k to **699 k** (551: 786 k, 550: never, 545/510: 429 k). At D 1000 (459 rows, 699 k → 1.11 M): kill 19.3, hit 0.38, fire 0.42, miss 46 m, Hc −0.66, Hd 0.155, kl 0.015, skip 0.03; 551: 20.6 / 0.29 / 0.58 / 62 m; 545: 20.4 / 0.37 / 0.29 / 64 m. Kill flat 18-24 over 400 k, neither rise nor collapse. α_fire leaves floor (2e-3 at end).
Probes save_26 (≈ 700 k) and save_41 (1.11 M), 3 episodes per side, `--force` (binary 19:27):

| probe | canon elev. p50 | share \|site\|<6° | net < 60 m good site (n) | net 60-100 m good site | tour<5° | hit/shell | emptied | shots/s |
|---|---|---|---|---|---|---|---|---|
| 550 s21 side 550 | −2.6° | 0.76 | 0.67 (461) | 0.35 | 0.19 | 0.29 | 0.55 | 0.49 |
| 550 s21 side 900 | −1.4° | 0.76 | 0.61 (231) | 0.33 | — | — | — | — |
| 552 s10 side 550 | −15.1° | 0.34 | 0.29 (115) | 0.03 | 0.03 | 0.25 | 0.50 | 0.26 |
| 552 s26 side 550 | −7.8° | 0.59 | 0.55 (245) | 0.26 | 0.08 | 0.55 | 0.09 | 0.18 |
| 552 s41 side 550 | −4.6° | 0.86 | **0.91 (77)** | **0.57** | 0.17 | 0.37 | 0.36 | 0.73 |
| 552 s41 side 1000 | −3.7° | 0.88 | 0.58 (60) | 0.56 | 0.19 | 0.34 | 0.51 | 0.33 |

(The value 0.31 of "550 s12" cited above was at s12; 550 s21 gives 0.67.)
- Canon back up: −15° (s10) → −7.8° (s26) → −4.6/−3.7° (s41); good-site share 0.86-0.88 (> 550). The low-canon confounder is gone at s41.
- Engagement: at good site, < 60 m 0.91 side 550 (550: 0.67) but 0.58 side 1000; 60-100 m 0.56-0.57 on both sides (550: 0.33-0.35). Net gain at 60-100 m, < 60 m within count noise (60-77 passes).
- Fewer passes: 140-145 passes < 60 m in 3 episodes vs 455-870 (550, 4 ep.) ⟹ the turret sweeps more slowly through targets; tour<5° 0.17-0.19 ≡ 550 s21 0.19.
- Trigger: s26 very rare (0.07-0.18 shots/s, hit/shell 0.55, reserve almost never emptied); s41 reopened (0.33-0.73 shots/s, hit/shell 0.34-0.37, emptied 36-51 %, disarmed ≤ 2 %).
- Kill not above: the better 60-100 m engagement does not translate into kills (19 ≡ 545/551).
Reserves: 552 s41 is at 1.11 M vs 550 s21 at ≈ 570 k; one seed; low < 60 m counts. Not settled: does the head explain the 60-100 m engagement gain, or training duration? Control = probe 551 at a late save (≥ 900 k) with the same script.

### Duration control: 551 save_35 (945 k, D 1000), 2026-11-24

Binary `train_551_ppo_liquid/probe_bin` with `LD_LIBRARY_PATH=probe_bin` (its RUNPATH otherwise loads `build/arenai_agent/libarenai_agent.so`, carrying the presence head ⟹ "Wrong state dict"). 3 episodes per side, same script.

| probe | elev. p50 | good site | net < 60 m good site (n) | net 60-100 m good site (n) | passes < 60 m | tour<5° | hit/shell |
|---|---|---|---|---|---|---|---|
| 551 s35 side 550 | −2.7° | 0.85 | 0.63 (246) | 0.37 (394) | 430 | 0.20 | 0.26 |
| 551 s35 side 1000 | −2.8° | 0.85 | 0.59 (161) | 0.26 (455) | 270 | 0.19 | 0.28 |
| 552 s41 side 550 | −4.6° | 0.86 | 0.91 (77) | 0.57 (117) | 140 | 0.17 | 0.37 |
| 552 s41 side 1000 | −3.7° | 0.88 | 0.58 (60) | 0.56 (174) | 145 | 0.19 | 0.34 |

- Duration does not explain the 60-100 m gain: 551 trained as long stays at 0.26-0.37 (≡ 550 s21 0.33-0.35); 552 at 0.56-0.57 on both sides. < 60 m: 552 ≥ 551 side 550, ≡ side 1000 (counts 60-77): no established gain. hit/shell 0.34-0.37 vs 0.26-0.28; tour<5° identical; CSV kill identical. 552 has 2-3× fewer passes: slower sweep. Reserve: a pass requires an entry at ≥ 8°/decision; a slower sweep selects other passes, so reaction is not measured on the same population.
Verdict: with the head the turret answers 60-100 m targets better and shells hit more; this does not translate into kills. One seed per arm.

## Lever search (2026-11-24): oracles, gesture credit, slope of V on save_41

Scripts: `probes/oracle/lecture.py` (probes `probes/oracle/s41_side*_*`), `probes/credit_geste.py`. 3 episodes per side, `build/` binary (552 code).
Oracles (touches/tank-s; kills/episode):

| | side 550 | side 1000 |
|---|---|---|
| policy | 0.275; 25.3 | 0.119; 17.7 |
| policy deterministic | 0.306; 27.3 | 0.113; 18.0 |
| oracle yaw on target in view | 0.230; 25.7 | 0.088; 17.7 |
| oracle yaw 360° | 0.390; 27.7 | 0.170; 30.7 |
| oracle yaw + site in view, policy trigger | **0.527; 32.0** | **0.365; 28.3** |
| oracle trigger, policy aim | 0.079; 17.3 | 0.036; 7.7 |
| all oracle in view | 0.766; 30.7 | 0.756; 43.7 |

- Under oracle yaw + site the **policy's trigger fires by itself** (1.2-1.3 shells/tank-s, hit 0.31-0.40), mostly at > 100 m (touches/tank-s > 100 m 0.250 vs 0.070 side 550; 0.274 vs 0.039 side 1000). Different from 520/522 where the trigger went silent under oracle aim: co-adaptation is no longer the bottleneck.
- Yaw alone brings nothing: site error stays +1.1 to +2.8° (target above the aimed point); under oracle aim 0.24-0.39°. Policy at aligned shot: site bias +0.45 to +2.1°. σ out of cause (deterministic ≡ stochastic). Margin = servoing the canon on a **visible** target at > 100 m, on both axes.
Gesture credit ρ = corr(A, ε·sign toward target), ε = (atanh a − μ)/σ, advantage recomputed from the probe `value` and save_41's normaliser (std √2.535): yaw −0.015 to +0.016 at all distances and λ (± 0.007-0.011); site < 100 m +0.009 to +0.032; site 100-300 m −0.020 to +0.003. Synthetic reference (reward = drop of |error| at next step, λ 0): yaw 0.031, site 0.049 — low ceiling because noise is only 12-15 % of turret movement (median |μ| 0.97, tanh′ ≈ 0.44, sweep).
Slope of V (armed, regression on |yaw| ≤ 40° and distance): −0.05 to −0.06 per 10° at < 100 m, −0.01 to −0.02 at 100-200 m (550 s21: −0.11 / −0.06 / −0.01; 551 s35: −0.13 / −0.05 / −0.02): transversal, does not translate into gesture credit (λ 0: ρ ≈ 0).
Verdict: perception settled, trigger ready, margin of ×1.9-3.1 touches in the aim of visible targets at > 100 m; no gradient leads there (ρ ≈ 0, even the dense reference does not exceed 0.05 in the current sweep regime).
Option A measured offline (`probes/credit_phi.py`, Φ₂ replica on nearest enemy, credit γΦ(s′) − Φ(s) normalised, λ 0): ρ yaw 0.030-0.042 (< 100 m), 0.035-0.064 (100-300 m), site 0.00-0.02; same Φ added to the reward with GAE λ 0.95 (invariant shaping): ≡ event-only (≤ 0.025). Under the 0.08 criterion (545/548: 0.08-0.10 with the real Φ₂, my replica underestimates).
Decomposition of the one-step credit SNR: ρ ≈ |∂e/∂ε|·|∂V/∂e| / sd(δ); |∂e/∂ε| = 0.42-0.61° per unit of noise (sweep, |μ| 0.97, tanh′ ≈ 0.44), sd(δ) = 0.155; current V slope 0.05/10° ⟹ ρ 0.014-0.020; geometric slope 0.3/10° ⟹ 0.08-0.12.
Option B measured offline (2026-11-24): `--features` export of save_41 (6 ep. × 2 sides, 283 k decisions, `probes/data/q_s41_*`), test = tanks ≡ 0 mod 5. Scripts `probes/banc_q.py`, `banc_svg.py`, `decodage_erreur.py`, outputs `banc_q_s41*.txt`, `banc_svg_s41.txt`. Q(z, a) on actor features + canon action + shot, targets reference / TD(0) / λ-return: ∂Q/∂a toward target 76-79 % (yaw < 100 m), 57-59 % (100-300 m), site 72-92 %. Control: same Q with a = tanh μ (no noise, pure inter-state correlation): 76-77 % ⟹ slope confounded. With noise alone (causal effect): 41-58 %, **reference included** ⟹ Q does not extract the gesture's effect; Stein: E[∇μ·∂Q/∂a] = E[A·ε/σ·∇μ], same information as PPO. SVG variant (latent dynamics f(z, ε) → Δz, test R² 0.61, then ∂v(z + f)/∂ε): reference 0.38-0.60, run V 0.43-0.57 ⟹ null. Cause of the reference's null: signed error poorly readable in actor features — signed yaw R² 0.33 (< 100 m) and **0.05 (100-300 m)**, site+drop 0.57 / 0.38. Label = nearest enemy (not necessarily the one seen): to confirm on the visible enemy (`--sightings`).
Where the bearing is lost (2026-11-24): export save_41 `--frames 5 --sightings --features`, 3 ep. × 2 sides (`probes/data/g_s41_*`, 2.5 GB), script `probes/gisement_par_etage.py`, output `gisement_par_etage_s41.txt`. Target = image position (ndc) of the nearest visible enemy (free view), 11 365 images, reading Conv 1×1 → MLP learned on 8.6 k images, test on tanks ≡ 0 mod 5. Control: block 7 recomputed ≡ exported CNN output (corr 1.000). Floor: same blocks of save_0.

| stage (save_41) | left/right side 60-100 / 100-200 / 200-400 m | R² ndc_x same |
|---|---|---|
| block 3 | 0.79 / 0.76 / 0.67 | 0.18 / 0.24 / 0.18 |
| block 4 (presence head) | 0.78 / 0.81 / 0.76 | 0.19 / 0.34 / 0.38 |
| block 7 = CNN output | 0.86 / 0.84 / 0.71 | 0.52 / 0.41 / 0.25 |
| liquid cell | 0.89 / 0.84 / 0.68 | 0.45 / 0.33 / 0.09 |
| save_0, blocks 3-7 | 0.62-0.72 / 0.60-0.64 / 0.48-0.54 | < 0 |

Median error ≈ 1.5-2.5° at 60-200 m and 5-7° at 200-400 m in zoomed field (40°, p_zoom 0.9; 3.0-6.9° / 11-16° if field 79°); < 1° at 200 m required (3 m tank = 0.86°). ⟹ bearing is not lost at one stage: it is **coarse from block 3** and stays so up to the cell; the presence head (5° cells) does not learn fine position. Reserve: reading learned on 8.6 k images = lower bound.
Fine-position bench (2026-11-24, `probes/banc_position_fine.py`, output `banc_position_fine.txt`): images of the 2026-11-23 bench (551 s20, forced zoom, 127 k images, 25.6 k training with visible enemy < 400 m, test = unseen episode 15.3 k), Linear-LN-SiLU-Linear reading on the block-7 256 output, target (ndc_x, ndc_y) of nearest visible enemy, 4 passes. Yaw: left/right / median error / share < 1°; site: median error.

| configuration | 60-100 m | 100-200 m | 200-400 m |
|---|---|---|---|
| CNN of 552 frozen (what the policy receives) | 0.92 / 1.26° / 0.42; 0.61° | 0.89 / 2.19° / 0.27; 0.82° | 0.77 / 5.34° / 0.13; 1.18° |
| blocks 1-4 of 552 frozen, 5-7 relearned | 0.94 / 0.92° / 0.53; 0.43° | 0.94 / 0.77° / 0.58; 0.36° | 0.89 / 1.34° / 0.41; 0.49° |
| CNN of 552 fully fine-tuned | 0.94 / 0.83° / 0.56; 0.44° | 0.94 / 0.63° / 0.64; 0.31° | 0.90 / 0.94° / 0.52; 0.40° |
| new weights | 0.92 / 1.04° / 0.49; 0.54° | 0.92 / 1.04° / 0.49; 0.47° | 0.81 / 2.64° / 0.29; 0.77° |
| always at center | 7.04° | 10.67° | 11.33° |

⟹ the architecture reaches < 1° up to 400 m; fine position **is present at block 4** of 552's CNN (readable at 0.77-1.34° relearning only blocks 5-7) and **it is 552's blocks 5-7**, learned by PPO alone, that blur it (2.2° at 100-200 m, 5.3° beyond). Corrects the "coarse from block 3" reading (Conv 1×1 → 8 channels on 8.6 k images too weak). Reserves: supervised, one seed, 551 images at forced zoom.

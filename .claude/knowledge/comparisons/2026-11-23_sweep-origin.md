# Origin of the sweep — 551 save_20 and 550 save_21, side 900

Runs: 551 (save_20, 540 k, D 900, direct branch), 550 (save_21, 567 k, D 900), 547/548 (reference).
Date: 2026-11-23. Data: stochastic probes `551/probes/save_20_side900` and `550/probes/save_21_side900`. Scripts and data in `outputs/comparisons/2026-11-23_origine_balayage/` (run with `.venv/bin/python <script>.py <probe prefix>`; `reponse.py`, `reaction.py`, `suite.py`, `decours.py`, `phase.py`, `champ.py`, `site.py`, `retour.py`, `visibilite.py`, `decodage.py`, `planche.py`, `banc_carte.py`, `banc_lectures.py`, `data/`; images `outputs/comparisons/2026-11-23_origine_balayage/planche_0_60_ignores.png`, `outputs/comparisons/2026-11-23_origine_balayage/planche_0_60_engages.png`, `outputs/comparisons/2026-11-23_origine_balayage/carte_entrainement_controle.png`).

Unit of analysis: the **pass**, i.e. the entry of the nearest enemy within ±20° of yaw while the turret sweeps (≥ 8°/decision). Reaction = drop of tanh μ canon_x between the mean of the 3 decisions before entry and the minimum of the next 5; "clear" > 0.5, "none" < 0.15.

## 1. Delay is not the cause (`reponse.py`)

The direct branch removed one decision of delay: μ drops at decision +1 after entry in 551, at +2/+3 in 550. Behaviour is the same (see `551/probes/save_20.md`).

## 2. The turret is bimodal per pass (`reaction.py`, `suite.py`, `decours.py`)

| 551 s20 | passes | no reaction | clear reaction | hits over 3 s (clear / none) | share of run's hits (clear) |
|---|---|---|---|---|---|
| target < 60 m | 670 | 35 % | 40 % | 2.10 / 0.14 | 51 % |
| 60-100 m | 1479 | 54 % | 19 % | 0.46 / 0.01 | 12 % |
| 100-200 m | 2245 | 66 % | 11 % | 0.02 / 0.00 | 0 % |

- 26 % of passes at < 100 m (the clear reactions) make 63 % of the run's hits. 550 s21: 22 % of passes, 47 % of hits.
- Without reaction: μ, σ, p_fire and the critic's value stay flat throughout the pass (tanh μ 0.71 → 0.74, V 0.07 → 0.07, p_fire 0.01).
- With reaction: V rises from decision −1, σ drops (0.28 → 0.19), μ goes from +0.78 to +0.09 in three decisions, yaw speed 14 → 5°/decision.
- The "partial one-sided brake" mean of earlier analyses is the mixture of these two populations.
- Share of clear reaction across training, target < 60 m: 16 % (550 s12, side 550), 32 % (550 s21), 40 % (551 s20).

## 3. What does not decide the reaction (`phase.py`, `champ.py`, `reaction.py`)

Passes with target elevation (site) between −8° and +4°:

- sweep phase (smallest |yaw| seen at decisions 0 and +1): 0.54 / 0.60 / 0.54 / 0.53 at 0-2 / 2-4 / 4-6 / 6-8° (< 60 m);
- sweep speed at entry: 0.43 to 0.59 from 8 to 18°/decision, no trend;
- number of decisions where the target is in the zoomed image (±22.5° × ±11.7°): 0.42 / 0.42 / 0.51 for 3 / 4 / 5 decisions of 5; target never in the zoomed image: 6 % of passes < 60 m, 12 % at 60-100 m;
- shell reserve: 0.31 to 0.53, no trend.

## 4. What decides the reaction

- **Distance**: 40 % / 19 % / 11 % at < 60 / 60-100 / 100-200 m.
- **The situation, persistently** (`phase.py`): over two consecutive passes in front of the same target (1.5 to 4.5 s apart), P(clear at 2nd | clear at 1st) = 0.63 versus 0.25 if none at 1st (base 0.36, n 237). 550: 0.51 versus 0.19 (n 178).
- **Target elevation in the image** (`site.py`), target < 60 m: 53-64 % when it is 0 to 6° below the axis, 15-36 % from 3 to 6° above, 0-3 % beyond 6° above (still in the image up to 11.7°). The cannon is more than 10° under the horizon in 20 % of passes (reaction 12 %). Causal reach bounded: the oracle elevation alone gave only +16 % hits on 550.

## 5. Once engaged, the turret does not recentre (`decours.py`, `retour.py`)

- Median offset after braking: +10 to +13° beyond the target, for at least 0.8 s; tanh μ stays positive (+0.09 to +0.33), no restoring force.
- p_fire 0.13-0.20, i.e. 1.2 shots/s for 5 shots/s allowed by the cooldown (1/6 s).
- A noise step toward the target durably moves the turret (13.6°/unit at 2 decisions out of 18.3 without compensation, ~1.9° per 1 σ) but changes neither shots nor hits at 1 s: 0.79 hit (toward target) versus 0.85 (opposite), target < 60 m, |yaw| 2-25°, n 2128 per tercile. ρ(noise, hits or reward at 0.5-5 s) between −0.035 and +0.025 in all contexts.

## 6. Line of sight (`visibilite.py`, new probe 551 save_20)

Column `nearest_sight_fraction` added to the probe: free share of the camera → nearest enemy segment (ray cast). "Free" = the obstacle hit is within 8 m of the target. The old probe used in §1-5 is in `data/`.

| sweeping passes | share of passes with free sight | clear, free sight | clear, never free | clear, free sight and site −8 to +4° |
|---|---|---|---|---|
| target < 60 m | 83 % | 42 % | 20 % | 55 % |
| 60-100 m | 77 % | 21 % | 18 % | 30 % |
| 100-200 m | 60 % | 11 % | 13 % | 18 % |

Witness: with no enemy within 45° of the axis, a "clear" μ drop occurs in 12 % of windows (n 2912).

- Ignored targets are visible: masking by scenery concerns 2 % of decisions at < 60 m, 12 % at 60-100 m, 32 % at 100-200 m. The "hidden target" hypothesis is ruled out.
- Beyond 60 m the turret does not respond to the target: 21 % reactions with free sight versus 18 % target masked and 12 % no target. The "clear reactions" of §2 at 60-200 m are mostly this background noise.
- At < 60 m, free sight and target near the axis in elevation: 55 % reactions, about one target in two once background noise is removed.

## 7. Target decoding in the actor's CNN (`decodage.py`)

Probe 551 save_20, side 900, 6 episodes, `--features` export (`data/feat_551_s20_side900.*`): actor CNN output (256) and features read by the heads (256). Linear classifier, cross-validation per tank. Rows = decisions 0 and +1 of the pass (before the brake effect), target in zoomed image, cannon within ±8° of horizon. Negatives: sweep with no enemy within 45° of axis (n 22 902).

AUC "target present" versus "no target", CNN output:

| target with free sight | engaged pass | ignored pass | masked target (placebo) | engaged versus ignored |
|---|---|---|---|---|
| < 60 m | 0.945 | 0.745 | 0.638 | 0.852 |
| 60-100 m | 0.871 | 0.719 | 0.547 | 0.864 |
| 100-200 m | 0.798 | 0.621 | 0.550 | 0.909 |

Controls, target < 60 m:
- at equal sample size (198), engaged versus no target: 0.923 (5 draws, 0.913-0.932);
- detector learned on engaged passes, applied to other tanks, at 5 % false positives: fires on 74 % of engaged, 36 % of weak reactions, **15 % of ignored**, 12 % of masked targets;
- near-identical geometry between ignored and engaged: median elevation −0.8° / −1.6°, |elevation| 4.0° / 2.1°, yaw 6.1° / 5.3°, distance 49.6 / 45.8 m.

Head features give the same figures (0.929 / 0.733 / 0.669 at < 60 m).

## 7 bis. Images (`planche.py`, `planche_0_60_ignores.png`, `planche_0_60_engages.png`)

Probe 551 save_20, 2 episodes, `--frames` export of all decisions (`data/img_551_s20_side900.*`, 7.4 GB). 12 passes drawn at random per class, target < 60 m, free sight, |yaw| < 12°, |elevation| < 7°, image of the decision where the target is most centred (43 ignored, 98 engaged available).

- In ignored passes the target is as visible to the eye as in engaged passes: large, whole, unmasked, contrasted on the background in 11 images out of 12.
- Only visible trend: target higher in the image when ignored (sample elevation +2, +6, −5, −1, −3, +3, +3, +4, +2, −3, +6, +6 versus −3, −6, −1, +3, −1, −2, −1, −3, −5, −4, −1, 0), often on a ridge, against the sky. Consistent with §4.
- Centred and low targets are also ignored (3 out of 12).

⟹ the image contains the target; neither partial masking nor colour camouflage. The CNN does not represent it though it is there.

## 8. Reading

Established:
- neither cell delay, nor sweep speed or phase, nor scenery masking explain the defect;
- the turret responds to a visible target only at less than 60 m, and to one in two; these stops make half the hits (2.1 hits per pass versus 0.14);
- **in an ignored pass, the CNN output resembles that of a masked target**: the target, with free sight, at the same distance and same image position, is barely represented. The defect is upstream of the heads and the cell: it is target extraction by the CNN, not the credit of the gesture on a seen target;
- it is durable from one turn to the next for the same target (§4), and the critic (another CNN) is flat in the same passes (§2);
- once stopped, the turret does not recentre (§5) and no isolated noise step pays: second, distinct defect, which belongs to credit.

Reading (Claude's, untested by intervention): trained on reward alone, the CNN learned to represent "tank where a shot pays" (near the axis, slightly below), not "enemy present". A target outside this zone produces nothing, hence neither brake nor elevation correction, hence never a hit that would teach it to represent it. Elevation does not explain everything: in the best elevation band the response caps at 46-55 %.

To reconcile: witness 548 (target position as input) did not slave the turret, but it measured the mean gain under Φ₂ credit, not the share of engaged passes, and barely swept (17 to 51 sweep entries per probe).

Marker: in 547/548 (Φ₂ credit) a pass at < 60 m yields 8-10 shots and 4.3 hits in 3 s, median offset 1-4° at 1 s, versus 3.1 shots and 2.1 hits for an engaged 551 pass.

Limits: one checkpoint, one run, target = nearest enemy only, camera frame reconstructed (camera at (0, 4, −20) from the cannon, without `CollisionCamera` recoil), real zoom not exported (p_zoom 0.85 in 551), linear decoder.

## 9. Offline bench: is the presence map learnable? (`banc_carte.py`)

Validated by Samuel on 2026-11-23 before any training code. Probe 551 save_20, side 900, forced zoom, 3 episodes, all decisions: 127 549 images (`data/banc_551_s20_zoom.*`, 11.6 GB) and, per image, the projection of each living enemy through the render camera (`--sightings`, `TrainTankEnvironment::enemy_sightings`) plus line of sight. Labels checked by eye (`data/banc_551_s20_zoom_etiquettes.png`). 0.77 visible enemy per image, at least one in 41 %.

Network: first blocks of the agent's CNN (same convolutions, GroupNorm, SiLU), fresh weights, then a 1×1 convolution → one logit per cell. Loss: weighted binary cross-entropy. Adam 3e-4, 6 passes on episodes 0-1 (85 587 images), test on episode 2.

Share of visible enemies detected at 1 % false positives per cell, unseen episode:

| distance | centre only, block 4 (8×16) | footprint, block 4 (8×16) | footprint, block 5 (4×8) |
|---|---|---|---|
| < 60 m | 0.22 | 0.91 (0.57 on the centre cell) | 0.73 |
| 60-100 m | 0.51 | 0.97 (0.81) | 0.84 |
| 100-200 m | 0.81 | 0.96 (0.89) | 0.84 |
| 200-400 m | 0.79 | 0.82 (0.77) | 0.65 |
| > 400 m | 0.38 | 0.31 (0.28) | 0.26 |
| AUC per cell | 0.975 | 0.972 | 0.960 |

"Footprint" = all cells covered by the tank (7 m × 3 m projected) are positive; detection is then counted on the best cell of the footprint, which leaves a near tank several tries (effective false positive rate above 1 %).

- The map is learnable at 256×128: 91 to 97 % of visible enemies within 200 m, versus a 551 turret response of about 49 / 20 / 7 % (§6).
- The centre-only label fails on near targets (0.22 at < 60 m): the tank covers several cells and only one is positive.
- Wiring one block further (4×8) is worse everywhere.
- Loss was still falling at the 6th pass; clear train / test gap on the centre cell at < 60 m (0.77 versus 0.57, n 105 and 426).

Limits: forced zoom (constant field), a single collection policy, fresh weights (not 551's CNN), reading the camera via `pos()` / `fov()` has side effects outside steady state (zoom in transition): usable in the forced-zoom probe, not as is in training.

## 10. Offline bench: three CNN readings (`banc_lectures.py`)

Requested by Samuel on 2026-11-23, without touching code. Same images as §9 (85 587 to learn, episode 2 to test), 5 passes, Adam 3e-4. Each network has the presence head (footprint, 8×16 map after block 4) and a reading output (Linear → LayerNorm → SiLU → Linear) that must give the horizontal position of the nearest visible enemy (< 200 m). Spatial softmax = pre-`cfb76a2e` implementation, taken as is (x, y, max per channel, one learned temperature).

Left/right correct (target more than 2° off axis) / median yaw error, unseen episode:

| regime | reading | < 60 m | 60-100 m | 100-200 m |
|---|---|---|---|---|
| presence + reading, all trained | 4 blocks + spatial softmax | 0.855 / 2.43° | 0.940 / 1.26° | 0.917 / 1.23° |
| | 4 blocks + flattened 8×16 | 0.917 / 1.55° | 0.943 / 1.36° | 0.916 / 1.74° |
| | 7 blocks flattened (current code) | 0.896 / 1.61° | 0.962 / 0.89° | 0.920 / 0.84° |
| blocks 1-4 trained by presence alone | 4 blocks + spatial softmax | 0.908 / 2.23° | 0.920 / 1.68° | 0.920 / 1.56° |
| | 4 blocks + flattened 8×16 | 0.945 / 1.44° | 0.938 / 1.36° | 0.896 / 2.54° |
| | 7 blocks flattened (current code) | 0.908 / 1.58° | 0.953 / 0.89° | 0.931 / 0.86° |
| reading only, no presence head | 4 blocks + spatial softmax | 0.904 / 2.03° | 0.942 / 1.07° | 0.915 / 1.15° |
| | 4 blocks + flattened 8×16 | 0.904 / 1.71° | 0.940 / 1.20° | 0.906 / 1.66° |
| | 7 blocks flattened (current code) | 0.902 / 1.46° | 0.960 / 0.76° | 0.933 / 0.74° |
| reference "always 0" | | 0.500 / 4.45° | 0.500 / 7.04° | 0.500 / 10.67° |

Test sizes: 1 256 / 2 339 / 6 499. Learned softmax temperature: 0.80 to 0.88.

- All three readings localise the target: 0.86 to 0.96 left/right. The spatial softmax is not blind to side (the 0.51 of the 2026-11-16 bench does not reproduce; that bench had 10.8 k images, zoom off, target = the most on-axis enemy).
- With blocks 1 to 4 trained by presence alone, position stays readable by the three readings (0.90 to 0.95): the presence map suffices to make it available.
- The 7 flattened blocks are the most precise from 60 m (0.74-0.89° versus 1.1-2.5°); the softmax is the least precise under 60 m (2.0-2.4° versus 1.4-1.7°).
- In supervised mode the presence head adds nothing to the reading: the reading signal suffices.

Limits: reading is supervised here, whereas in training it receives only PPO's signal; one seed; a single collection policy; forced zoom. This bench measures what each reading can transmit, not what PPO will make it learn.

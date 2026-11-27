# train_421_ppg — analysis (2026-11-12, run in progress)

## Header

- PPG, 32 tanks, vision 256×128, 4-stage CNN [16, 32, 64, 128], MLP 256/128.
- Launched 2026-11-11 22:10 → last row 2026-11-12 07:23 (9.2 h), **running**, index 515k.
- Throughput 55.9k steps/h (1.79 M tank-steps/h), ≈ 1.9× the SACs 419/420 (29.8k steps/h).
- Effective commit: **c02137c6** « absolute angle for turret » (commit 22:05, binary 22:10:07,
  run 22:10:42; config = defaults of `ppg_hyperparams.h` at that commit). High confidence.
- Saves every 27k: save_K ≈ K × 27k (save_19 ≈ 513k).

## What this run tests

First PPG since 395 (2026-11-02) — in between: reward redesign of 06/11 (damage as fraction
of HP, Cauchy aim potential, fire cost 0.03, starvation removed) and the SAC lineage 411-420.
Direct comparison with 395 is therefore not readable; useful references are the SACs 415-420
(same reward).

Changes of c02137c6, **confounded**:

1. **Absolute aim command** (retry, see the 24/09 revert): 5 continuous actions
   `dir_x, dir_y, yaw_cos, yaw_sin, pitch` instead of 4. Yaw = `atan2(yaw_sin, yaw_cos)` in the
   chassis frame, pitch = `pitch × MAX_ANGLE`; in both cases the step is limited to
   `max_rad_per_frame` (`turret.cpp`, `canon.cpp`). The player stays in delta.
2. Per-dimension entropy targets: `[0, 0, −1, −1, −1]` (before: 0 for all).
3. `AlphaParameters::MIN_ALPHA` 1e-3 → **1e-4** (affects PPO/PPG/liquid).
4. Bigger network and 2× vision vs the SAC defaults (no recent PPG to say what changed on the PPG side).

## CSV reading

D milestones: 250 @ 1k, 300 @ 21k, 350 @ 114k, **400 @ 145k**, then stuck at 400 (370k decisions).

| phase (index) | D | hit | kill | fire/s | miss m | s | ev | Hc | Hd | skip |
|---|---|---|---|---|---|---|---|---|---|---|
| 0–20k | 250 | 0.076 | 3.3 | 1.24 | 21 | 68 | 0.69 | −0.03 | 0.69 | 0 |
| 21–113k | 300 | 0.088 | 5.7 | 1.13 | 23 | 63 | 0.71 | 0.09 | 0.65 | 0.003 |
| 114–144k | 350 | 0.101 | 6.8 | 1.09 | 26 | 72 | 0.73 | 0.19 | 0.52 | 0.006 |
| 145–206k | 400 | 0.086 | 9.1 | 1.41 | 31.5 | 61 | 0.70 | −0.22 | 0.55 | 0.14 |
| 206–360k | 400 | 0.083 | 14.2 | 1.69 | 32 | 60 | 0.66 | −0.51 | 0.56 | 0.44 |
| 360–515k | 400 | 0.077 | 16 | 1.75 | 33 | 61 | 0.62 | −0.72 | 0.39 | 0.37 |

- Hit peak 0.103 @ 133k (D=350). At D=400, hit slides 0.086 → 0.077 over 370k decisions.
- kill 7 → 19 (max 19.7 @ 502k) and r_hit rises, **but hit/fire does not rise**: the increase comes
  from fire volume (fire 1.1 → 2.1/s, max 2.9 @ 497k), not precision. Miss 26 → 36 m at equal D.
- Hc 0.23 (140k) → −1.0 (515k), min −1.34 @ 501k; Hd 0.69 → 0.35 (≈ H(zoom), fire masked).
- α: all at the 1e-4 floor from ~40k to ~200k. α_dir_x/y leave the floor from ~220k
  (max 5.6e-3 / 1.4e-2 around 365k) → displacement entropy fell under its target 0.
  α_yaw_cos/sin/pitch only take off after ~440k (1.7e-3 / 6.6e-4 / 7.8e-4 at the end) → aim entropy
  just reached −1. α_fire/zoom at the floor all run.
- Optimisation: kl per rollout 0.035 (< target 0.05) but skip 0.35-0.55 since 200k and clip
  0.27-0.35; ev 0.9 (154k) → 0.40 at the end, v ×2-3.

## Diagnosis

**Established**
- The absolute command does not unlock aim in this run: at D=400, hit/fire flat then slightly
  declining (0.086 → 0.077) and miss rising (31 → 36 m), while σ tightens (Hc −0.2 → −1.0).
  Known pattern « σ collapses without hit gain » (373, 383).
- The rise of kill and r is a cadence effect: more shells at constant precision. Do not
  read kill/r as aim progress.
- Entropy regularisation did not slow the contraction: α floor lowered to 1e-4 and aim targets
  at −1 → no entropy cost on aim until ~440k. The aim tightening is therefore free, and brings
  nothing to hit.
- KL regime since 200k (skip ~0.4, clip ~0.3): a good part of the actor epochs is cut.

**In the noise / undecided**
- D=400 vs SAC 415: hit 0.08 (421) vs 0.11 (415). But network, vision, algorithm and action change
  together; the gap cannot be attributed to the absolute command.
- ev decline 0.9 → 0.4: may come from non-stationarity (cadence ×2, more deaths) rather than
  a degraded critic; no correlated hit drop 10-20k later.
- The CSV does not say whether aim follows the target (corr μ/error), whether the trigger is selective, or
  whether the reserve is emptied: the fire increase may be dump.

**Mechanism point to keep in mind**: the command is in the chassis frame while the
vision sees from the turret; to aim, the actor must add the current turret angle
(proprioception) to the offset seen in the image. In delta, the image offset sufficed.

## Options (Samuel's decision)

1. Probe save_5 (≈ 135k, hit peak at D=350) and save_19 (≈ 513k): corr(μ_yaw/pitch, error),
   hit per shell by alignment, p(fire | angle), empty reserve. Settles whether absolute aim follows
   the target and whether the fire increase is selective or a dump.
2. Stop the run: 370k decisions at D=400 without hit/fire gain.
3. To isolate the absolute command: relaunch 421 identically in delta (same network/vision/
   MIN_ALPHA) — otherwise the four changes stay confounded.

## Markers

- hit/fire and miss at D=400, not kill/r.
- Hc under −1 with aim α taking off: the −1 target starts to act; if hit does not move by
  ~600k, contraction was not the lock.
- sustained skip > 0.5 → KL regime freezing the actor.

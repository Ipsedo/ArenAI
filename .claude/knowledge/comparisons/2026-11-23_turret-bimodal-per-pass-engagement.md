# Turret is bimodal per target pass (ignores or brakes sharply); no delay, no speed, no masking cause

Runs: 550 (save_21, side 900), 551 (save_20, side 900), 547/548 (reference)

Analysis 2026-11-23 on 551 save_20 and 550 save_21 (side 900), after the failure of the direct branch in the liquid cell (comparisons/2026-11-23_direct-branch-liquid-cell.md). Report and scripts: `outputs/comparisons/2026-11-23_origine_balayage/` (`.venv/bin/python <script>.py <probe prefix>`).

- Direct branch: delay removed (μ drops at +1 decision vs +2/+3), behaviour unchanged ⟹ delay is not the cause.
- Per pass (entry of the nearest within ±20° during sweep): no reaction 35 / 54 / 66 %, net reaction 40 / 19 / 11 % at < 60 / 60-100 / 100-200 m. Net reactions at < 100 m = 26 % of passes and 63 % of the run's hits (550: 22 % and 47 %); 2.1 hits per engaged pass < 60 m vs 0.14 ignored.
- Without reaction: μ, σ, p_fire **and critic V** flat. Independent of sweep phase and speed, of number of frames where the target is in the zoomed field, of reserve.
- Persistent per target: P(net at 2nd pass | net at 1st) 0.63 vs 0.25 (550: 0.51 / 0.19).
- Elevation: target > 6° above the axis ⟹ 0-3 % reaction; canon > 10° below horizon in 20 % of passes.
- Once engaged, the turret does not recentre: stays +10 to +13° past the target, tanh μ > 0, p_fire 0.13-0.20 (1.2 shot/s of 5 allowed, cooldown 1/6 s). A noise step toward the target moves the turret ~1.9° durably without changing shots or hits (ρ ≈ 0 everywhere).
- Share of net reaction < 60 m over the run: 16 % (550 s12) → 32 % (550 s21) → 40 % (551 s20).

Line of sight (column `nearest_sight_fraction` of the probe, validated by Samuel 2026-11-23, uncommitted: `train_environment.{h,cpp}` + `probe.cpp`, `build/` rebuilt; `visibilite.py`):
- ignored target ≠ hidden target: clear view in 83 / 77 / 60 % of passes at < 60 / 60-100 / 100-200 m; scenery masking 2 / 12 / 32 % of decisions.
- background noise: « net » μ drop in 12 % of windows with no target at all. Reaction with clear view 42 / 21 / 11 %, masked target 20 / 18 / 13 % ⟹ beyond 60 m the turret does not respond to the target; at < 60 m, clear view and good elevation: 55 % (≈ one in two net over background).
- Not settled: target not extracted from the image (response ∝ apparent size) or seen but stop unrelated to closed trigger (does not explain the ignored half at < 60 m).
- Reference 547/548 (Φ₂): pass < 60 m = 8-10 shots, 4.3 hits in 3 s, 1-4° gap at 1 s.

Decoding (`decodage.py`, probe `--features`, 6 episodes, linear, per-tank validation), actor CNN output, target < 60 m clear view vs « no target »: engaged pass AUC 0.945 (0.923 at equal sample), ignored 0.745, placebo masked target 0.638. Detector trained on engaged, at 5 % false positives: 74 % of engaged, 15 % of ignored, 12 % of masked, at equal geometry (elevation, yaw, distance). ⟹ in an ignored pass the CNN barely extracted the target: defect upstream of the heads (perception), distinct from non-recentring (credit). Cause of the uneven extraction not found (appearance, background, partial masking); proposed next: `--frames` export of ignored/engaged passes. To reconcile with comparisons/2026-11-21_target-localisation-source-cause.md (548 measured mean gain under Φ₂, not per-pass engagement).

Images (`planche.py`, `planche_0_60_*.png`, probe `--frames`): in ignored passes the target is as visible to the eye as in engaged ones (large, whole, contrasted) ⟹ neither partial masking nor camouflage. Only trend: target higher in the image (ridge, sky) when ignored. Claude's reading, untested: the CNN trained on reward alone represents « tank where a shot pays », not « enemy present ».

**Rule:** no longer read the turret as a mean (« partial one-sided brake » = mixture of the two populations); decompose per pass with `reaction.py` / `suite.py`. See runs 551, 550.

**Auxiliary head accepted by Samuel 2026-11-23** (« si le cnn n'apprend pas au final je suis ok », « pose des questions avant de faire quoi que ce soit »). Validated by questions: presence map, on both CNNs (actor and critic), separate optimiser, offline bench first. Bench (`banc_carte.py`, report §9, 127 549 images of 551 s20 forced zoom, probe `--sightings`): visible enemies detected at 1 % false positives, unseen episode — label over the whole tank footprint, branching after block 4 (8×16): 0.91 / 0.97 / 0.96 / 0.82 / 0.31 at < 60 / 60-100 / 100-200 / 200-400 / > 400 m; label at centre only: 0.22 at < 60 m; block 5 (4×8) worse everywhere. Pieces to validate before code: footprint, projection source (camera `pos()`/`fov()` with side effects), learning rate, scope (3 trainers?). (see docs/decisions.md 2026-11-03 lifted 2026-11-23, then removed 2026-11-25)

# Agent visual observation is one decision late (`draw_and_get_frame` reads the previous render)

Runs: 538 (measurement), 539 (first run with the fix)

2026-11-19: the agent's visual observation dates from one decision (0.1 s) — `draw_and_get_frame` reads the previous render; found by code reading + measurement. Fixed same day.

Found 2026-11-19 while investigating why the shot value read from the image credits nothing at one step (aim-synthesis-2026-11-19-the-circle, report `outputs/comparisons/2026-11-19_bruit_avantage_visee/rapport.md` §8).

- `VulkanOffscreenRenderer::draw_and_get_frame` (arenai_view/src/vulkan/scene/renderers/offscreen_renderer.cpp): `draw()` submits into `slots_[slot_index_]` then advances the index; the read takes `slots_[slot_index_]`, i.e. the other slot (`FRAME_SLOTS` = 2) = the render of the previous call. Since the Vulkan migration (0dc62306, 2026-07-18).
- `TrainTankEnvironment::step` renders once per decision ⟹ the image seen at decision t shows state t−1 (0.1 s); the proprio is up to date.
- Measurement (538 s12): H image variation over one step ↔ geometry variation of the previous step 0.148, of the same step 0.102. Gesture credit read in the image: 0 at 0.1 s, 0.056 at 0.2 s; geometry: 0.147 from 0.1 s.
- The delay does not explain the level (delayed geometry AUC 0.856 vs 0.863). Remaining image/geometry gap = data volume (AUC 0.68 → 0.75 from 9 to 36 episodes); colours, framing, resolution ruled out.
- Turret plant over two decisions + image one late: aim loop ≈ 0.3 s.

**Why it matters:** any value or critic read from the image does not see, at one step, the action just taken; consistent with the useless λ 0 critic (one-step aim critic measurement on 522 s68, comparisons/2026-11-18_critic-blind-to-aiming-gesture.md).

**Rule:** in any image ↔ geometry measurement, shift by one decision. Fix (read the slot just submitted) = behaviour change for all checkpoints + possible throughput cost: Samuel's decision.

**Fixed 2026-11-19 ~13:45 (validated by Samuel, uncommitted):** `draw_and_get_frame` reads `slots_[(slot_index_ + FRAME_SLOTS - 1) % FRAME_SLOTS]`; « first image black » assertions removed from test_offscreen, test_offscreen_diffuse, test_vk_offscreen; build/: view 58, core 28, agent 2 442 tests ok. `cmake-build-release` not rebuilt (538 running there with the delay); CI goldens (lavapipe) not regenerated; comment « warm-up: 2 render loops to prime PBO double-buffering » of environment.cpp:200 became inexact, code left.

**Measurement of B redone with the up-to-date image (report §9)**: one-step ρ of H learned from the image 0.00 → **0.100 ± 0.008** (1-10°: 0.139; criterion 0.08 reached, Φ level), AUC 0.749 → 0.797, geometric bound 0.170, image/geometry offset peak back to the same step. Export +7 % slower. Offline, one seed, policy 538 trained with the delay. No run with the fix at 2026-11-19 14:30: effect on kills unknown. (run 539 = 538 + this fix; see its run folder)

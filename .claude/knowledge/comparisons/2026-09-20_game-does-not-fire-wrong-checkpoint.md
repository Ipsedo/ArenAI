# « Agent does not aim/fire in game » = wrong checkpoint loaded via preferences.json, not a Bernoulli bug

Runs: 402 (save_0 loaded), 405 (save_26), 413

2026-09-20 — verified empirically: symptom caused by the wrong checkpoint selected in preferences.json.

2026-09-20, symptom « checkpoint 26 of train_405 neither aims nor fires in game ». Verified: no bug in the Bernoulli chain (sigmoid → sample/max → threshold 0.5 → fire/zoom → canon). Replayed save_26 in the real TrainTankEnvironment in deterministic mode (exact game path): 86–93 % of frames with fire pressed, turret active, hits even at spawn_side 1000 m.

Real cause: the desktop game loads the selection stored in `~/.cache/arenai/preferences.json`, which silently overrides defaults; it pointed to `train_402_ppo_liquid_zoom/save_0`, a barely trained checkpoint (p(fire)=0.40 < 0.5 → never fires in deterministic argmax, Beta mode ≈ 0 → still turret). The prefs had not moved since 2026-09-18 15:18 → no game played on this machine since, despite the impression of having tested save_26.

Reflex for this symptom: check `agent_folder` in preferences.json and the file mtime before looking for a bug in the code. A `save_0` selected by mistake in the menu's file explorer produces exactly this symptom.

Reusable quick diagnosis: load the actor alone (LiquidActor ctor + load_torch) and print p(fire)/modes on synthetic inputs — a save_0 gives p(fire)≈init and modes≈0.

**Caution (2026-09-24): « p(fire) ≈ init » no longer proves « bad checkpoint ».** Run 413 gives p(fire) = 0.391 after 3 M steps, with well-learned modes — the discrete head drifts too slowly to cross 0.5 before ~6 M steps. Check both: the checkpoint index *and* the run's step count. See comparisons/2026-09-24_dead-trigger-deterministic-threshold.md and comparisons/2026-08-17_ppo-entropy-regulation-known-gaps.md.

Additional pitfalls confirmed (2026-09-20, not fixed — design to validate by Samuel):
- No checkpoint compatibility guard: a pre-Bernoulli save (≤ run 401, e.g. embedded resources/trained_models/ppo_liquid_400) loads without error in the current build (same Linear(128,2) shapes) but the discrete head is reinterpreted (softmax [fire, no-fire] → sigmoid [fire, zoom]): save_96 gives p(fire)≈0.41 < 0.5 → never fires in deterministic.
- AI menu: "select_folder" does not reset agent_config, and resolve_agent_config prioritises agent_config if non-empty → an old config.json selected once sticks to any future folder selection.
- Typo in arenai_desktop/src/main.cpp: default config = trained_models/ppo_train_400/config.json (the folder is named ppo_liquid_400).
- Untested suspicion: the desktop launched with --cuda would crash at the first act() (states_to_tensor stays CPU, actor on CUDA; only train.cpp does .to(device)).
- In blind vision (black/noise frames), save_26 STILL FIRES (p≈0.6-0.8) but does not aim → « fires without aiming » = broken vision; « neither fire nor aim » = wrong checkpoint.
(pitfall candidate for a skill/doc: check `~/.cache/arenai/preferences.json` agent_folder + mtime when a game checkpoint seems not to aim/fire)

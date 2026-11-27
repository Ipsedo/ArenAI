# Perception « lock »: enemy bearing not decodable from actor features (470 save_6) — later corrected as a consequence

Runs: 470 (save_0, save_6), 471 (save_1)

Measurement 2026-11-03 (probe `--features`, bearing regression ← actor features): after 700 k steps (470 save_6) the yaw of the visible enemy is NOT decodable from features (RMSE 20° = std, AUC aligned<3° = 0.5, left/right AUC 0.68), identical to the untrained CNN. **Correction by Samuel the same day: this is a consequence, not a lock** (see below).

Tool: `arenai_agent_probe --features <bin>` writes per decision (episode, step, tank) + the liquid output read by the heads (128) + the CNN output (384, spatial softmax x/y/presence per channel). `LiquidActorOutput` gained `features` and `vision_features` (internal struct src/networks/recurrent). Script analyze_features.py (session scratchpad): ridge, MLP 64 and logistic, split by episode, targets = yaw/pitch of the nearest enemy when in field (|yaw| < 39.5°, |pitch| < 22.5°, < 200 m), best_angle < 3°/6°.

| network | yaw RMSE (std) | pitch RMSE (std) | aligned < 6° AUC | left/right AUC |
|---|---|---|---|---|
| 470 save_0 (init) | 24-29° (23°) | 10-12° (13°) | 0.49-0.61 | — |
| 470 save_6 (700 k) | 20-23° (20°) | 7.5-8.4° (8.7°) | 0.55-0.63 | 0.68 |
| 471 save_1 (108 k) | 29-47° (25°) | 7-9° (10°) | 0.63-0.86 (n=8 positives, noise) | — |

By distance (470 s6): 0-40 m RMSE 18-26° for std 20° ⟹ even a 25-px-wide tank is not localised. Positive controls: μ_canon_x decoded R² 0.98 (the method works); shell ratio (in the proprio!) R² 0.03: the liquid does not even keep this input.

**Initial reading:** result explaining the previous weeks (comparisons/2026-08-17_agent-does-not-aim.md, run 467 probe flat trigger / weak turret, run 470 15-s episode, run 471 shot_cost 0.03): the policy has only a vague left/right (corr yaw/μ_cx −0.28) and no notion « target on axis », so no selectivity is expressible whatever the reward; hit 6-9 % is spraying at 32 tanks. The only gradient reaching the CNN is the policy gradient (rare, noisy, delayed advantages): in 700 k steps it learned nothing beyond random init.
**Correction 2026-11-03 (Samuel):** this is NOT a CNN lock: the same network converged (aimed, fired) with shell regen (405-429). The measured undecodability is the consequence of the absence of events since 2026-10-23 (comparisons/2026-10-29_event-density-collapsed-since-10-21.md). Auxiliary head refused (the CNN had already converged). (see docs/decisions.md 2026-11-03: auxiliary head refused, lifted 2026-11-23; 2026-11-10 perception ruled out)
**Rule:** keep probe `--features` + analyze_features.py as a measurement (it says whether aim is learned), but do not deduce an architecture change from it.

# PPO liquid train 429 — 10 Hz, plateau (2026-10-21 to 2026-10-23)

Host: ElMatador (memory notes); raw data no longer on disk.
Source note dates: 2026-10-21/23. `train_429_ppo_liquid_10Hz` (commit `fdaa2b4a`): action_repeat 3 (decision at 10 Hz), γ 0.991 / λ 0.97, ReturnScaler, aim reward `0.1·exp(-miss/(3.25+tan2°·dist))` back in the reward (shell's nearest enemy, no line of sight — same flaw as the 09-09 camping, see event-based reward rework). 16 tanks (419: 32 ⟹ kills/episode not comparable). 8.36 M decisions (= CSV index, not frames).

- At equal D, 429 ≡ 428 (hit 0.063-0.070, kill 2.6-3.6). Peak at 4-5 M (hit 0.071, kill 3.75, D 650-700), then Hc 0.28 → 0.05 between 5 and 6 M, kill 2.6, curriculum back to 600. kl 0.004-0.005 (10× under target 0.05), clip 0.05-0.08. `fire` ≈ 1.1/s **invariant** on 419/428/429 and all D (ammo ceiling estimated ~1.4/s: regen 0.67/s + 10 initial/35 s, ×1/(1−5·hit)).
- Learned σ (sigma_logit, save_77): [0.46; 0.38; 0.23; 0.17] from 0.5 — slow decay ⟹ Hc fall to 0.05 comes mainly from μ pushed to the bounds (truncated law), not σ.
- Artefacts: mean ev −2.6 due to a −1090 spike at 7.57 M (return variance ≈ 0 in a minibatch, clamp EPSILON 1e-8) ⟹ read the median (0.21). Regen comment in jolt_enemy_tank.cpp (« capped at the initial reserve ») ≠ code (`min(nb_shells, max_shells)` = 30).
- The hit ceiling is common to ~15 runs (vanilla/liquid, CNN, lr, minibatch, 30/10 Hz) ⟹ look in the formulation (non-decisional trigger, weak self-play signal) rather than the optimiser.
- **Follow-up (2026-10-23, validated by Samuel):** passive shell regen **removed everywhere** (desktop game included), initial reserve 10 → 20, +5/hit and max 30 unchanged. Break-even: hit 20 %. Prediction: `fire` must drop under the ammo ceiling and `hit` rise > 7-8 %; failure = `fire` → 0 and `kill` collapsing (as 345). Unrelated unstable test: `TruncatedNormalEdgeTest.LogPdfAndPdfConsistent` (unseeded randn) sometimes fails under ctest.

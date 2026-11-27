# Review of runs 523-542 since release v1.3.2 (= run 522): H is not the lever

Runs: 522, 523-542, 538, 539, 541, 542, 543

2026-11-20: H calms the turret but does not learn aim; empty-reserve penalty = 40-70 % of the hit reward at D 800; no run holds D 1000. Requested by Samuel 2026-11-20 (« récap et challenger les changements depuis la dernière release », « on peut revoir totalement l'économie / reward »). Report: `outputs/comparisons/2026-11-20_bilan_depuis_v1.3.2.md`.

- Last release = tag v1.3.2 (0c678b42, 2026-11-16), state of run 522 (penalty at empty): kill 19.2, hit 0.38 at D 1000 over 1 M steps. `gh` not installed: read tags with `git ls-remote --tags origin`.
- 14 changes since: real gains = flattened CNN, up-to-date image; hygiene = batch 2400, kills curriculum, 5 shells/hit; the rest neutral or negative.
- **H (runs 541, 542)**: gain μ/yaw −0.002 to −0.011 and corrector 0.51-0.59, never above 539 s3 without credit; corr(ΔH, gesture) ≤ 0.06; reproduced effect = calmed turret (command 0.5-0.6 °/decision), no sweep, dwell 0.7-1.3 s; hits/tank-s 0.05-0.09 at side 550. Reading (Claude's, untested): H ≈ taking the canon out of GAE.
- **Trigger**: `reste` = r − r_hit − r_rcv = empty-reserve penalties; at D 800 they are 23-24 % of r_hit (522, 539), 39 % (541), 68 % (542, one dump / 21 s per tank, `r` < 0 from 210 to 345 k). Shell break-even: 6 % at D 800, 2.4 % at D 250; aligned shot +0.35 to +0.42 per shell but its trigger closes with the common level (fixed logit gap 3.4-4.1). Drops at D 750-850 in 538, 539, 541, 542. Cause via indexing on D: inferred, not tested.

Choices posed to Samuel, not decided: « canon without credit » control, constant empty penalty, economy overhaul. Nothing coded before his answers, one question per piece.

**Decided and coded 2026-11-20 (« okay pour 0.5 en constante et on enlève le code qui sert plus », plan validated before code):** `EMPTY_RESERVE_PENALTY` 1 → 0.5 constant in `train_environment.cpp`, D indexing removed (constructor parameters `curriculum` / `initial_spawn_side` / `final_spawn_side` of `TrainTankEnvironment`, member `empty_reserve_penalty`). Penalty now 0.5 also in probe and `sigma_gradients` (before: 0 without curriculum). Uncommitted, `build/` only, 2 459 agent tests ok; `cmake-build-release` to be rebuilt by Samuel. Run markers: fire cadence before 100 k (risk: run 479 dead trigger via death at empty), dump resolved or not at s9-s13, trigger holding at D 800 → 950.

Same turn 2026-11-20: `discrete_target_proba` back to 0.01 in `liquid_ppo_hyperparams.h` at Samuel's request ⟹ next run = 541 + constant empty penalty 0.5, only difference.

**Run 543 (= 541 + constant penalty 0.5), at 176 k / D 450, 2026-11-20 21:00** (`outputs/train_543_ppo_liquid/analysis.md`): trigger not extinguished (fire 0.7-1.1, kill 20-23), but dump frequency unchanged vs 541 (one dump / 14-17 s per tank) despite penalty ×2 at D 250 ⟹ `reste` = 60-90 % of r_hit, `r` ≈ 0. kill/hit ≡ 541 up to D 400. Intended effect not readable before D 750-950. D milestones identical to the thousand in 539/541/542/543 (kills curriculum at max cadence): they separate nothing before D ~700.

**Probe 543 save_8 (216 k, D 550), side 550** (`outputs/train_543_ppo_liquid/probes/save_8.md`): dump intact (median empty reserve 3.6 s, 82 % empty, 49 % disarmed), trigger selective (ratio 1.99, p_al 0.80, p_off 0.40), hit/shell 11.4 %, turret < 5° 12.9 % ⟹ ≡ 542 s9, below 541 s9 (23.1 % / 15.8 % / 29 % disarmed). To probe next: saves at D 800-950 vs 541 s12-s16.

**Correction 2026-11-20 evening**: the dump RESOLVES itself in 539/541/542/543 between 300 and 430 k (probes side 550: disarmed < 5 %); the CSV `reste` does not show it. The real failure mode = closing of the whole trigger (p_al 0.08-0.17, < 0.15 shot/s); only 539 s12 sorts (p_al 0.56, p_off 0.03). **543 save_13 (351 k)**: p_al 0.148, p_off 0.006, 0.089 shot/s, disarmed 4 %, turret < 5° 4.4 %, mute in deterministic ⟹ closing of 541 s14-s16 reproduced earlier; constant penalty 0.5 does not change the outcome. Reading criterion to keep: success = p_off < 0.05 AND p_al ≥ 0.5 AND ≥ 0.4 shot/s.
(see docs/decisions.md 2026-11-22: empty-reserve penalty rule = refill + constant 0.25 under curriculum)

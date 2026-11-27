# Trigger entropy regulator caps selectivity (510 probes replay, save_9 → 32)

Runs: 510, 514, 515, 520

2026-11-14: replay of the 510 probes save_9→32 (binary 7422d098) — aim held, trigger loosened at 540 k when α_fire ×20; target H(0.05) > entropy of a selective trigger.

Probes 510 save_9/16/20/24/28/32 at the run's binary (worktree `/home/samuel/CLionProjects/arenai_wt_510`, build/), outputs/train_510_ppo_liquid/probes/rejeu_7422d098/timeline.md.

- Aim held at fixed distance (h<5° 32-34 %, tour<5° 28-30 %) from save_16 to save_32 ⟹ no precision regression.
- Trigger loosened at save_20 (540 k): p_off ×4-6, then 2.5× save_16 until the end; cadence 1.5-2×, empty reserve earlier.
- Mean trigger entropy (armed): 0.09-0.18 at save_16 < target H(0.05) = 0.199 ⟹ α_fire 0.0005 → 0.011 (380 → 530 k), then H 0.16-0.31. 514: same α_fire rise at 500-600 k and plateau.
- ~40 % of armed decisions are off-axis ⟹ a selective trigger is necessarily below the target: the regulator forbids selectivity.

**Why:** explains « 510's peak does not hold »; follows run 510 (per-group KL stop) and the Stooke PID (run 497).

**Test plan:** temporal causality only; test = 515 with lower `discrete_target_proba` (common fire + zoom target) or per-head target (code). Samuel's decision.

**Samuel's choice (2026-11-14):** lower common target, `discrete_target_proba` 0.05 → 0.01 (H 0.056) as default in ppo/ppo_liquid/ppg, config 515 otherwise ⟹ run 520 (uncommitted). Read α_fire and p_off at 400-600 k vs 510/514.

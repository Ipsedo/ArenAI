# Per-shell reward centred on its running mean (ShotBaseline), run 477

Runs: 465, 467, 471, 473, 476, 477

2026-11-04 — per-shell reward centred on its sliding mean (ShotBaseline, window 4096 shells) + Cauchy 2.5 m + 3° × 0.1 restored, kill left raw (hit 0.3 per hit, kill +1.7); ammo curriculum removed, fixed economy 10 / +10 / max 30; run 477 to compare to 476.

2026-11-04, after the failure of all « fixed cost / economy / ADR » variants (465-476), Samuel first rejected the p_fire clamp, the asymmetric critic and the teacher-student (« béquilles », « je veux de vrais leviers sur le reward »), then validated the reward lever alone:

- **Diagnosis**: expected value of a shot negative (465, 471) ⟹ trigger absorbed to 0; positive (467, 473, 476) ⟹ dump. Only stable value: zero, which no fixed cost holds since the mean gain moves with the policy (8 % spam → 50 % aligned). Credit delay not at fault (flight ~0.15 s at 250 m/s, 1-2 steps).
- **Coded**: `ShotBaseline` (arenai_agent/src/core) = exact mean up to `window` shells then EMA; in `TrainTankEnvironment::step`, `rewards[i] -= b × nb_landed_shells`, centred channel g = aim + hit (kill raw). `RewardDetail` now has `aim`, `hit` (0.3 per hit, kills included), `kill` (1.7 more ⟹ 2.0 total unchanged). Constants `HIT_REWARD`/`KILL_REWARD` in constants.h. Cauchy taken from 3c2d540c: tol 2.5 m + 3°, scale 0.1 (turret gradient on every shell, not only the 8 % of hits).
- **CSV only** (no progress bar): `r_aim`, `b` via `get_csv_only_metrics()`, without flag `print_in_progress_bar` (removed).
- **Economy**: ammo curriculum (interpolate, initial_/final_) removed; a single `ShellEconomy` 10 initial / +10 per hit / max 30 for game, train (`--shells`, `--shells_per_hit`, `--max_shells`) and probe; proprio = nb_shells / max_shells (no more `PROPRIOCEPTION_SHELL_SCALE`). γ 0.995 and spawn ADR thresholds 0.05/0.08 kept.
- `--shot_baseline_window` 4096 shells (~3-4 rollouts at 32 tanks).

**Announced success criterion**: probe p_fire depending on angle at step ≥ 10; time before empty reserve lengthening without fire falling to 0.02/s. Listed risks: EMA too short, dump by inertia (neutral shot), noisy kill (hence raw kill). See runs 476 (trigger saturated, ADR plateau), comparisons/2026-11-02_ammo-dump-is-the-reward-optimum.md, run 471 (shot_cost, trigger extinguished). (see docs/decisions.md 2026-11-06 hit reward from closest miss distance removed; 2026-11-22 Φ removed)

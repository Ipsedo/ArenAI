# Aim/trigger loop diagnosis: probes 508-543 cross-read, trigger sorts as well without Φ

Runs: 508-543 (510-522 Φ runs; 524-539 event runs; 541-543 H runs), 543, 544, 545

2026-11-20: the trigger sorts as well without Φ, it is the turret that lets go when the trigger closes; empty-reserve penalty ruled out as a cause (543). Requested by Samuel the evening of 2026-11-20 after 543's failure (« analyse tous les runs pour comprendre »). Report: `outputs/comparisons/2026-11-20_diagnostic_gachette_visee.md`. Stochastic probes side 550, save ≥ 9.

- **The trigger is not the failing piece**: aligned close / off-axis logit gap 2.2-3.2 in Φ runs (510-522), 3.0-4.3 in event runs (524-539) and H runs (541-543).
- **All families go through a closure at 240-430 k**, 522 included (armed p_fire 0.025 at s10, 0.017 at s16); the release fires 0.3-0.5/tank-s, like the « extinguished » phases of 524/528/529.
- **The difference is the turret during closure** (armed p_fire ≤ 0.035): < 5° 31-45 % of the time with Φ credit, 6-12 % event-only, 2-13 % with H. corr(rate, turret < 5°) between saves: −0.49 Φ, +0.57 events, +0.88 H.
- **CSV**: the 8 Φ-credit runs reach D 1000 with kill 17-22; none of the others holds kill ≥ 17 at D 1000.
- **Empty-reserve penalty ruled out**: 543 (constant 0.5) closes at D 700-750 / 351 k, earlier than 541. The dump resolves in all runs; CSV `reste` does not show it, only probes.
- Reading (inferred, not tested by intervention): aiming pays only if a shell leaves, firing pays only if aligned; without an aim signal independent of firing, the trigger's descent cuts the turret's signal ⟹ extinction or oscillation. H worsens (normalised advantage without labels).

**Rule:** stop looking for the cause in the trigger, economy or penalty. Read a run on turret holding when the trigger is closed. Options posed to Samuel 2026-11-20, not decided: Φ credit back on the current tree, target-independent potential (measure offline first), Φ as primer. Nothing to code before his answers.

**Follow-up 2026-11-20/21 — Samuel's objection and measurement.** Objection to any potential (Φ or hit probability): it does not represent what is most urgent to aim at, « ex : un ennemi sur le flanc ». Offline measurement requested (« ouai mesure le »), report `outputs/comparisons/2026-11-21_potentiel_proba_touche/rapport.md`, script `proba.py` (reusable on any `sigma_gradients` CSV, which exports positions + canon heading):
- P = lateral hit probability, 1 − ∏(1 − p_i) over all enemies, dispersion σ. ρ yaw gesture < 100 m, σ 1.6°: 0.051 / 0.052 / 0.050 / 0.113 (543 s8 c550, c1000, 541 s12, 543 s13) vs Φ replica 0.046 / 0.056 / 0.050 / 0.080 ⟹ P ≡ Φ, without designating a target.
- Event GAE (run's critic): 0.00-0.01 on all gestures, out-of-field included. Sum P + w·GAE: harmless at w ≤ 0.25, −20 to −40 % at w = 1, no measurable gain ⟹ the idea « events will choose the target » is not supported.
- Elevation: separate term 0.11-0.15; multiplied with yaw 0.06 and yaw falls to ~0.02 ⟹ one term per axis.
- Wide σ (10°) carries at 100-300 m on 543, not on 541.
- Nothing credits the gesture toward an out-of-field enemy (ρ ≈ 0 for all). Claude's Φ replica underestimates (0.05 vs 0.085-0.12 measured for real on 522): compare only P vs Φ on the same rows.
Decision of Samuel awaited on the pieces (P per axis, σ, GAE weight, fate of H); nothing coded.

**Φ₂ validated piece by piece and coded 2026-11-21 (uncommitted, `build/` only, Samuel launches the run from CLion).** Samuel: « ok d'optimiser l'action de visée si l'heuristique tient compte des ennemis sur le flanc », then « on tente d'abord la nouvelle heuristique et dans un second temps la résolution » (enlarged field of view = next separate trial).
- Formula: Φ₂ = max_i u_i · (bell_i + β · slope_i); bell = 1/(1+(miss/3 m)²) on the theoretical trajectory cut by terrain; slope = 1 − |horizontal bearing|/180° from the canon pivot; u = min(1, 50 m / distance); β = 1. All living enemies, no visibility test. Constants `AIM_TARGET_RADIUS`, `AIM_URGENCY_DISTANCE`, `AIM_BEARING_WEIGHT` (constants.h), public contract `EnemyTank::get_aim_potential(tanks)`.
- Offline measurement (`outputs/comparisons/2026-11-21_potentiel_proba_touche/flanc.py`): ρ yaw gesture < 100 m 0.077-0.081 (Φ 0.046-0.056), 100-300 m 0.047-0.056, out of field 0.014-0.030 (Φ 0). Nearest enemy at ≥ 40° in 75 % of decisions, ≥ 90° in 48 %.
- Wiring: canon_x/canon_y credited on normalised((γ·Φ₂(s′) − Φ₂(s))·(1 − cut)), rest GAE, reward unchanged. `agents/aim_credit.{h,cpp}` (AimPotentials before/after), `TrainTankEnvironment::last_step_aim_potentials()` (Φ₂ read at start and end of step), the 3 trainers. Head H, shot_credit, CSV column `shot_value` and probe column `shot_value` removed ⟹ critics of 541-543 no longer load (sigma_gradients); the probe, which no longer reads the critic, still works on their actors.
- Verified: ctest 5/5, 93 model tests, real 4-min trial (kl 0.006-0.010, skip 0). Throughput not cleanly measured (Φ₂ evaluated 2× per decision).
- Run markers: turret < 5° with closed trigger (target ≥ 30 % like 510-522), holding D 1000, kill ≥ 17. If turret aims but misses flanks ⟹ field-of-view trial.

**Run 544 (= 543 + Φ₂ instead of H, launched 2026-11-21 02:54), point at 85 k / D 400:** D milestones 300 @ 21 k, 350 @ 48 k, 400 @ 67 k (541-543: 47 / 67 / 114 k); kill 25.7-27.4 at D 300-400 (541: 19.7-20.5; 539: 22-26), r +3.3e-3 (541 +1e-3); throughput 127 k steps/h alone on GPU. Probe s3 side 550: turret < 5° 23.4 % (541 s3 10 %, 539 s3 16 %), median angle 18°, trigger still flat (0.74/0.73, normal at s3), dump (empty 2.7 s, disarmed 31 %). Watch: σ canon already 0.03/0.05 (543 s8: 0.08/0.09), Hc −0.97 and kl 0.043 rising, alpha_canon_x 1.3e-3 rising. Next rendezvous: s9 (243 k) trigger, s13-s16 turret with closed trigger.

**Run 545 = isolated control of Φ₂ (decided by Samuel 2026-11-21, one question per setting, coded uncommitted, `build/` only).** 544 killed at 109 k at his request (D 400, kill 27, s4). Reset to 522 values: target_kl 0.10 → 0.05; shells per hit 5 → 2 (default `--shells_per_hit`); empty penalty 0.5 constant → −D/1000 (curriculum re-injected in the constructor of `TrainTankEnvironment`, 0 without curriculum hence 0 in the probe); canon entropy target −1 → 0; canon-specific ceilings removed (`CANON_LOG_SIGMA_MAX` 0.25 and `CANON_INITIAL_SIGMA_MAX` 0.20, SoftClamp scalar again). Left: linear command (a·|a| not restored, « comme 510 »). Kept: flattened CNN, up-to-date image, batch 2400, kills curriculum. Remaining gaps of 545 vs 522: these four + Φ₂ + a·|a|. 2447 agent tests, ctest 5/5.
(see docs/decisions.md 2026-11-22: Φ removed, 2026-11-24: separate aim credit on Φ refused)

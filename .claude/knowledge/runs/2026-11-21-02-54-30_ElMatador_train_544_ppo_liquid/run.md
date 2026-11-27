# train_544_ppo_liquid

| | |
|---|---|
| machine | ElMatador |
| user | samuel |
| launch | 2026-11-21 02:54:30 |
| commit at launch | `cc4ee96b` — add new graphic parameters to GUI |
| commits ≤ 90 min after | none |
| diff at launch | unknown (identity recorded after the run ended) |
| raw data | `outputs/train_544_ppo_liquid` on ElMatador |

## Established facts

- 2026-11-21: run 544 (= 543 + Φ₂ instead of H, launched 2026-11-21 02:54; Samuel runs from CLion) — Φ₂ validated piece by piece by Samuel and coded 2026-11-21 (uncommitted, `build/` only). Samuel: « ok d'optimiser l'action de visée si l'heuristique tient compte des ennemis sur le flanc », then « on tente d'abord la nouvelle heuristique et dans un second temps la résolution » (wider field of view = next, separate trial). Formula: Φ₂ = max_i u_i · (bell_i + β · slope_i); bell = 1/(1+(miss/3 m)²) on the theoretical trajectory cut by terrain; slope = 1 − |horizontal bearing|/180° from the canon pivot; u = min(1, 50 m / distance); β = 1. All living enemies, no visibility test. Constants `AIM_TARGET_RADIUS`, `AIM_URGENCY_DISTANCE`, `AIM_BEARING_WEIGHT` (constants.h), public contract `EnemyTank::get_aim_potential(tanks)`. Offline measure (`outputs/comparisons/2026-11-21_potentiel_proba_touche/flanc.py`): ρ yaw gesture < 100 m 0.077-0.081 (Φ 0.046-0.056), 100-300 m 0.047-0.056, out of field 0.014-0.030 (Φ 0). Nearest enemy ≥ 40° in 75 % of decisions, ≥ 90° in 48 %. Wiring: canon_x/canon_y credited on normalised((γ·Φ₂(s′) − Φ₂(s))·(1 − cut)), rest GAE, reward unchanged. `agents/aim_credit.{h,cpp}` (AimPotentials before/after), `TrainTankEnvironment::last_step_aim_potentials()` (Φ₂ read at step start and end), the 3 trainers. H head, shot_credit, CSV column `shot_value` and probe column `shot_value` removed ⟹ critics of 541-543 no longer load (sigma_gradients); the probe, which no longer reads the critic, still works on their actors. Verified: ctest 5/5, 93 model tests, real 4-min trial (kl 0.006-0.010, skip 0). Throughput not measured cleanly (Φ₂ evaluated 2× per decision). Run markers: turret < 5° with trigger closed (target ≥ 30 % like 510-522), holding of D 1000, kill ≥ 17. If the turret aims but misses flanks ⟹ field-of-view trial.
- 2026-11-21 (point at 85 k / D 400): D milestones 300 @ 21 k, 350 @ 48 k, 400 @ 67 k (541-543: 47 / 67 / 114 k); kill 25.7-27.4 at D 300-400 (541: 19.7-20.5; 539: 22-26), r +3.3e-3 (541 +1e-3); throughput 127 k steps/h alone on GPU. Probe s3 side 550: turret < 5° 23.4 % (541 s3 10 %, 539 s3 16 %), median angle 18°, trigger still flat (0.74/0.73, normal at s3), dump (empty 2.7 s, disarmed 31 %). To watch: σ canon already 0.03/0.05 (543 s8: 0.08/0.09), Hc −0.97 and kl 0.043 rising, alpha_canon_x 1.3e-3 rising. Next appointments: s9 (243 k) trigger, s13-s16 turret with trigger closed. 544 killed at 109 k at Samuel's request (D 400, kill 27, s4), replaced by run 545 (isolated control of Φ₂).

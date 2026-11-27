# Aim — conclusion of the 10° potential lever (507) and diagnosis for an overhaul

Runs: 504, 507.
Date: 2026-11-12. Sources: probes 504 save_2/5/9/12/16 (+ deterministic save_16/18) and 507 save_2/5/9/12, sides 550 and 1000, 4 episodes × 32 tanks. Code re-read: `train_environment.cpp`, `jolt_enemy_tank.cpp`, `environment.cpp` (arena), `liquid_ppo_trainer.cpp` (GAE).

## 1. The 10° potential lever (507): inconclusive for its objective

Announced objective: a better-aligned turret (threshold fixed before the run: turret<5° > 18 %).

| save (index) | turret<5° 507 / 504, side 550 | side 1000 | slope μ_cx/10° 507 / 504 (550) |
|---|---|---|---|
| 2 (54 k) | 9.0 / 10.0 % | 6.0 / 9.9 % | 0.036 / 0.030 |
| 5 (135 k) | 13.2 / 13.8 % | 9.6 / 12.7 % | 0.051 / 0.057 |
| 9 (243 k) | 14.9 / 12.5 % | 13.4 / 10.1 % | 0.076 / 0.062 |
| 12 (324 k) | 13.9 / 14.5 % | 11.6 / 13.1 % | 0.116 / 0.085 |

- Alignment is the same over 4 saves. The corrector is stronger at save_12 (+35 %) with no effect on turret<5°.
- 507 leads at save_12 on the trigger (ratio 8.7 / 13.6, reserve 21-28 s, hit/shell 18.7 / 11.8 %), but 504 made the same transition at save_9 before falling back. The economy oscillates in both runs: seed noise, not an attributable effect.
- Verdict: the potential does not improve aim. The mechanism is partly explained in §2: Φ pays alignment on the best-aligned enemy wherever it is, hence mostly targets at 200-400 m that are almost never hit.

## 2. What the probes establish (504/507 lineage, identical in both runs)

**F1 — Hitting is a short-range matter.** Hits per shell by distance of the aimed target (side 550, 504 save_12 / 507 save_12):

| target | share of shots | hit/shell | hit/shell if aligned < 5° |
|---|---|---|---|
| < 50 m | 21 / 29 % | 39 / 41 % | 60 / 61 % |
| 50-100 m | 16 / 14 % | 11 / 11 % | 25 / 23 % |
| 100-200 m | 24 / 21 % | 7 / 10 % | 13 / 22 % |
| > 200 m | 39 / 35 % | 7 / 10 % | 12 / 16 % |

Beyond 100 m, even a shot aligned to 5° misses 8 times out of 10: a tank spans ~1° at 200 m, and the 256 px camera for 90° gives 0.35°/px.

**F2 — The near enemy is almost always out of view.** The nearest enemy is within 100 m in 52-91 % of decisions. It is in the camera field (±40°) in only 31-35 % of cases, barely above chance (22 %), and within 10° in 13-15 %. Its median bearing is 65-87°: the turret does not turn toward it. 88-91 % of shots are fired while an enemy is within 100 m, but 60 % aim at a target beyond 100 m.

**F3 — The agent cannot know where the near enemies are.** The only perception is the cannon-mounted camera (90° × 45°). Proprioception contains no enemy, and the liquid memory has a time constant of about 0.6 s. Finding the near enemy requires sweeping with the cannon, i.e. misaligning the weapon. Looking and aiming are the same action.

**F4 — Not exploration noise.** 504 save_16 deterministic: turret<5° 17.7 % versus 14.5 % stochastic. The policy mean itself does not centre.

**F5 — Elevation (hausse) is a second lock.** When yaw is aligned (< 5°) on the near enemy (~40 m), the elevation error is −10 to +16° (p10-p90) and is under 2° in only 18-25 % of cases. The terrain has 200 m of relief over 2 km (`heightmap6`, scale 10 × 200 × 10).

**F6 — The reward rewards the far lottery.** hit and kill are worth as much at 300 m as at 30 m. Φ takes the max over all enemies, so it pays pointing at the far pack that fills the field. The curriculum pushes D toward 1000, hence toward this regime.

**Overall reading.** The bulk of possible hits (< 50 m, 40-60 % per shell) is out of view 2 times out of 3. What is visible is mostly far and hit at 7-10 %. The ~100 tuning runs (entropy, σ, critic, economy, shaping) acted on none of these three factors: information, geometry, elevation. The common ceiling of 10-20 % hit/shell follows.

## 3. Proposals (proposed 2026-11-12, for Samuel to decide)

### P0 — Measure the game's ceiling before any overhaul (probe, no training change)

Add an oracle mode to `arenai_agent_probe`: turret and cannon slaved to the nearest enemy (exact yaw and ballistic elevation), fire when error < 1°. Measure hit/shell and kills by distance, plus two variants: yaw-only oracle (elevation left at 0) and oracle limited to the camera field.

What it settles:
- If the oracle hits at 60 % or more up to 100 m and little beyond, the game is learnable at short range and the lock is information (P1) and geometry (P2).
- If the yaw-only oracle collapses, elevation (P3) is the priority.
- If the oracle hits badly even at short range, the problem is in the game (dispersion, hitbox, shell physics).

Cost: ~½ day, code in `probe/` only. (Done the same day: see `2026-11-12_aim-oracle-p0.md`.)

### P1 — Make off-field enemies visible (game / observation)

Today the agent has no way to know that an enemy is 40 m behind it. Two forms:

- **P1a, radar in the state vector**: for the K = 3-4 nearest living enemies, bearing relative to the turret (cos, sin), normalised distance and height difference. Changes `ENEMY_STATE_VECTOR_SIZE` (`include/`).
  - Warning: this is not the 2026-11-03 refusal (« duplication vision → proprio »), which concerned the alignment of an already visible target. Here the information is absent from vision in 65-70 % of decisions. But it must be validated explicitly.
- **P1b, second camera** fixed to the chassis, low-resolution 360° panorama (e.g. 256 × 16). The agent stays "vision only", looking and aiming are decoupled, but rendering and CNN cost more.

Expected effect: turning toward the near enemy becomes an immediate-credit action (target in field, then firing at 40-60 %), where today it is a blind search.

### P2 — Bring combat back to useful range (geometry / curriculum)

- **P2a (config only)**: bound the curriculum at D 500 instead of 1000. Density ×4 and typical distances ÷2.
- **P2b (game)**: area damage of the shell, over a radius of a few metres, decreasing. A shot at 300 m missing by 3 m yields something: the physical gradient the Cauchy tried to imitate, but paid by the game itself and identical for the player. To check: effect on the desktop game.

### P3 — Elevation (conditional on P0's result)

If the yaw-only oracle loses most hits: game-side assisted elevation (the cannon takes the ballistic elevation toward the point under the reticle, the agent commands only yaw), or less rugged terrain during the curriculum.

### Reward

- Remove Φ, or restrict it to enemies within ~100 m (or weight it by a P(hit | distance) measured by P0). In its current form it pays the far lottery (F6).
- hit and kill stay event-based; no other crutch.

### Proposed order

P0 first (½ day, no hypothesis committed). Then P1 + P2a together if the oracle confirms the game is learnable at short range, with Φ removed. P2a is a simple config change, P1 changes the observation contract.

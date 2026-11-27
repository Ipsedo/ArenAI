# train_471_ppo_liquid

| | |
|---|---|
| machine | ElMatador |
| user | samuel |
| launch | 2026-11-03 09:52:40 |
| commit at launch | `173a0865` — hit: +0.3, damage: -0.2 => attacking is more attractive |
| commits ≤ 90 min after | none |
| diff at launch | unknown (identity recorded after the run ended) |
| raw data | `outputs/train_471_ppo_liquid` on ElMatador |

## Established facts


- 2026-11-03: config.json identical to 470 (diff = output_folder) + local uncommitted commit-to-be `shot_cost` 0.03 subtracted at impact in `detail.aim` (see 470 facts). (see docs/decisions.md: fire cost removed 2026-11-07)

  | index | fire/s | hit | kill | s | Hd | D |
  |---|---|---|---|---|---|---|
  | 10 k | 0.58 | 4.0 % | 0.15 | 24 | 0.31 | 250 |
  | 60 k | 0.08 | 2.2 % | 0 | 28.6 | 0.26 | 250 |
  | 100 k | 0.025 | 4.6 % | 0 | 28 | 0.27 | 250 |
  | 150-190 k | 0.015-0.02 | 4-7 % | 0 | 28 | 0.30-0.33 | 250 → 300 (curriculum on hit ratio, agent mute, like 465) |

  Episodes 28 s = starvation 30 s. Absolute hits ÷ 20 (0.023 → 0.001/s per tank).
- 2026-11-03: probe save_1 (108 k, 8 episodes side 550, 75 k decisions, 189 shots, 3 hits (2 %)): p_fire(can_fire) 0.002, median 0.001, 0 % of decisions > 0.5 ⟹ in deterministic game never fires. False positive to avoid: vs raw best_angle, p_fire 0.065 (0-2°) / 0.05 / 0.04 / 0.015 / 0.001 (> 20°) looks like ×50 selectivity, but 94 % of lines with p > 0.05 are at step < 10: **restricted to step ≥ 10, p_fire = 0.001 in all slices, 0-2° included**. The only remaining shot is a spawn reflex (1st second, p 0.05), confounded with spawn geometry. Always condition on step before reading p_fire vs angle. CSV hit rise (2 % → 4-7 %) is that reflex (ratio per episode on ~1 shot per tank, noisy; r_hit 0.00004 flat, kill 0).
- 2026-11-03: **turret unlearned**: corr(yaw, μ_canon_x) |yaw| < 40° = −0.001 (467: −0.31; 470: −0.25), μ_cx constant +0.13 (slow uniform rotation), per-tank range 0.75 (470: 1.9), enemy in field 4.3 % (470: 28 %), best_angle < 5° visited 0.6 % of time (470: 10.5 %). μ_canon_y +0.37 (canon permanently raised). Cauchy aim reward only paid at shell landing: without shots, no aim gradient, the turret drifts. Hd 0.24 → 0.33 from 100 k = zoom head (p_zoom 0.24), not the trigger.
- 2026-11-03: why: same trajectory as 465 despite simultaneous cost/gain payment: the paying state (aligned+near, 1.2 % of shots) is not recognised, every current shot is losing (−0.003), the trigger extinguishes in 60 k then aim extinguishes for lack of shots (loop: no shot → no aim reward → random turret → aligned never visited → shot never profitable). A uniform shot cost cannot work while aim depends on shots. Run to stop (nothing will come out). Next levers (to validate by Samuel; vision-observation duplication excluded): aim reward decoupled from the shot (paid continuously on alignment, but camping seen in event-based reward rework), ramp on shot_cost (0 while hit < threshold), scripted trigger during aim learning. Probe: always read p_fire vs angle at step ≥ 10.

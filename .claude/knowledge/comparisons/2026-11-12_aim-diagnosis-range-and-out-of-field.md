# Aim diagnosis: hit/shell ceiling = information + geometry (range, out-of-field)

Runs: 504, 507

2026-11-12: lineage hit/shell ceiling = information + geometry: hits < 50 m at 40-60 %/shell, > 100 m at 7-10 %; close enemy out of field 65-70 % of the time; proposals P0 oracle, P1 radar/360 camera, P2 D 500 / area damage, P3 elevation aid.

Established on probes 504/507 (save_2 to 16, + deterministic):
- hit/shell by target distance: < 50 m 39-53 % (60-77 % aligned < 5°), 50-100 m 11-16 %, > 100 m 4-10 % (aligned 12-22 %); 60 % of shots aim > 100 m.
- nearest enemy < 100 m in 52-91 % of decisions, in field ±40° 31-35 % (chance 22 %), median bearing 65-87° ⟹ the turret does not turn toward it; the proprio contains no enemy, camera = canon (looking = aiming).
- deterministic ≈ stochastic (tour<5° 17.7 vs 14.5 %): not a noise problem.
- elevation: yaw-aligned on the nearest, |pitch| < 2° only 18-25 % (terrain 200 m relief).
- Φ (max over all enemies) pays the far lottery.

**Why:** ~100 runs of tuning (entropy, σ, critic, economy, shaping) touched neither information, geometry nor elevation.

Proposed to Samuel 2026-11-12 (nothing decided): P0 oracle in the probe (game ceiling by distance, with/without elevation aid); P1 radar K enemies in the state (distinct from the refusal of duplicating vision observation: info absent from vision) or 360° hull camera; P2a bounded curriculum D 500, P2b area damage; P3 assisted elevation if P0 designates it; remove/restrict Φ. Report: outputs/comparisons/2026-11-12_diagnostic_visee_refonte.md. (see docs/decisions.md 2026-11-03: alignment/target info in proprioception refused; 2026-11-23/25 panorama/presence head validated later)

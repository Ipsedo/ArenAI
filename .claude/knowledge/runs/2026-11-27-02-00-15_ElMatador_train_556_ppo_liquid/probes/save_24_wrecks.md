# 556 save_24 — trigger on wrecks vs living tanks (2026-11-27)

Samuel saw the agent fire at wrecks in game (2026-11-27). Question: does the trigger tell a wreck from a
living tank? Probe side 1000, 4 episodes, 32 tanks, **deterministic**, train rules, binary = `5367a45e` + the
wreck export (`TrainTankEnvironment::wreck_sightings`, columns `wreck_angle`, `wreck_distance`: best-aligned
dead tank off the gun axis). Armed decisions, step ≥ 10 (112 021); a wreck exists for 97 % of them.

Facts read in the code at `5367a45e`: a killed tank is never respawned before the episode ends (only a dry
tank is); `JoltEnemyTank::on_death` kills every life item, so a shell on a wreck pays no hit and no shell;
`arenai_view` has no dead state, a wreck is drawn with the colours of a living tank.

| object on the gun axis (< 3°), nothing else within 10° | distance | n | fires | p_zoom | median \|μ_canon_x\| |
|---|---|---|---|---|---|
| living enemy | < 60 m | 228 | 0.74 | 0.90 | 0.06 |
| wreck | < 60 m | 417 | 0.37 | 0.84 | 0.08 |
| living enemy | 60-100 m | 648 | 0.62 | 0.90 | 0.06 |
| wreck | 60-100 m | 586 | 0.47 | 0.86 | 0.06 |
| living enemy | 100-200 m | 649 | 0.21 | 0.80 | 0.13 |
| wreck | 100-200 m | 331 | 0.17 | 0.81 | 0.13 |
| living enemy | > 200 m | 1 479 | 0.05 | 0.76 | 0.82 |
| wreck | > 200 m | 486 | 0.03 | 0.74 | 0.72 |
| nothing within 20° (no wreck, no enemy) | — | 38 936 | 0.012 | 0.73 | 0.84 |

- Shells: 14 % of all shells (5 097) are fired with a wreck within 5° of the axis and no living enemy within
  10°. Of the shells fired with no living enemy within 20° (1 030): wreck within 5° 33 %, wreck at 5-20° 20 %,
  no wreck within 20° 47 % (map items, per the frames of `save_24_offtarget_frames.md`).
- Armed time spent with a wreck on the axis (< 3°, no enemy < 10°): 1.6 %; with a living enemy on the axis: 5.7 %.

## Verdict

Established: the agent engages a wreck with the same posture as a living tank (turret stopped, zoom on) and
fires at 50-80 % of the rate it fires at a living tank at the same distance (0.37 vs 0.74 under 60 m, 0.47 vs
0.62 at 60-100 m). The trigger discriminates weakly. The wreck identification inferred from the frames is
confirmed. Not established: whether the residual discrimination comes from the image (detached parts) or from
memory (the liquid state after its own kill); the same measurement on 554 / 555 (needs their binaries with the
export).

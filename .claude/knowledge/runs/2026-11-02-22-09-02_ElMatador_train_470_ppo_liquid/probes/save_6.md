# Probe 470 save_6 (~700 k) — 2026-11-03

`arenai_agent_probe`, side 550, 6 episodes, 22.5 k decisions, CSV + scripts in scratchpad of session 031f6285. Compared with 467 save_3.

- p_fire(can_fire) 0.68 (467: 0.78), median 0.978, 70 % of decisions > 0.5 (467: 79 %) ⟹ in deterministic the game fires at every can_fire = 6/s, reserve emptied in ~2 s.
- vs best_angle: 0.74 / 0.70 / 0.70 / 0.66 / 0.67 / 0.73 / 0.70 from 0-2° to 40-90°: flat like 467. Cauchy + hit 0.3 created no « target on axis » condition.
- Only non-firing state (p < 0.1, 24 % of can_fire, 467: 12 %): step ≈ 4 (first 2 s), |μ_canon_x| 0.87, p_zoom 0.02 — « sweep the turret at spawn »; identical fraction (0.38-0.41) at 0-4° and 15-25°. From step 20, p 0.9 whatever happens; first shot median step 6.
- Reserve: p 0.53 with initial reserve, 0.86-0.94 at ≤ 1 shell ⟹ inverse of rationing.
- Turret weaker than 467: corr(yaw, μ_cx) |yaw| < 40° = −0.19 (467: −0.31), bias +0.05 on all slices; enemy in field 25 % (467: 31 %).
- Faster emptying: reserve 0.18 → 0.07 in 5 s (467: 0.19 → 0.13).

# Shot-level measurement (probe 470 save_6, 4173 shells, 2026-11-03)

`arenai_agent_probe` now writes `<output>_shots.csv` (one shell per line: best_angle, best_distance, distance/yaw/pitch of nearest and reserve at fire, landing step, size of landed group this step, hits, miss). FIFO attribution fire → landing via `TrainTankEnvironment::last_step_reward_details()` (nb_landed_shells, hit = 0.3/hit, 2/kill). Limit: a shell leaving the arena flies its full 20 s life (speed 250 m/s, ~11 % of shots) and a shorter later shot lands before it ⟹ 10-20 % of hits misattributed, which **attenuates** the contrasts below, does not create them. Script analyze_shots.py (scratchpad of session 031f6285).

Results (8 episodes side 550, 4317 shots, 313 hits, 6.4 %):
- vs best_angle at fire: < 1° 19 %, 1-2° 21 %, 2-5° 10 %, 5-12° 8 %, 12-20° 6 %, 20-40° 2.7 %, > 40° 0.1 %.
- vs distance of centred target: < 20 m 40 %, 20-40 m 15 %, 40-60 m 7 %, > 60 m 3-6 %.
- aligned < 5° AND target < 40 m: 50 % (n = 49); aligned but > 100 m: 7 %.
- shot split: aligned+near 1.2 %, aligned+far 14 %, 5-20° 55 %, > 20° 30 %. The « centred » target is usually far (best_d median > 150 m) while the nearest is ~50 m: it fires at what crosses the axis, not at the nearest.
- vs reserve: tanks > 10 shells (so having already hit) 14 %, initial reserve 3.6 %, ≤ 1 shell 0.4 %.

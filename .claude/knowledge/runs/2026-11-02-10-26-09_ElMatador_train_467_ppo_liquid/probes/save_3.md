# Probe 467 save_3 (340 k) — offline, 2026-11-02

Tool: `arenai_agent_probe` (CMake target, `arenai_agent/probe/main.cpp`) loads `actor.pt` of a save, replays the training env and writes per decision and per living tank: distance/yaw/pitch of the nearest enemy in the canon frame, `best_angle` (most centred living enemy), can_fire, shells, p_fire, p_zoom, μ/σ/turret action. `TrainTankEnvironment::aim_geometry()` provides the geometry. Analysis scripts in session scratchpad (analyze_probe.py, analyze_closing.py): p_fire bins, signed closing E[Δyaw·sign(yaw)] split turret command / rest (chassis + enemy).

Measures 467 save_3 (340 k), side 550, 6 episodes, ~25 k decisions; 250 identical:
- Trigger: p_fire 0.78 when can_fire (median 0.975); can_fire only 18 % of time (cooldown 1/6 s + reserve) ⟹ cadence bounded by the game, not the policy. p_fire vs nearest enemy: 0.88 at < 10°, 0.70 behind; vs most centred enemy (`best_angle`): flat 0.74-0.81 from 0-2° to 90-180°. With 1 shell left: 0.86; at initial reserve: 0.65. No rationing, no « target on axis » condition.
- Turret: correct sign (corr(yaw, μ_canon_x) −0.27 under 40°) but max command −1.8°/decision out of 7.6 possible (≈ 20 % of speed), dead zone under 6° (−0.1 to −0.3), chassis/enemy motion cancels half ⟹ net closing 3-8°/s. Nearest enemy in field 31 % of time, an enemy within 5° of axis 10 % of time.
- save_0 = init (p_fire 0.100 everywhere, μ ≈ 0): the method fabricates no correlation.
- Distance of nearest at fire: p25/p50/p75 = 18/35/59 m. With tol = 10 m + 6°·d, tolerance is 14 m ≈ 20° at 35-40 m: the 10 m floor (chosen 2026-10-29 for 100 m+ shots) flattens the reward at real engagement distance (shot at 20° = 0.036, perfect = 0.09).

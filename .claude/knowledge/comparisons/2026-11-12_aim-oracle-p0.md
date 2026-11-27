# Aim oracle (P0) — game ceiling and value of perception

Runs: 507 (save_12 policy used for movement and zoom).
Date: 2026-11-12. `arenai_agent_probe --oracle` (build/ binary of 2026-11-12 ~13:40). All tanks use the oracle, which slaves turret and cannon to the nearest living enemy within a cone of `--oracle_fov` around the cannon axis: gain 0.3 per decision (the hinge position motor lags by about 2 decisions), ballistic drop compensation, no movement anticipation. It fires when the error is < 1° in yaw and elevation. With no target in the cone, the policy turns the turret and does not fire. Movement and zoom: policy of 507 save_12. 4 episodes × 32 tanks per configuration, economy 10 / +2 / 20. Raw data: `outputs/oracle/` (`oracle_<fov>_side<550|1000>.{csv,log}`, `*_shots.csv`, `attitude_side*.{csv,log}`, `attitude_side550_det.*`).

## Results

hits/tank/min (throughput) and hits per shell:

| configuration | side 550 | side 1000 |
|---|---|---|
| policy 507 save_12 | 5.4 (19 %) | 1.9 (12 %) |
| oracle 45° (zoomed field) | 10.7 (45 %) | 7.5 (26 %) |
| **oracle 80° (current camera)** | **11.0 (50 %)** | **8.5 (29 %)** |
| oracle 120° | 12.3 (47 %) | 10.3 (34 %) |
| oracle 180° | 14.2 (51 %) | 10.6 (36 %) |
| oracle 360° | 14.7 (53 %) | 11.5 (41 %) |
| oracle 360°, policy elevation | 8.3 (25 %) | 5.0 (16 %) |
| oracle 360°, target at image centre | 13.8 (48 %) | 11.4 (36 %) |
| oracle 80°, target at image centre | 11.6 (40 %) | 8.4 (29 %) |

hit/shell of a perfectly aimed shot by target distance (360°, sides 550 / 1000): < 50 m 97 / 94 %, 50-100 m 71 / 72 %, 100-200 m 36 / 42 %, > 200 m 15 / 9 %.

## What it settles

1. **Perception explains only a small part of the gap.** With the current camera (80°), a perfect shooter does 2× (side 550) to 4.4× (side 1000) better than the policy. Going to 360° then adds only +34-36 %. Most of the gap is aiming precision, not what is out of field.
2. **Elevation is the main cause.** Perfect yaw + policy elevation brings throughput back to 8.3 / 5.0 and hit/shell to 25 / 16 %, close to the policy (19 / 12 %). The policy loses mainly its shots in elevation. Consistent with the probe: yaw aligned on the near enemy, |elevation error| < 2° only 18-25 % of the time.
3. **Camera parallax costs a lot at short range.** The camera is at (0, 4, −20) in the cannon frame and aims at the point 100 m along the axis; `CollisionCamera` moves it closer further when terrain obstructs. A shooter that centres the target in the image (instead of the firing axis) drops from 93-99 % to 61-70 % under 50 m and from 71-77 % to 53-66 % between 50 and 100 m. That is exactly the zone where hits are made. Total throughput moves little here because the oracle is perfect beyond, but for an agent learning "target at centre", the visual landmark is wrong at every distance except 100 m.
4. **Beyond 200 m the game caps at 9-15 % even with a perfect shot**: moving target, flight time, no anticipation. More resolution would only pay on these targets.
5. Side finding: the hinge position motor lags by about 2 decisions. A single-decision command turns the turret only 19-37 % of its nominal travel during that decision, the rest arrives afterwards (steady state: 18°/decision, nominal).

## Origin of the agent's elevation error (probe 507 save_12, attitude columns added)

Decisions where yaw is aligned (< 3°) on the nearest enemy, between 10 and 130 m. Error e = T − C − R, with T the world elevation of the target, C the elevation the chassis attitude gives the cannon, and R the elevation commanded by the agent (chassis-relative).

| side 550 (1000) | std | \|e\| < 2° |
|---|---|---|
| stochastic agent | 10.9° (11.9°) | 19 % (17 %) |
| deterministic agent | 9.7° | 24 % |
| cannon frozen relative to chassis, no command at all | 8.0° (7.6°) | 21 % (20-23 %) |
| cannon stabilised to the horizon, no command at all | 5.5° (4.8°) | 28 % (36 %) |

- The elevation need (T − C, std 8°) comes mostly from chassis attitude (C: std 8°, median pitch 6°, roll 4.6°), more than from target position (T: std 5°).
- The agent follows this need slowly (corr 0.5, slope 0.8) but adds its own agitation of about 10° std unrelated to it. Decision by decision, it does not compensate chassis movements (corr(ΔR, −ΔC) = 0.04-0.07).
- Result: its elevation command is worth no more than a frozen cannon (21 %), stochastic as deterministic. So it is not exploration noise.
- A cannon stabilised in the world, with no elevation command at all, would already do 1.5 to 2× better than the agent (28-36 % versus 17-19 %) and reduce the target to follow to std 5°.

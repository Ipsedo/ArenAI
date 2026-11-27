# 556 save_21 — why the gun is low: diagnosis (2026-11-27)

Symptom: armed gun elevation median −11.4° (554 s17 −2.0°, 555 s17 −3.0°); under 100 m, 57 % of yaw-aligned
shells leave with the nearest target > 2° above the gun axis. Exports: side 1000, 4 episodes, stochastic
(`outputs/train_55{4,5}_ppo_liquid/features/save_17_side1000`, `outputs/train_556_ppo_liquid/probes/save_21_side1000`).
`pitch` = elevation of the nearest enemy in the gun frame (> 0: gun below the target); a negative canon_y
action raises the gun (probe oracle: a = −gain · pitch_error).

## Measurements

1. **Pitch servo exists, right sign.** corr(μ_canon_y, pitch), |yaw| < 10°, d < 150 m, |pitch| < 20°:
   −0.31 (554 −0.34, 555 −0.28). Not a dead or inverted head.
2. **Firing setpoint = target centred in the camera image.** Zero crossing of μ_canon_y vs pitch, by distance
   band, against the pitch at which the target sits at the image centre (camera 4 m above and 20 m behind the
   gun, aimed at (0, 0, 100)):

   | distance | parallax prediction | 554 s17 | 555 s17 | 556 s21 |
   |---|---|---|---|---|
   | < 40 m | +4.4…+4.8° | +3.8° | −7.8° | +6.5° |
   | 40-70 m | +1.7° | +0.7° | −5.8° | +2.0° |
   | 70-110 m | +0.3° | +0.1° | −4.0° | +0.7° |
   | 110-200 m | −0.6° | +0.1° | +0.7° | +2.9° |

   556 and 554 follow the parallax curve under 110 m (556 1.7° lower at < 40 m); 555 holds the gun above the
   target. 556 at 110-200 m stays 3° low.
3. **Resting posture has no restoring force in 556.** No enemy in front (|yaw| > 60°), mean μ_canon_y by relative
   elevation bucket [−40,−25,−15,−8,−3,3,20]°: 556 +0.05 / −0.01 / −0.01 / 0.00 / +0.03 / +0.08 (lowers the gun
   as soon as it is above −3°, never raises it); 554 +0.11 / −0.07 / −0.05 / −0.02 / +0.01 / +0.02 and 555
   +0.16 / −0.02 / −0.04 / −0.03 / −0.01 / +0.04 (raise it between −25° and −3°, equilibrium near 0…+4°).
   Elevation median by step in 556: −7° at 2-5 s, −12° after 60 s.
4. **Oracle pitch only** (`--oracle --oracle_policy_yaw --oracle_policy_fire`, yaw and trigger from the network,
   pitch servoed geometrically with drop on the nearest enemy in view; `save_21_side1000_oracle_pitch`):

   | measure | policy | oracle pitch |
   |---|---|---|
   | hit / shell | 0.185 | 0.281 |
   | hit / shell, target < 100 m | 0.42 | 0.68 |
   | shots / tank-s | 0.46 | 0.34 |
   | hits / tank-s | 0.086 | 0.095 |
   | first empty reserve, median (s) | 17.9 | 34.6 |
   | armed gun elevation, median (°) | −11.4 | +5.0 |

   Share of decisions dry 0.09 → 0.25 with the oracle: not explained.

## Verdict

Established on this save: elevation alone accounts for the hit/shell gap to 554 / 555 (0.185 → 0.281; 555 0.286)
and for the early dump (18 s → 35 s; 554 43 s, 555 35 s). Two components: a firing setpoint that centres the
target in the image (parallax, shared with 554), and a resting posture without return to the horizon (specific
to 556). Not established: why 556 lost the resting return (σ_canon_y median 0.185 vs 0.28-0.32; mask encoder;
seed) and when — `probe-timeline` on saves 5, 9, 13, 17.

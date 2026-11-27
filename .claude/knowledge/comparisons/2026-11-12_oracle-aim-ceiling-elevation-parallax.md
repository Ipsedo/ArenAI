# Probe oracle: aim ceiling, elevation, parallax (507 save_12)

Runs: 507 (save_12)

2026-11-12 — probe oracle (`--oracle`, `--oracle_fov`, `--oracle_policy_pitch`, `--oracle_camera_center`): current camera ×2-4.4 the policy, 360° only +35 %, elevation = main lock, 3rd-person camera parallax costs 30 % under 50 m.

On 507 save_12, hits/tank/min side 550 / 1000: policy 5.4 / 1.9; oracle 80° (current camera) 11.0 / 8.5; 120° 12.3 / 10.3; 360° 14.7 / 11.5; 360° + policy elevation 8.3 / 5.0 (hit/shell 25 / 16 %); centring the target in the image instead of the axis: < 50 m 61-70 % instead of 93-99 %. Perfect shot > 200 m: 9-15 %.

**Why:** Samuel dislikes the radar (« ça substitue la vision ») and P2; he seeks a vision lever. The oracle shows out-of-field is not the bulk of the gap: it is precision, especially in elevation, and the visual reference distorted by the camera (0, 4, −20) aiming at the point at 100 m (+ CollisionCamera bringing it closer).

**Lever proposed:** aiming camera on the canon axis (image centre = firing line), possibly wider default FOV + zoom to aim. Resolution would only pay > 200 m. Report: outputs/oracle/oracle.md. The hinge position motor has ~2 decisions of delay (servo gain 0.3 in the oracle). Linked to comparisons/2026-11-12_aim-diagnosis-range-and-out-of-field.md.

**Elevation decomposition (2026-11-12, columns canon_elevation/canon_relative_elevation/chassis_pitch/chassis_roll in the probe):** parallax is not the cause (median gap ≈ 0, it is a ±5° IQR dispersion). Elevation need std 8° dominated by chassis attitude (relief); the agent follows it slowly (slope 0.8) but adds ~10° of own agitation and does not compensate step by step ⟹ |e| < 2°: agent 17-24 % ≈ frozen canon 21 % < stabilised canon without command 28-36 %. Lever proposed: canon stabiliser (game), then on-axis camera. (outcome: gun laying validated 2026-11-12, see comparisons/2026-11-12_canon-stabiliser-gun-laying.md; later revised by comparisons/2026-11-15_local-optimum-aim-trigger-coordination.md)

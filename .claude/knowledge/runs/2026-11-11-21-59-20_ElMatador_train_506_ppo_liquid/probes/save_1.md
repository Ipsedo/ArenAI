# Probe 506 save_1

- Run: train_506_ppo_liquid (505 + per-dimension entropy target [0, 0, −1, −1, −1])
- Save: save_1, index 27 k, D 300
- Binary: `build/arenai_agent/arenai_agent_probe`, compiled 11/11 22:04:46. Sources date 22:00:01 and tree clean at commit c02137c6, so code is the run's. Diff flagged by script comes from commit dating (aed53757), hence `--force`.
- Sides 550 / 1000, 4 episodes, stochastic, economy 10 / +2 / 20

## Summary

| side | ratio | p_al | p_off | p_arm | hit/sh | h<5° | shots<5° | enrich | empty_s | %empty | %disarmed | shots/s | turret<5° | σyaw° | σ cos/sin/pitch | \|μ\| cos/sin |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 550 | 1.03 | 0.72 | 0.70 | 0.71 | 3.4 % | 7.9 % | 18.1 % | 1.05 | 2.4 | 100 % | 60 % | 1.70 | 17.9 % | 80.8 | 0.23 / 0.26 / 0.21 | 0.07 / 0.10 |
| 1000 | 1.02 | 0.72 | 0.70 | 0.70 | 1.1 % | 2.5 % | 18.0 % | 1.02 | 2.4 | 100 % | 61 % | 1.63 | 18.6 % | 78.6 | 0.23 / 0.26 / 0.21 | 0.08 / 0.11 |

Landmark: 505 save_2 (54 k, D 350), i.e. 2× larger index: p_arm 0.53 / 0.49, hit/sh 4.2 / 1.5 %, empty 3.5 / 3.7 s, turret<5° 15.9 / 16.2 %, σyaw° 77-80, σ 0.28 / 0.25.

## Reading

- **Trigger**: flat (0.70 in all angle slices, ratio 1.0), but rose from 0.5 to 0.70 in 27 k, vs 0.53 for 505 at 54 k. Deterministic: fires everywhere. alpha_fire at floor 1e-4.
- **Dump**: reserve empty at 2.4 s in 100 % of tanks, 60 % of steps disarmed. Faster than 505 save_2 (3.5 s).
- **Aim**: turret<5° 18 % (505 save_2: 16 %), median angle 12°. Within noise.
- **Heads**: σ yaw/pitch fall below init 0.3 (0.21-0.26, estimated H −0.2 to 0). Target −1 not reached, so α stays at floor: intended effect is there. But |μ_yaw| 0.07-0.11, well under σ, and setpoint dispersion stays 79-81°, like 505. Yaw is not yet a setpoint: μ learned no direction.

## Established vs noise

- Established: σ_yaw/pitch freed (0.21-0.26 vs 0.25-0.28 at 505 save_2, at half index), flat trigger rising toward dump (0.70), setpoint dispersion unchanged (≈ 80°) as |μ| ≈ 0.
- Noise: turret<5°, hit/sh (different indices).

## Next

save_2 (54 k) vs 505 save_2 at same index: |μ_yaw| must exceed σ and σyaw° must fall. Also watch p_arm, rising while α_fire is at floor.

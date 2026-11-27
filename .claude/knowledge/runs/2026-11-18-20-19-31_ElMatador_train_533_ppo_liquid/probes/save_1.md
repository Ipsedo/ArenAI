# Probe 533 save_1: does aim start to learn?

Run `train_533_ppo_liquid` (coloured AR(1) noise ρ 0.9, σ init 0.30, short value head γ 0.9), uncommitted on 68c41e20. save_1, index 27 k, D 250 (curriculum not moved yet at 46 k). Binary `build/arenai_agent/arenai_agent_probe` compiled 2026-11-18 20:04, same commit, no inference diff. The probe samples with coloured noise (measured noise autocorrelation 0.89 at 0.1 s, 0.30 at 1 s). Sides 250 and 550, 4 episodes, stochastic. Controls: save_0 of the same run (untrained, same noise, side 250) and 528 save_1 (same age, white noise, side 250, `sigma_gradients` export of 2026-11-18).

| side | ratio | p_al | p_off | p_arm | hit/sh | h<5° | shots<5° | empty_s | %disarmed | shots/s | tour<5° | σx | \|μ\|cx |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 250 (s1) | 1.02 | 0.62 | 0.61 | 0.61 | 9.5 % | 26 % | 19 % | 3.0 | 37 % | 2.07 | 16.5 % | 0.20 | 0.16 |
| 550 (s1) | 1.02 | 0.62 | 0.61 | 0.61 | 4.9 % | 13 % | 23 % | 2.6 | 49 % | 1.88 | 19.9 % | 0.20 | 0.16 |
| 250 (s0) | 1.00 | 0.50 | 0.50 | 0.50 | 6.1 % | 24 % | 10.5 % | 3.1 | 64 % | 1.09 | 6.2 % | 0.20 | 0.003 |

Script `tour<5°` = angle to best target; rows below use yaw toward the nearest enemy.

## Aim (yaw toward nearest enemy, step ≥ 10)

| | 533 s0 (250) | 533 s1 (250) | 533 s1 (550) | 528 s1 (250, white) | 529 s3 (250) | 532 s3 (250) |
|---|---|---|---|---|---|---|
| yaw < 5° | 2.6 % | 4.4 % | 4.4 % | 5.1 % | 4.7 % | 4.3 % |
| yaw < 20° | 10.6 % | 15.9 % | 15.0 % | 17.6 % | 16.2 % | 15.0 % |
| share of turret μ opposed to yaw (5-40°, < 100 m) | 0.49 | 0.59 | 0.60 | 0.585 | 0.55 | 0.55 |
| corr(command, yaw) 5-40° | 0.00 | −0.25 | −0.22 | −0.20 | −0.11 | −0.12 |
| gain (° of command per ° of yaw) | 0 | −0.008 | −0.007 | −0.007 | −0.005 | −0.004 |
| dwell < 5° (s) | 0.65 | 0.55 | 0.67 | 0.36 | 0.50 | 0.61 |
| passes ≥ 1 s | 19.5 % | 15 % | 23 % | 4.7 % | 11.6 % | 20 % |
| median σ canon_x | 0.20 | 0.20 | 0.20 | 0.43 | 0.22 | 0.21 |
| μ canon_x autocorr at 1 s | 0.69 | 0.43 | 0.53 | 0.10 | 0.05 | 0.05 |

Convention verified on 533 s1: positive canon_x action ⟹ yaw increases (+9.9° in 0.3 s for act > 0.3), so μ opposed to yaw = corrective gesture.

## Reading

- Turret: a proportional corrective reflex already exists (59-60 % of commands in the right direction, 49 % at save_0; s.e. ≈ 0.5 point), but it is at 528 save_1's level at same age (58.5 %): usual start, not a measurable coloured-noise effect. Gain −0.008: for 20° gap the command is 0.16°/decision, order of magnitude unchanged since 523.
- Time on target: yaw < 5° 4.4 % vs 2.6 % untrained and 5.1 % for 528 s1. Nothing new.
- Dwell: 0.55-0.67 s but save_0 gives 0.65 s knowing nothing. At this age dwell measures σ (0.20) and turret slowness, not learning; it will discriminate only once the μ sweep appears (528/529: after D 500).
- Mechanical effect of coloured noise, established: turret keeps its heading ~1 s (action autocorr 0.90 / 0.28 at 0.1 / 1 s; white: 0.18 / 0.02). μ itself became smooth (autocorr 0.43-0.53 at 1 s vs 0.05-0.10): observation trajectories more coherent, the intended condition for different events to emerge from noise.
- Trigger: flat at 0.61 at any angle, dump 2.6-3.0 s, disarmed 37-49 %, deterministic = fires everywhere. Normal at 27 k (528 s1-s5 same).
- Shot: hit/shell 9.5 % at 250, 26 % for a shell fired < 5°; 19 % of shots aligned vs 10.5 % untrained (turret gain, not trigger).

## CSV at same index

| index | run | kill | hit | fire/s | kl | skip | Hc |
|---|---|---|---|---|---|---|---|
| 27 k | 533 | 16.3 | 0.079 | 2.10 | 0.067 | 0.35 | −0.12 |
| 27 k | 528 | 14.1 | 0.084 | 1.04 | 0.034 | 0.00 | 0.23 |
| 27 k | 529 | 9.4 | 0.048 | 0.82 | 0.034 | 0.00 | 0.06 |
| 27 k | 532 | 8.4 | 0.051 | 0.44 | 0.049 | 0.01 | −0.04 |
| 45 k | 533 | 21.9 | 0.102 | 2.06 | 0.066 | 0.41 | −0.19 |
| 45 k | 528 | 18.6 | 0.118 | 1.41 | 0.042 | 0.00 | 0.29 |
| 45 k | 529 | 8.7 | 0.058 | 0.98 | 0.045 | 0.00 | 0.11 |

Start in the high part of the series (kill 21.9 at 45 k), without the 50 k plateau of 529/532 which share σ 0.2. The kill surplus comes from cadence ×1.5-2 (fire 2.1/s), not hit per shot (0.10 vs 0.12 for 528). Vigilance: kl 0.066 and skip 35-41 % from 10-27 k, when 528/529 were at skip 0 until ~150 k: a third of actor updates already cut by the KL stop.
Conclusion: established: coloured noise works as planned; the turret's corrective reflex starts, at the usual save_1 level. Not established: an aim advantage over white-noise runs (at 27 k / D 250 no previous run stood out either; too early). Weak encouraging signs: fast CSV start despite σ 0.2, corr command/yaw −0.25 (strongest of series at this age, gap within noise), temporally coherent μ. Unfavourable sign: KL skip 40 %.
To follow: save_3 (81 k) and save_5 (135 k), sides 250 + 550: share of μ opposed to yaw (marker 0.55 for 529/532 s3 — previous runs fell between s1 and s3; a rise would be the first real sign), gain, yaw < 5°; save_9 (243 k): dwell and sweep vs 528 s9 (0.26 s) / 529 s9 (0.53 s), script tour<5° vs 14-15.6 %; follow skip/kl in CSV.

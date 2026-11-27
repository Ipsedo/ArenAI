# Probe chronology 541: save_3 to save_26, side 550

Run `train_541_ppo_liquid` (539 config + canon credited at one step on H, critic head), running, 703 k at reading. Start details: save_3.md. Probe binary `build/arenai_agent/arenai_agent_probe` of 2026-11-20 01:39 for all saves, no inference diff. Side 550, 4 episodes, stochastic. Control: 539 s5 / s8 / s10 / s12 same side (single advantage).

## Table (side 550)

| save | index | D | kills | yaw < 5° | dwell (s) | corrective | command (°) | σx | p_fire < 2° / > 45° | shots < 5° | hit/sh | shots/s | empty (s) | disarmed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 541 s3 | 81 k | 350 | 15.3 | 4.2 % | 1.13 | 0.51 | 0.56 | 0.07 | 0.80 / 0.67 | 22 % | 7.1 % | 1.09 | 2.6 | 65 % |
| 541 s5 | 135 k | 400 | 16.3 | 6.5 % | 0.75 | 0.59 | 0.74 | 0.15 | 0.46 / 0.37 | 18 % | 9.1 % | 0.90 | 4.1 | 58 % |
| 541 s9 | 243 k | 600 | 27.5 | 12.4 % | 0.78 | 0.58 | 1.73 | 0.08 | 0.61 / 0.35 | 37 % | 15.8 % | 1.71 | 4.7 | 29 % |
| 541 s12 | 324 k | 750 | 25.8 | 10.7 % | 0.64 | 0.53 | 2.54 | 0.05 | 0.68 / 0.17 | 40 % | 17.4 % | 1.22 | 9.7 | 25 % |
| 541 s16 | 432 k | 750 | 2.5 | 4.1 % | 0.67 | 0.51 | 0.55 | 0.12 | 0.05 / 0.00 | 33 % | 23.6 % | 0.04 | 151 | 0 % |
| 541 s20 | 540 k | 750 | 27.0 | 11.3 % | 1.28 | 0.57 | 1.21 | 0.02 | 0.48 / 0.12 | 38 % | 16.8 % | 1.37 | 11.0 | 6 % |
| 541 s26 | 702 k | 800 | 20.8 | 11.0 % | 0.75 | 0.56 | 2.07 | 0.16 | 0.25 / 0.02 | 54 % | 28.8 % | 0.48 | 37.5 | 2 % |
| 539 s5 | 135 k | 400 | 20 | 9.1 % | 0.73 | 0.59 | 0.99 | 0.08 | 0.78 / 0.67 | 27 % | 9.1 % | 2.20 | 2.8 | 38 % |
| 539 s8 | 216 k | 550 | 25.5 | 8.1 % | 0.49 | 0.56 | 1.72 | 0.22 | 0.38 / 0.18 | 25 % | 12.9 % | 1.43 | 9.5 | – |
| 539 s10 | 270 k | 650 | 23 | 5.7 % | 0.28 | 0.52 | 3.17 | 0.23 | 0.19 / 0.05 | 29 % | 18.9 % | 0.64 | 27.1 | – |
| 539 s12 | 324 k | 800 | 21 | 5.5 % | 0.24 | 0.51 | 3.13 | 0.20 | 0.11 / 0.03 | 30 % | 20.9 % | 0.49 | 48.6 | 4 % |

Yaw, dwell, corrective, command: nearest enemy < 100 m, step ≥ 10 (`aim_lin.py`).

## Reading per axis

- Aim: dead at s3, starts between s3 and s9: aligned time 4 → 12 %, corrective 0.51 → 0.58-0.59. Then holds apart from collapse: 11 % and 0.56-0.57 at s20 and s26, where 539 fell back to 5.5 % and 0.51 from s10-s12. Dwell 0.64-1.28 s vs 0.24-0.28 s.
- Gesture: corrective plateaus at 539's early level (0.56-0.59) and μ gain on yaw stays weaker (−0.008 to −0.011 vs −0.014 to −0.023). Aligned time comes mostly from holding the target, not a franker gesture.
- Trigger: selective on angle from s9, more and more: p_fire < 2° / > 45° 0.61 / 0.35 → 0.25 / 0.02. Share of shots at < 5° 37-54 % (539: 25-30 %).
- Dump: resorbed: empty 2.6 s → 11-37 s, disarmed 65 % → 2-6 %. Shot: hit/shell 16-17 % at s9-s20, 29 % at s26 (cadence three times lower).
- Instability: s16: trigger extinct (armed p_fire 0.003, 0.04 shot/s, 2.5 kills), turret back to 4 % and corrective 0.51. Back whole at s20.

## Link with CSV

D 800 at 328 k like 539, but kill 25 held from 165 k to 345 k (539: 25 → 17 from 270 k to 345 k). Two cycles: fire 0.9 → 0.04/s and kill 25 → 4-6 from 360 k to 480 k (D 850 → 700), recovery to 25 from 495 k to 600 k (D 700 → 850), then fire 0.02-0.1/s and kill 9-10 from 630 k (D 900 → 800). During high phases: Hc −0.8 to −1.3, kl 0.03-0.09, `skip_canon` up to 10 %. During low phases: Hc −0.1 to −0.3, kl 0.012-0.02. s26 (CSV kill 10, D 800) makes 20.8 kills at side 550: the ongoing CSV fall is at least partly the trigger closed at range, not aim loss. s16 was a real extinction.
Established: aim started after s3 and holds at s20 and s26 above 539 at equal save (aligned time ×2, dwell ×3, aligned shots 37-54 %); trigger selective; dump gone. Noise: probe kills 27 vs 23-25 (one run per arm, 4 episodes). Not settled: cause of trigger extinctions (360 k, 615 k); holding at D 1000. Possible next: probe s14 (378 k) and s23 (621 k) to date the order trigger / turret in the two collapses; side 800-1000 on s20 and s26 for the trigger at range.

## Why the trigger closes (2026-11-20; probes s13-s15, s22-s24 side 550; s12, s20, s23, s26 side 1000)

Armed decisions, step ≥ 10. "aligned" = < 5° from a target < 100 m; "off-axis" = ≥ 20°. Logit = ln(p / (1 − p)) of mean p_fire.

| save | index | D | kills 550 | logit aligned | logit off-axis | gap | p_fire aligned < 40 m | armed aligned time | armed off-axis time | tour < 5° |
|---|---|---|---|---|---|---|---|---|---|---|
| s12 | 324 k | 750 | 25.8 | +1.03 | −1.66 | 2.7 | 0.84 | 6.0 % | 52 % | 17.2 % |
| s13 | 351 k | 800 | 23.3 | +0.56 | −2.52 | 3.1 | 0.71 | 3.8 % | 57 % | 11.9 % |
| s14 | 378 k | 850 | 7.5 | −1.04 | −4.78 | 3.7 | 0.35 | 0.9 % | 85 % | 2.7 % |
| s15 | 405 k | 800 | 6.8 | −2.17 | −5.60 | 3.4 | 0.16 | 1.7 % | 85 % | 4.8 % |
| s16 | 432 k | 750 | 2.5 | −2.78 | −6.29 | 3.5 | 0.08 | 1.2 % | 91 % | 2.3 % |
| s20 | 540 k | 750 | 27.0 | +0.06 | −2.05 | 2.1 | 0.64 | 5.2 % | 55 % | 20.0 % |
| s22 | 594 k | 800 | 25.8 | +0.13 | −2.68 | 2.8 | 0.69 | 3.8 % | 62 % | 17.7 % |
| s23 | 621 k | 850 | 20.0 | +0.04 | −4.04 | 4.1 | 0.50 | 1.8 % | 82 % | 7.4 % |
| s24 | 648 k | 900 | 12.0 | −1.61 | −5.63 | 4.0 | 0.12 | 3.0 % | 79 % | 8.1 % |
| s26 | 702 k | 800 | 20.8 | −0.80 | −3.95 | 3.2 | 0.33 | 4.8 % | 57 % | 17.2 % |

Hit per aligned (< 5°) shell by target distance, all saves: 0.82-0.99 at < 40 m, 0.51-0.75 at 40-100 m, 0.16-0.42 at 100-200 m, 0.06-0.20 beyond. Side 1000 (kills 21.5 / 22 / 12.8 / 15.5 for s12 / s20 / s23 / s26): same trigger as side 550 at equal save (p_fire aligned < 40 m 0.86 / 0.56 / 0.41 / 0.48), armed aligned time half (3.0 / 2.0 / 0.7 / 2.1 %).

Established:
- The trigger does not lose its selectivity, it shifts en bloc. The aligned / off-axis logit gap stays 3.4-4.1 during both collapses; both levels fall together by about 1 logit per save (27 k). At s16 it still separates aligned from off-axis by a factor 40, but at 0.08 vs 0.002.
- Not a closure at range. In collapse 1 it closes as much at < 40 m (0.84 → 0.08) as at > 200 m (0.36 → 0.03), although an aligned shell at < 40 m hits 82-99 %.
- Not the reward optimum. Aligned shots remain profitable at all distances under 200 m; CSV `r` falls from 0.0013-0.0022 to 0.0002-0.0004 when it closes.
- The turret lets go before. Collapse 1: aligned armed time 6.0 → 3.8 % at s13, aligned trigger still 0.71. Collapse 2: 3.8 → 1.8 % and tour < 5° 17.7 → 7.4 % at s23, aligned trigger intact (logit +0.13 → +0.04), only off-axis trigger fell. The aligned trigger falls only at the next save.
- Both collapses start at the D 800 → 850 passage (366 k and 599 k). The run's empty-reserve penalty (−D / 1000 per empty respawn) is then −0.85.
- Nothing holds the trigger: `alpha_fire` at floor (1e-4) since ~300 k, discrete target 0.01.
- Low-reserve effect: aligned p_fire 0.47 (reserve ≤ 3) vs 0.72 (reserve > 20) at s13.
Proposed reading (not proved): (1) D rises ⟹ less aligned time at range (half at side 1000) and stronger empty penalty ⟹ off-axis shots cost more, off-axis trigger falls (wanted). (2) The turret loses alignment; cause not established. Candidate: H is labelled only on shots made; when the trigger fires only aligned (shots < 5° 37 → 47-59 %), H sees almost no missed off-axis shots and can no longer dig the aligned / non-aligned gap that credits the canon. (3) Armed states become 80-90 % off-axis; the downward push from these states shifts the trigger's common level, aligned states (1-2 % of data) do not compensate. (4) Without shots no kills; curriculum lowers D, aligned time returns, all recovers.
What would settle: evaluate H offline on probe states at s12, s14, s20, s23 (σ(H) aligned vs off-axis): confirms or rules out point 2. Intermediate probes impossible (one save every 27 k): the turret / trigger order rests on one save of gap in each collapse.

## H evaluated offline (2026-11-20, s12, s14, s20, s23, side 550, files `save_K_side550_H.*`)

Probe recompiled in `build/`: loads `critic.pt`, runs the critic beside the actor, column `shot_value` = σ(H) at each decision. Step ≥ 10, 4 episodes.

| | s12 (324 k) | s14 (378 k) | s20 (540 k) | s23 (621 k) |
|---|---|---|---|---|
| σ(H), target < 100 m, at < 2° | 0.133 | 0.180 | 0.229 | 0.177 |
| σ(H), target < 100 m, at 2-5° | 0.104 | 0.170 | 0.163 | 0.130 |
| σ(H), target < 100 m, at 20-45° | 0.016 | 0.014 | 0.022 | 0.012 |
| σ(H), target < 100 m, at > 45° | 0.009 | 0.012 | 0.013 | 0.006 |
| AUC of H on shot touch | 0.83 | 0.78 | 0.81 | 0.83 |
| shots at < 5°: real touch / σ(H) | 0.37 / 0.10 | 0.44 / 0.16 | 0.45 / 0.17 | 0.46 / 0.13 |
| AUC of H on "yaw < 5°", enemy < 100 m | 0.69 | 0.66 | 0.76 | 0.75 |
| corr(ΔH, yaw reduction), one step | +0.005 | −0.000 | +0.032 | +0.017 |
| mean ΔH, yaw closing / opening | 0.0000 / +0.0001 | −0.0010 / 0.0000 | +0.0020 / −0.0022 | +0.0006 / −0.0012 |
| std of ΔH | 0.061 | 0.075 | 0.065 | 0.058 |

ΔH = γ·σ(H(s′)) − σ(H(s)), nearest enemy < 100 m and yaw < 40°.
- Point-2 hypothesis refuted. H does not lose its negatives during collapses: at s14 and s23 it separates aligned from off-axis as well as at s12 and s20 (0.13-0.18 vs 0.01), AUC 0.78-0.83.
- H is miscalibrated (underestimates by factor 2.5 to 4) but orders correctly. The credit being normalised, scale does not matter.
- The one-step credit carries almost nothing on the gesture. The variation of H when yaw closes is at best +0.002, for std 0.06: correlation 0.00 to 0.03. Null at s12 and s14, weak at s20 and s23.
- So H recognises an aligned state, but its step-to-step difference is dominated by something other than turret rotation. 541's target holding does not rest on a strong signal, coherent with a corrective gesture stuck at 0.56-0.59 and with its fragility.
- Cause of the turret's drop at D 800 → 850 remains not established.

# Run 546 — probe chronology save_9 / save_16 / save_20 (then to save_27)

## Header

save_9: 243 k, D 650. save_16: 432 k, D 1000 (reached at 429 k). save_20: 540 k, D 1000. Binary `build/arenai_agent/arenai_agent_probe` recompiled 2026-11-21 18:14 on `1720af12` (run code, clean tree): exact compatibility. Sides 550 and 1000, 4 episodes, 32 tanks, stochastic, economy 10 / 2 / 30. Reference: 545 save_9 and save_16 (`train_545_ppo_liquid/probes/save_9_16.md`), probed with 545's code. 545 save_20 not probed.

## Summary (550 / 1000)

| save | run | shots/s | hit/sh | h<5° | shots<5° | shots ≥20° | touches/tank-s | 1st empty | disarmed | tour<5° | p_al | p_off |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 9 | 546 | 1.34 / 0.72 | 14.2 / 7.5 % | 44 / 32 % | 28 / 20 % | 44 / 54 % | 0.19 / 0.05 | 4.1 / 4.1 s | 45 / 67 % | 11.9 / 7.8 % | 0.91 / 0.92 | 0.41 / 0.28 |
| 9 | 545 | 1.16 / 0.84 | 15.0 / 11.4 % | 43 / 40 % | 30 / 24 % | 40 / 45 % | 0.17 / 0.10 | 5.4 / 6.1 s | 39 / 54 % | 16.9 / 12.4 % | 0.87 / 0.85 | 0.29 / 0.24 |
| 16 | 546 | 0.96 / 0.53 | 25.5 / 23.7 % | 46 / 49 % | 52 / 45 % | 24 / 32 % | 0.24 / 0.13 | 10.2 / 15.5 s | 8.1 / 9.3 % | 20.2 / 13.8 % | 0.45 / 0.32 | 0.06 / 0.04 |
| 16 | 545 | 0.78 / 0.42 | 32.8 / 27.8 % | 60 / 55 % | 51 / 47 % | 30 / 28 % | 0.26 / 0.12 | 20.2 / 33.6 s | 1.3 / 2.7 % | 31.6 / 19.7 % | 0.20 / 0.27 | 0.06 / 0.02 |
| 20 | 546 | 1.04 / 0.61 | 27.0 / 25.0 % | 52 / 43 % | 47 / 56 % | 33 / 21 % | 0.28 / 0.15 | 11.3 / 13.3 s | 2.5 / 2.7 % | 22.3 / 19.6 % | 0.23 / 0.39 | 0.08 / 0.03 |

(touches/tank-s = shots/s × hit/shell.)

save_9 (D 650): full dump, turret late. Empty in 4.1 s both sides, 81-95 % of tank-episodes empty, 45-67 % of steps disarmed. Trigger high everywhere (p_off 0.28-0.41), σ canon very tight side 550 (0.03 / 0.075). Turret < 5° 12 / 8 % of time (545: 17 / 12 %).
save_16 (D 1000): dump resorbing. Trigger lowered and selective (ratio 7.8 / 8.3, p_off 0.04-0.06), hit/shell ×1.8 / ×3.2, disarmed 8-9 %. Turret 20 / 14 %.
save_20 (D 1000): dump resorbed, highest touch throughput. Disarmed 2.5-2.7 %, shots 1.04 / 0.61 per second, touches/tank-s 0.28 / 0.15. Turret 22 / 20 %. Side 550 selectivity falls back (ratio 7.8 → 2.9, p_off 0.06 → 0.08, 33 % of shells at ≥ 20° for 0.3 % touches); side 1000 sharpens (ratio 14, p_off 0.03, 56 % of shells aligned).
Heads: no saturated μ, no σ at ceiling. σ canon bimodal (p10 0.04-0.06, p90 0.59-0.64), wide median side 1000 (0.53 / 0.59 at save_20). Zoom held (p_zoom 0.48-0.50), decoupled from the trigger (corr 0.95 → 0.45). Deterministic at save_20: p > 0.5 on 27 / 23 % of aligned states (545 save_16: 4.5 %).

## Vs 545 at same save

Established (held on both sides, saves 9 and 16): turret less often on target: −5 points at save_9, −11 / −6 at save_16; aligned shot less precise at save_16: 46 / 49 % vs 60 / 55 %; dump resorbed later: disarmed 8-9 % at save_16 vs 1-3 %; 546 gets there only at save_20; higher cadence: +23 to +26 % at save_16; equal touch throughput at save_16 (0.24 / 0.13 vs 0.26 / 0.12): volume compensates precision.
In the noise / not settled: 546 save_20 (0.28 / 0.15) exceeds 545 save_16 and equals 520 s32 (0.276 / 0.142), but without a 545 probe at save_20 the equal-index comparison is missing; selectivity relapse side 550 at save_20: one point, 4 episodes; trigger more audible deterministic: a single reference point on 545's side.
Not measured: share of arena-exit deaths, turret reaction to received-shot direction; the probe exports neither.
Next: save_24 (648 k): touches/tank-s, p_off side 550, tour<5° (reference 545 s16: 32 / 20 %); 545 save_20 with a binary at `d85cc90f`, for equal index at D 1000; counting arena exits and turret yaw − received-shot direction needs two more probe columns: to decide.

## Update: save_24 (648 k) and save_27 (729 k), D 1000

| save | shots/s | hit/sh | h<5° | shots<5° | shots ≥20° | touches/tank-s | 1st empty | disarmed | tour<5° | p_al | p_off | p_arm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 1.04 / 0.61 | 27.0 / 25.0 % | 52 / 43 % | 47 / 56 % | 33 / 21 % | 0.28 / 0.15 | 11 / 13 s | 2.5 / 2.7 % | 22 / 20 % | 0.23 / 0.39 | 0.08 / 0.03 | 0.12 / 0.07 |
| 24 | 0.57 / 0.25 | 33.5 / 33.4 % | 53 / 56 % | 59 / 57 % | 19 / 18 % | 0.19 / 0.085 | 34 / 39 s | 0.7 / 0.5 % | 31 / 17 % | 0.12 / 0.21 | 0.03 / 0.01 | 0.06 / 0.03 |
| 27 | 0.35 / 0.20 | 43.3 / 38.3 % | 64 / 56 % | 65 / 69 % | 17 / 14 % | 0.15 / 0.078 | 58 / 50 s | 0.2 / 0.3 % | 36 / 23 % | 0.05 / 0.09 | 0.02 / 0.01 | 0.04 / 0.02 |

- Aim rising continuously: turret < 5° 22 → 36 % side 550, aligned shot 52 → 64 %, median miss 45 → 16 m. save_27 exceeds 545 save_16 (32 / 20 %, 60 / 55 %) and equals 522 s68's record on aligned shot.
- Trigger closing at constant shape: p_arm ÷3, p_al ÷4.5 side 550, ratio 2.9 → 2.9. Even at [0-2°) p_fire is only 0.11-0.13.
- Touch throughput halved since save_20 (0.28 / 0.15 → 0.15 / 0.078): precision gain does not compensate cadence. CSV: kill 22-24 to 680 k, 17 since 700 k.
- Deterministic: p > 0.5 on 2.6 / 3.6 % of aligned states at save_27 (27 / 23 % at save_20). Dump absent: disarmed < 1 %, 1st empty at 50-58 s.
Vs 545: same trajectory (545 s16: p_arm 0.09 / 0.05; CSV kill 19-22 and fire 0.2-0.3 from 549 to 873 k). 546 held 22-24 from 540 to 680 k then went under 545 (17 vs 20). The criterion set at 548 k (kill ≥ 22 and fire ≥ 0.4 beyond 600 k) is not held.

## Update: ranking of saves (11 probed) and cause of closing

| save | index | shots/s | hit/sh | touches/tank-s | h<5° | disarmed | tour<5° | det. aligned |
|---|---|---|---|---|---|---|---|---|
| 16 | 432 k | 0.96 / 0.53 | 25.5 / 23.7 % | 0.244 / 0.125 | 46 / 49 % | 8.1 / 9.3 % | 20 / 14 % | 37 / 24 % |
| 18 | 486 k | 0.34 / 0.17 | 36.5 / 34.6 % | 0.123 / 0.059 | 58 / 57 % | 0.4 / 0.6 % | 33 / 21 % | 0.4 / 0.5 % |
| 19 | 513 k | 0.77 / 0.41 | 29.7 / 26.8 % | 0.228 / 0.109 | 52 / 50 % | 1.4 / 1.5 % | 28 / 21 % | 6.7 / 5.5 % |
| **20** | 540 k | 1.04 / 0.61 | 27.0 / 25.0 % | **0.281 / 0.152** | 52 / 43 % | 2.5 / 2.7 % | 22 / 20 % | 27 / 23 % |
| 21 | 567 k | 0.63 / 0.41 | 30.1 / 28.0 % | 0.189 / 0.115 | 50 / 53 % | 1.2 / 2.0 % | 27 / 16 % | 8.2 / 10 % |
| 22 | 594 k | 0.48 / 0.29 | 36.9 / 35.4 % | 0.178 / 0.104 | 58 / 60 % | 0.9 / 1.6 % | 23 / 15 % | 9.5 / 9.3 % |
| 23 | 621 k | 0.49 / 0.34 | 38.3 / 32.2 % | 0.187 / 0.111 | 60 / 55 % | 0.7 / 0.7 % | 28 / 19 % | 3.0 / 5.3 % |
| 24 | 648 k | 0.57 / 0.25 | 33.5 / 33.4 % | 0.189 / 0.085 | 53 / 56 % | 0.7 / 0.5 % | 31 / 17 % | 5.7 / 6.5 % |
| 25 | 675 k | 0.26 / 0.14 | 41.0 / 34.3 % | 0.106 / 0.049 | 58 / 51 % | 0.2 / 0.3 % | 25 / 13 % | 2.1 / 3.9 % |
| 27 | 729 k | 0.35 / 0.20 | 43.3 / 38.3 % | 0.153 / 0.078 | 64 / 56 % | 0.2 / 0.3 % | 36 / 23 % | 2.6 / 3.6 % |

(det. aligned = share of aligned states where p_fire > 0.5.) Saves 17 and 26 not probed.
Best save: **save_20**: highest touch throughput on both sides, dump resorbed, trigger audible deterministic. save_16 second (more audible, but 8-9 % disarmed and aligned shot less precise). save_27 the most precise, almost mute deterministic. Trigger level oscillates by a factor 3 in 27 k (0.34 → 0.77 → 1.04 → 0.63 from save_18 to save_21) while aim and precision advance regularly.

### What closes the trigger (545 and 546)

Reward = 3·touch + kill + received + death − empty-reserve penalty (D/1000, i.e. 1 at D 1000). Residual `r − r_hit − r_rcv` ≈ kill + death − penalty, ×1e-3 per tank-decision:

| run | index | fire | r_hit | r_rcv | residual | r |
|---|---|---|---|---|---|---|
| 545 | 319-374 k | 0.92 | 6.4 | −2.1 | −4.2 | 0.1 |
| 545 | 484-539 k | 0.33 | 4.8 | −1.6 | −1.2 | 2.0 |
| 545 | 814-869 k | 0.36 | 4.8 | −1.6 | −0.7 | 2.5 |
| 546 | 374-429 k | 1.00 | 7.5 | −2.5 | −4.7 | 0.4 |
| 546 | 649-704 k | 0.28 | 5.7 | −1.9 | −1.4 | 2.5 |
| 546 | 759-814 k | 0.15 | 2.9 | −1.0 | −0.3 | 1.6 |

- Established: the closing from 1.0 to ~0.3 shot/s is a rise of total reward (0.1-0.4 → 2.5): the avoided penalty (+3.3) exceeds the lost touches (−1.7 to −1.8).
- Established: it is not selective. p_fire at < 2° goes from 0.31-0.48 (save_20) to 0.09-0.17 (save_27) while these shots hit 56-64 %, above the 50 % where a shell repays itself. 17 % of shells still leave at ≥ 20° (touch < 1 %).
- Established: it does not depend on the reserve. p_fire at < 2° with 1-3 shells (0.14-0.15) ≥ with 7-10 (0.09) at save_27; same for save_16 at 24 and 545 save_16. The penalty lowers a global bias.
- α_fire at floor (1e-4) since 380 k, discrete target 0.01: nothing supports the trigger.
- 546 after 700 k: r falls back (2.5 → 1.6), the closing overshoots the reward optimum. 545 had the same trough at 539-594 k (kill 18.8) then went back up. Cause not settled.

## Update: does the critic value the reserve? (save_20, save_27)

Probe recompiled with a `value` column (liquid critic, hidden state reset per episode). V in units of normalised return; return std 1.68-1.73 (`return_scaler.pt`).
Mean V by reserve (sides 550 / 1000, step ≥ 10):

| reserve | save_20 | save_27 |
|---|---|---|
| 1 | −0.52 / −0.48 | −0.48 / −0.32 |
| 3 | −0.44 / −0.34 | −0.36 / −0.17 |
| 4-6 | −0.40 / −0.33 | −0.18 / −0.17 |
| 7-9 | −0.27 / −0.26 | −0.10 / −0.02 |
| 10 | −0.20 / −0.21 | −0.05 / −0.05 |
| 11-15 | −0.10 / −0.09 | +0.08 / +0.10 |
| 16-30 | −0.07 / −0.09 | +0.16 / +0.12 |

- The critic values the reserve: slope ≈ 0.033 per shell from 1 to 10, ≈ 0.056 of reward per shell; almost null above ~12. The penalty does not arrive as a surprise.
- Constant price from 1 to 10 shells: consistent with a trigger independent of reserve in this range and more open at 11+ (p_fire < 2° 0.48-0.50 vs 0.28-0.31 at save_20).
- Break-even at this price, with ~0.5 reward per touch and 2 shells refunded: h* = c / (R + 2c) ≈ 9 %. Shots ≥ 20° (touch < 1 %): −0.05 each. Shots 5-20° (6-9 %): ≈ 0. Shots < 5° (52-64 %): ≈ +0.28 each.
- ΔV at the shot step: save_20 −0.007 to −0.012 whatever the alignment; save_27 +0.006 to +0.012 aligned, 0 to +0.006 off-axis. V aligned − V off-axis appears only at high reserve at save_20 (+0.25 at 11+, 0 at ≤ 10); at save_27 +0.06 to +0.12 from 7-10.
Consequence: the shell price justifies closing off-axis shots (33 % of shells at save_20), not aligned shots. Halving the penalty would only move the threshold from 9 to ~5 %; the bottleneck is the trigger's discrimination (it falls en bloc), not the amount nor the critic's attribution. Not measured: the advantage actually received by aligned shots in training.

## Update: GAE advantage by decision type (save_20, save_27)

Probe with `_rewards.csv` export (reward, done, truncated per step). GAE recomputed offline as in training (γ 0.998, λ 0.95, cut on done or truncated, reward / return std, penalty −1 added to truncations since the probe has no curriculum), then centred and reduced over all decisions. Script `probes/gae_par_decision.py`. Armed decisions, step ≥ 10.
Mean advantage (in std), shot / no shot, sides 550 then 1000:

| class | save_20 side 550 | save_20 side 1000 | save_27 side 550 | save_27 side 1000 |
|---|---|---|---|---|
| < 2° | +1.35 / +0.56 | +2.08 / +0.67 | +1.21 / +0.24 | +1.73 / +0.35 |
| 2-5° | +0.59 / +0.14 | +0.62 / +0.08 | +0.43 / +0.03 | −0.07 / −0.01 |
| 5-20° | −0.33 / −0.15 | −0.37 / −0.09 | −0.63 / −0.07 | −0.90 / −0.06 |
| ≥ 20° | −0.46 / −0.17 | −0.48 / −0.13 | −0.87 / −0.11 | −1.12 / −0.08 |
| < 5° and < 40 m | +1.81 / +0.69 | +2.38 / +1.11 | +1.35 / +0.18 | +2.05 / +0.35 |

Direction of the shot-logit gradient, mean of (shot − p_fire)·A:

| class | save_20 side 550 | save_20 side 1000 | save_27 side 550 | save_27 side 1000 |
|---|---|---|---|---|
| < 2° | +0.027 ± 0.011 | +0.035 ± 0.015 | +0.034 ± 0.006 | +0.023 ± 0.007 |
| 2-5° | +0.005 ± 0.007 | +0.009 ± 0.005 | +0.008 ± 0.002 | −0.002 ± 0.002 |
| 5-20° | −0.001 ± 0.002 | −0.002 ± 0.001 | 0.000 ± 0.001 | 0.000 ± 0.001 |
| ≥ 20° | −0.003 ± 0.001 | −0.001 ± 0.000 | 0.000 ± 0.001 | −0.001 ± 0.001 |

- The credit is good: firing aligned receives +1.2 to +2.1, firing off-axis −0.5 to −1.1. The touch paid at impact does not prevent the shot from being credited above expectation (gap +0.8 to +1.4).
- The gradient says open when aligned at both saves, and close off-axis at save_20. Yet p_fire at < 2° goes from 0.36-0.39 to 0.11-0.13 between the two.
- Not firing when aligned also receives a positive advantage (+0.24 to +0.67): the critic undervalues aligned states. With a shell in flight: +0.6 to +0.9.
Consequence: "touch credit" and "shell price" leads dismissed. The trigger falls against its policy gradient; the cause is in the actor update. Unverified hypothesis: asymmetry of PPO clip on a rare action. A shot's ratio (p ≈ 0.1) reaches 1.2 as soon as p rises to 0.12, then its gradient is cut; a non-shot's ratio cannot exceed 1.11 and is never cut, and these non-shots have positive advantage when aligned. Other candidate: drift of the shared trunk under the other heads.

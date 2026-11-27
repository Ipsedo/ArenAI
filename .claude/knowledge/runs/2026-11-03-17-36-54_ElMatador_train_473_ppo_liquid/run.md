# train_473_ppo_liquid

| | |
|---|---|
| machine | ElMatador |
| user | samuel |
| launch | 2026-11-03 17:36:54 |
| commit at launch | `682bc8f3` — potential reward shaping |
| commits ≤ 90 min after | `28efc583` |
| diff at launch | unknown (identity recorded after the run ended) |
| raw data | `outputs/train_473_ppo_liquid` on ElMatador |

## Established facts


- 2026-11-03: launched 17:36 (config.json identical to 472; diff = commit `28efc583`: `AIM_POTENTIAL_TOLERANCE_DEGREES 20`, `shot_cost 0`, CSV-only metric `phi`). Throughput ~75 k/h.
- 2026-11-03: at 108 k (save_1), CSV: fire 0.7-1.2/s from start (471/472: 0.03), hit 5-7 % (= chance at 32 tanks), kill 0.2-0.6/window (470 at same D: 0.6-0.95), D 250 → 300 at 57 k (470: 75 k), s 15-18 s (dump then empty-reserve death, like 470), kl 0.01, skip 0, Hc 0.21 → 0.13, Hd 0.33 → 0.47 (p_fire rising). `phi` flat 0.345 → 0.36; r_aim −0.0007 = tax −(1−γ)·0.1·Φ (−0.00036) + drop to 0 at terminal: no net improvement of Φ. r (−0.003) ≠ r_aim + r_hit + r_rcv: the rest is empty-reserve death (−0.5) and death (−1).
- 2026-11-03: probe save_1 (4 episodes, side 550, `--features`, scripts in scratchpad of session 54576057): trigger p_fire 0.73 at step ≥ 10 (0.57 at step < 10), 95 % of decisions > 0.5, flat vs best_angle (0.74 at 0-2°, 0.68 at 20-45°, 0.74 at 90°+); shells emptied in 5 s (shells 0 on 86 % of lines, can_fire 0.03 after 5 s); hit 2 % (34/1478). Turret: mu_canon_x = +0.53 constant whatever the yaw sign (|mu| > 0.8: 0.2 %), corr(yaw, mu_cx) −0.02: continuous rotation at half speed (see agent-does-not-aim note); enemy in field 15 %, best_angle < 5° 7.8 %, Φ(20°) estimated 0.42 ≡ untrained 470 save_0 (17 %, 6.9 %, 0.45). Perception: yaw RMSE 25.8-28.6° (std 24.3°), fine zone 16-20° (std 8.8°); AUC aligned < 6° 0.65-0.71 (470 s6: 0.55-0.63, n_test 762, marginal); pitch 6.5° (std 11°) as before.
- 2026-11-03 reading: shot_cost 0 does reopen the trigger (one-way confirmed, see run 471) but the wide tolerance changed nothing about aim in 108 k: the potential is neither learned by the critic (ev 0.4) nor exploited by the actor, which still has no reading of the bearing. **Gate « yaw decodable at save_1 » not reached** for the 3rd time (470, 472, 473). Auxiliary head refused by Samuel (the CNN already converged with the regen; the problem is event density, not perception).

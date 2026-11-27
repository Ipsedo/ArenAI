# Probe 539 save_12: D 800

Run `train_539_ppo_liquid`, save_12 (324 k, D 800 reached at 328 k). Probe binary of 2026-11-19 13:49 (current image), no inference diff. Sides 250, 550 and 800, 4 episodes, stochastic. Control: 539 save_5 (135 k, D 400).

| save | side | ratio | p_al | p_off | hit/sh | h<5° | shots<5° | empty_s | %disarmed | shots/s | σx | \|μ\|cx |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s5 | 250 | 1.14 | 0.83 | 0.73 | 13.2 % | 38 % | 22 % | 3.2 | 27 % | 2.34 | 0.04 | 0.07 |
| s12 | 250 | 6.4 | 0.48 | 0.08 | 21.0 % | 43 % | 30 % | 25.0 | 4.5 % | 0.95 | 0.19 | 0.28 |
| s5 | 550 | 1.24 | 0.84 | 0.67 | 9.1 % | 25 % | 27 % | 2.8 | 38 % | 2.20 | 0.08 | 0.08 |
| s12 | 550 | 16.5 | 0.56 | 0.03 | 20.9 % | 47 % | 30 % | 48.6 | 4.0 % | 0.49 | 0.20 | 0.41 |
| s12 | 800 | 30.3 | 0.69 | 0.02 | 18.8 % | 40 % | 32 % | 45.7 | 3.8 % | 0.32 | 0.20 | 0.45 |

Armed p_fire by angle alone (all distances), s12: 0.11 / 0.07 within 2° (sides 550 / 800), 0.03 / 0.02 beyond 45°. Deterministic (p > 0.5): 23 % / 10 % / 5 % of aligned & close decisions (sides 250 / 550 / 800). Kills per probe episode, s12: 23-28 (250), 17-24 (550), 14-19 (800).

| aim | s5 (250 / 550) | s12 (250 / 550 / 800) |
|---|---|---|
| dwell < 5° (s) | 0.66 / 0.73 | 0.26 / 0.24 / 0.22 |
| passes ≥ 1 s | 21.7 / 24.1 % | 4.3 / 3.9 / 3.1 % |
| yaw < 5° | 7.5 / 9.1 % | 5.6 / 5.5 / 5.0 % |
| corrective turret μ, 5-40° | 0.561 / 0.587 | 0.536 / 0.511 / 0.512 |
| mean command (°/decision) | 0.85 / 0.99 | 2.51 / 3.13 / 3.37 |

- Dump gone: disarmed 4 % (27-38 % at s5), 1st empty at 25-49 s.
- Trigger very selective but on distance: 0.48-0.69 aligned within 40 m, 0.05-0.11 aligned all distances. Cadence 0.3-0.5 shot/s at sides 550 and 800.
- Touches per shell 19-21 % on the three sides (13 / 9 % at s5), 40-47 % for an aligned shot.
- Target holding lost: dwell 0.22-0.26 s, under 538 (0.30-0.34). σ canon back up from 0.04-0.08 to 0.19-0.20 (ceiling 0.25), command ×3, corrective 0.51-0.54 (chance = 0.5).
- Kills at side 800: 14-19, limited by volume of shots at range.
To follow: CSV at 414 k: D fell back to 750, kill 12.9, fire 0.09/s; intermediate probe (save_8 or save_9, D 600-650) to date the σ canon rise and the dwell loss relative to the trigger closing.

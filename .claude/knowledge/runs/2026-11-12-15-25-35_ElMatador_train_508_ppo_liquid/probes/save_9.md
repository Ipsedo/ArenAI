# Probe 508 save_9 (243 k, D 650) vs 507 save_9 — stabiliser verdict

`build/` binary of 12/11 15:24 (run code, stabiliser included), no inference diff. 4 episodes per side, stochastic, economy 10 / +2 / 20. Same index and same D as 507 save_9 (D milestones identical up to 650: curriculum does not separate the two runs).

| run | side | p_arm | ratio | p_al / p_off | hit/sh | h<5° | shots<5° | enrich | empty_s | %disarmed | shots/s | hits/tank/min | turret<5° |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 508 | 550 | 0.16 | 6.8 | 0.60 / 0.09 | 15.2 % | 33.2 % | 23 % | 1.89 | 6.5 | 30 % | 0.89 | 8.0 | 13.1 % |
| 507 | 550 | 0.17 | 4.5 | 0.56 / 0.13 | 9.9 % | 21.1 % | n/a | n/a | 4.3 | 41 % | 0.87 | 5.1 | 14.9 % |
| 508 | 1000 | 0.07 | 11.6 | 0.51 / 0.04 | 7.9 % | 23.3 % | 19 % | 2.10 | 13.8 | 29 % | 0.47 | 2.2 | 9.6 % |
| 507 | 1000 | 0.12 | 4.7 | 0.46 / 0.10 | 5.1 % | 10.8 % | n/a | n/a | 4.1 | 38 % | 0.66 | 2.0 | 13.4 % |

## Agent's aim (test fixed at save_5)

| | yaw gain (μ_cx per 10°, ±40°) | pitch \|e\| < 2° (turret < 3°, target 10-130 m) |
|---|---|---|
| 508 save_5, 550 / 1000 | −0.059 / −0.031 | 28 % / 27 % |
| 508 save_9 | −0.062 / −0.028 | 24 % / 24 % |
| 507 save_5 | −0.051 / −0.026 | n/a |
| 507 save_9 | −0.076 / −0.050 | 21 % / 20 % |
| 507 save_12 | n/a | 19 % / 18 % |

- **Turret correction gain did not increase.** Stays at −0.06 / −0.03 between save_5 and save_9, below 507 at same save. Turret<5° falls to 13.1 / 9.6 %, under 507 (14.9 / 13.4 %).
- **Pitch keeps its mechanical advantage, but it shrinks.** 24 % vs 20-21 % for 507, whereas 28 % at save_5. Pitch gap grows (std 10-12° vs 9-11° at save_5): agent's own agitation increases.
- By the criterion fixed at save_5 (yaw gain clearly beyond −0.06, pitch command tracking the target), **the stabiliser did not make aiming learn**.

## What progresses: the trigger

- 508 left the dump between save_5 and save_9. Reserve empty at 6.5 / 13.8 s (4.4 s at save_5) and 30 % of steps disarmed, vs 38-41 % for 507. The transition 504 made at the same save and 507 did not.
- Selectivity rises: ratio 6.8 / 11.6, vs 4.5 / 4.7 for 507 and 6.1 / 7.0 for 504 save_9. 23 % / 19 % of shots leave aligned within 5°, enrichment 1.9-2.1 (1.1 at save_5).
- Hits per shell 1.5× those of 507 (15.2 / 7.9 % vs 9.9 / 5.1 %), at 504 save_9's level (16.1 / 7.6 %). Aligned shots hit better: 33 / 23 % vs 21 / 11 % for 507, and 29 / 16 % for 504. Under 50 m, an aligned shot hits 74 % / 70 %, vs 64 % / 66 % for 507.
- Hits per tank per minute: 8.0 / 2.2 vs 5.1 / 2.0 for 507. +58 % at mid range, +8 % at long range.

## Verdict

- **Established:** the stabiliser improves what it mechanically corrects. An aligned shot leaves truer, especially in pitch and at short range. CSV kills keep about +25 % over 507 at equal D (16.9-17.9 vs 13.1-15.4 from D 500 to 650).
- **Established:** it does not unlock aim learning. Turret gain did not move, and pitch agitation rises. The aim bottleneck was thus not canon physics (drift, delay): it is in learning (turret credit via critic, per-step signal).
- **Undecided:** exit from the dump. 504 made it at the same save without stabiliser, 507 not. Remains compatible with seed noise on transition timing, even if more profitable aligned shots make it plausible.

## Next

save_12 (324 k) and save_13: check that the economical trigger holds (504 lost it later) and whether turret<5° keeps falling. The aim lever remains to be found on the learning side, not mechanics.

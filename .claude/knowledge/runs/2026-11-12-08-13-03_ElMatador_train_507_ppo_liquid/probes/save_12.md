# Probe 507 save_12 (324 k, D 800-850) vs 504 save_12 (D 800)

`build/` binary of 12/11 08:01 (`--force` for both: 507 = same code, 504 = velocity return + SAC additions without effect on the liquid actor). 4 episodes per side, stochastic.

| run | side | p_arm | ratio | p_al / p_off | hit/sh | h<5° | empty_s | %disarmed | shots/s | turret<5° | \|μ\|cx | corr(yaw, μ_cx) ±40° (slope /10°) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 507 | 550 | 0.06 | 8.7 | 0.31 / 0.035 | 18.7 % | 29.7 % | 20.8 | 20 % | 0.49 | 13.9 % | 0.47 | −0.34 (−0.116) |
| 504 | 550 | 0.14 | 6.7 | 0.53 / 0.078 | 14.4 % | 24.3 % | 7.2 | 32 % | 0.83 | 14.5 % | 0.58 | −0.22 (−0.085) |
| 507 | 1000 | 0.03 | 13.6 | 0.29 / 0.021 | 11.8 % | 22.2 % | 28.1 | 15 % | 0.28 | 11.6 % | 0.47 | −0.26 (−0.077) |
| 504 | 1000 | 0.10 | 7.9 | 0.55 / 0.070 | 7.0 % | 11.6 % | 7.8 | 33 % | 0.60 | 13.1 % | 0.57 | −0.15 (−0.054) |

CSV at 330 k: 507 D 850, hit 15 %, kill 9, fire 0.29; 504 D 800, hit 12 %, kill 10, fire 0.55.

- 507 left the dump between save_9 and save_12 (reserve 4 s → 21-28 s), like 504 between save_5 and save_9 (then 504 fell back: 22 s at save_9, 7 s at save_12). Timing of this transition oscillates in both runs: seed noise, not an effect of the lever.
- Most selective trigger of the lineage at side 1000 (ratio 13.6) and hit/shell 18.7 / 11.8 %; compare 504 save_16 (D 950: 11.2 %).
- Turret: corrector gain +35-40 % (slope 0.116 vs 0.085), but turret<5° identical (13.9 / 11.6 % vs 14.5 / 13.1 %). Stronger corrector does not translate into alignment.

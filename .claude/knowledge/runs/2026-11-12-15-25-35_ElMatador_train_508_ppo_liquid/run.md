# train_508_ppo_liquid

| | |
|---|---|
| machine | ElMatador |
| user | samuel |
| launch | 2026-11-12 15:25:35 |
| commit at launch | `25a75a34` — backtrack to velocity turret move, and add aim potential reward |
| commits ≤ 90 min after | none |
| diff at launch | unknown (identity recorded after the run ended) |
| raw data | `outputs/train_508_ppo_liquid` on ElMatador |

## Established facts

- 2026-11-12: Run = 507 + GunLaying stabiliser + 20 Hz hinges (first test of the canon stabiliser; oracle said pitch was the main cost). Probe save_5 vs 507 save_5 (same index, same D 450): turret <5° 17.4/13.1 % (507: 13.2/9.6); pitch |e| <2° 28/27 % (507 save_12: 17-19 %, prediction for stabilised barrel without command 28-36 %); hits/tank/min 7.5/2.4 (507: 4.5/1.8); hit/shell unchanged (9.9/4.3 %); shots/s 1.26/0.93 vs 0.75/0.60; reserve empty 4.4 s vs 6-8 s; trigger ratio 1.9-2.2 vs 2.9-3.7. Agent pitch command still uncorrelated with target (gap std 9-11° > 5° of the barrel left as is). D milestones identical to 507 up to 450. CSV kills +20-45 % at equal D. Pitch of 508 is read with `attitude.py` (the « if barrel were stabilised » line meaningless under stabiliser). Single seed: increased fire rate may be noise.
- 2026-11-12: Verdict save_9 (243 k, D 650, 18:50): yaw gain −0.062/−0.028 unchanged (507 save_9: −0.076/−0.050), turret <5° 13.1/9.6 % < 507 (14.9/13.4); pitch <2° 24 % (507: 20-21 %, 508 save_5: 28 %) ⟹ stabiliser did NOT make aiming learn, bottleneck = learning. But exits dump (reserve 6.5/13.8 s, ratio 6.8/11.6, hit/shell 15.2/7.9 % ≈ 504 save_9, hits/tank/min 8.0/2.2 vs 5.1/2.0) — transition timing perhaps seed.

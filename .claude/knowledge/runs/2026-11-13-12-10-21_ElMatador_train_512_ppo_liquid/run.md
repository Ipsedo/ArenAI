# train_512_ppo_liquid

| | |
|---|---|
| machine | ElMatador |
| user | samuel |
| launch | 2026-11-13 12:10:21 |
| commit at launch | `c43516da` — ad PPG support for probe |
| commits ≤ 90 min after | `68b9697d` |
| diff at launch | unknown (identity recorded after the run ended) |
| raw data | `outputs/train_512_ppo_liquid` on ElMatador |

## Established facts

- 2026-11-13: Run 512 = 511 + respawn delayed 10 s in place, cut at respawn (commit `68b9697d`; validated 13/11, awaiting 511 save_9). 511's dump equilibrium broken: trigger learns at 510's pace, reserve lasts slightly longer. Dump not gone (p50 3-4 s). CSV at 108 k: `s` 43 s, fire 0.69/s (510: 1.47), hit 17 % (510: 14.5 %), kills 17. 512 probes run under the respawn rule: %disarmed, shots/s and hit/shell not comparable with 510/511 probes (made without respawn); only 1st empty reserve and trigger are. The 12:05 binary is wrongly dated to the previous commit by the script (false diff) ⟹ `--force`.
- 2026-11-13: save_5 (135 k): trigger takes off like 510 (ratio 1.65/1.49, p_arm 0.42/0.35), 1st reserve 3.7/3.2 s (510 3.1/3.0; 511 2.3), 92-97 % still empty; precision behind (h<5° 21/15 vs 37/24), confounded with the probe rule. Analysis `outputs/train_512_ppo_liquid/probes/save_5.md`.
- 2026-11-13: save_9 (243 k, D 650), 510 save_9 re-probed under 512's rule: 1st reserve 7.1/10.5 s (510: 9.5/14.6), ratio 4.0/4.3 (510: 5.9/10.9), h<5° 24/21 % (510: 33/25), better turret (<5° 33/25 vs 24/19). Same D milestone. 511's dump is fixed but 512 does no better than 510; >80 % of tanks still empty their 1st reserve in both. Analysis `probes/save_9.md`.
- 2026-11-13: save_16 (432 k, first save at D = 1000 in both runs; D reached at 429 k vs 428 k for 510), 510 save_16 probed under same rule: 512 behind 510: ratio 6.2/11.2 (510: 13.6/16.9), 1st reserve 6.0/8.2 s (510: 7.9/11.5), hit/shell 20/16 % (510: 25/21), disarmed 39/37 % (510: 26/22). 512 ahead on turret: <5° 39/33 % (510: 33/26). σ canon collapsed in 512: 0.12-0.28 vs 0.54-0.60 for 510; CSV Hc −0.22 vs +0.11, kl 0.21 vs 0.05. μ dir_x saturated in 30-40 % of states. 512's lower CSV `fire` was an artefact of the 10 s disarmed: under same rule 512 fires more (0.76 vs 0.59/s). Verdict: delayed respawn does not bring the hoped economy; rule choice left to Samuel. Analysis `probes/save_16.md`. Outcome: delayed respawn reverted (510's rule restored) for run 513 (see run 513).

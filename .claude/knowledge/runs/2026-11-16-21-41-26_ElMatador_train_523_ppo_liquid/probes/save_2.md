# Probe 523 save_2 (54 k, D 350)

Run 523, ppo_liquid, effective commit 5c338159 (7-conv CNN + Flatten, single advantage). Binary `build/arenai_agent/arenai_agent_probe` of 2026-11-16 21:46, compiled after last source changes (21:38) ⟹ run code. The script dates it cfb76a2e and flags a diff (`train_environment.cpp`, `sigma_gradients.cpp`): aim-reward accessors removed, no inference effect; run with `--force`. Sides 550 and 1000, 4 episodes, 32 tanks, stochastic. Decoding: `--features` side 550, save_2 (185 k decisions) and control save_0 (212 k), `decode_gpu.py` of 522.

| side | ratio | p_al | p_off | p_arm | hit/sh | h<5° | shots<5° | empty_s | %disarmed | shots/s | tour<5° | σx | σy | \|μ\|cx |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 550 | 1.04 | 0.62 | 0.60 | 0.60 | 0.072 | 0.190 | 0.168 | 3.0 | 0.70 | 1.14 | 0.130 | 0.47 | 0.50 | 0.23 |
| 1000 | 1.04 | 0.62 | 0.60 | 0.60 | 0.024 | 0.071 | 0.140 | 2.7 | 0.72 | 1.02 | 0.106 | 0.47 | 0.52 | 0.21 |

Early-probe references, side 550:

| probe | idx | p_arm | ratio | hit/sh | h<5° | shots<5° | tour<5° | %disarmed | σx |
|---|---|---|---|---|---|---|---|---|---|
| 522 save_1 | 27 k | 0.63 | 1.01 | 0.054 | 0.133 | 0.201 | 0.170 | 0.57 | 0.23 |
| **523 save_2** | 54 k | 0.60 | 1.04 | 0.072 | 0.190 | 0.168 | 0.130 | 0.70 | 0.47 |
| 522 save_3 | 81 k | 0.67 | 1.05 | 0.103 | 0.274 | 0.204 | 0.135 | 0.61 | 0.24 |
| 514 save_3 | 81 k | 0.81 | 1.03 | 0.079 | 0.190 | 0.239 | 0.202 | 0.58 | 0.20 |
| 510 save_3 | 81 k | 0.80 | 1.05 | 0.150 | 0.242 | 0.314 | 0.350 | 0.89 | 0.18 |

Reading: trigger flat at 0.60 over all angle slices (0.595-0.605), ratio 1.04 on n = 581/183 aligned-close; no selectivity as every run at this stage; lowest level of the series (0.63-0.81 elsewhere), consistent with lower CSV fire; deterministic it fires everywhere (p > 0.5 in 97-98 % of decisions). Dump: 100 % of tank-episodes empty, 1st empty at 3.0/2.7 s, 70-72 % of time disarmed, identical to early 510-522. Shot: hit/shell 0.072 and h<5° 0.190 side 550, between 522 save_1 and save_3. Aim: tour<5° 0.130 (522 save_3: 0.135), no turret led onto the target. Heads: σ canon 0.47-0.52, twice predecessors (0.18-0.24); no μ saturation, nothing at ceiling.

## Decoding (linear probe, side 550)

| target | features | 523 save_2 | 523 save_0 | 522 save_68 (550) | 522 save_0 |
|---|---|---|---|---|---|
| left or right, \|yaw\| > 3° (AUC) | liquid / CNN | **0.745 / 0.789** | 0.59 / 0.60 | 0.58 / 0.56 | 0.55 / 0.67 |
| signed yaw, in field (R²) | liquid / CNN | **+0.11 / +0.15** | −0.05 / −0.02 | −0.01 / −0.08 | −0.07 / +0.04 |
| signed yaw, target < 100 m (R²) | liquid / CNN | +0.13 / +0.18 | −0.03 / 0.00 | +0.01 / +0.02 | −0.07 / +0.07 |
| fine yaw (R²) | liquid / CNN | +0.13 / +0.23 | −0.15 / −0.19 | +0.04 / −0.09 | −0.28 / −0.29 |
| fine site (R²) | liquid / CNN | +0.20 / +0.24 | −0.04 / +0.04 | +0.22 / +0.16 | −0.04 / +0.11 |
| aligned < 2° & target < 100 m (AUC) | liquid / CNN | 0.83 / 0.83 | 0.64 / 0.64 | 0.86 / 0.82 | 0.59 / 0.67 |
| aligned < 2° vs 2-10° (AUC) | liquid / CNN | 0.575 / 0.60 | 0.54 / 0.58 | 0.76 / 0.71 | 0.49 / 0.48 |

p_fire AUC (armed): aligned < 2° & < 100 m 0.635 (save_0: 0.48; 522 s68: 0.83). corr(yaw, μ canon_x): −0.20 in field, −0.26 fine (save_0: −0.05/−0.07; 522 s68: −0.15/−0.19). corr(site, μ canon_y) fine: −0.34.

Established: (1) yaw sign became decodable: left/right 0.75-0.79 at 54 k vs 0.56-0.63 on 522 after 1.8 M and 0.59-0.60 on the untrained control of the same CNN; gain learned, not architectural at init. (2) The command starts to follow it: corr(yaw, μ canon_x) −0.20/−0.26 from 54 k, at or above 522 save_68. (3) "Near tank on axis" detector already there (0.83; 522 s68 0.86) but trigger does not exploit it yet (0.635 vs 0.83 decodable). (4) Fine discrimination (< 2° vs 2-10°) not learned (0.58-0.60).
Noise / not settled: yaw R² +0.11 to +0.23 positive everywhere but weak (a linear probe on a flattened output reads position poorly); nothing yet translates into behaviour (tour<5°, hit/shell in early band); attribution: flat CNN and single advantage confounded (the other PC's decoding would settle); 4 episodes, one side for decoding.
Next useful: same probe + decoding at save_4-5 (108-135 k, D 400-450): does left/right rise, does corr yaw/μ canon_x pass −0.3, does tour<5° leave 0.13? Same decoding on the other PC's run at same index.

# train_542_ppo_liquid

| | |
|---|---|
| machine | ElMatador |
| user | samuel |
| launch | 2026-11-20 11:46:27 |
| commit at launch | `cc4ee96b` — add new graphic parameters to GUI |
| commits ≤ 90 min after | none |
| diff at launch | unknown (identity recorded after the run ended) |
| raw data | `outputs/train_542_ppo_liquid` on ElMatador |

## Established facts

- 2026-11-20: run 542 (ppo_liquid, launched 2026-11-20 11:46, `cmake-build-release`, uncommitted): sole config difference with run 541 = `discrete_target_proba` 0.01 → 0.05. Probes `outputs/train_542_ppo_liquid/probes/save_{3,5,7}_side550.*`.
- 2026-11-20 (200 k, D 500, identical milestones): the 0.05 target does not act yet: entropy target 0.20, Hd 0.66-0.68 ⟹ `alpha_fire` falls to floor at the same pace as 539/541 (2e-4 at 200 k); it bites only if p_fire drops under ~0.05; until then 542 = another seed of 541. kill 24-25 from 60 to 180 k (541: 19-21, 539: 24-26), then 21 at 200 k with fire 1.5 → 0.7-0.9/s, Hc −1.3 → −0.3 to −0.5, `r` 0.0028 → 0.0003. Probes side 550 s3/s5/s7: whole dump (empty 2.3-2.4 s, disarmed 49-63 %; 541 s9 had resorbed it); trigger flat then selective at s7 (ratio 1.05 / 1.08 / 1.27, off-axis p_fire 0.74 → 0.66); corrective 0.56-0.57; μ/yaw gain −0.004 flat (541: −0.002 → −0.007 → −0.011 from s3 to s9); yaw < 5° at < 100 m 7.7 / 8.1 / 5.8 %; dwell 1.3 s; probe kills 22 / 24.5 / 19.
- 2026-11-20 (305 k, D 750, D milestones ≡ 541): kill 18-22 since 200 k (541: 25-26), fire ~1.0/s, `r` < 0 from 210 k to 305 k (541: +0.001), hit 0.09-0.11 (541: 0.12-0.14), `alpha_fire` at floor 1e-4 from 270 k, Hd 0.61-0.67. Probes s9 / s11 side 550: increasingly selective trigger (ratio 1.90 / 2.72, off-axis 0.38 / 0.27, aligned 0.72-0.74) but dump barely dented (empty 3.2 / 4.7 s, disarmed 56 / 37 %; 541 s9-s12: 11-37 s, 2-6 %); turret still without gesture (gain −0.005 / −0.002, 541 s12 −0.010; corrective 0.55 / 0.54; yaw < 5° at < 100 m 7.3 / 6.1 % vs 10.7-12.4 %); probe kills 20.5 / 25.
- 2026-11-20: s13 (351 k, D 800): 542 rejoins 541's trajectory with ~4 saves of lag (542 s13 ≈ 541 s9): dump resorbed (empty 12.7 s, disarmed 17 %), ratio 5.0 (aligned 0.54, off-axis 0.11), hit/shell 15.6 %, h<5° 37.6 %, gain −0.008, passes 225 → 489, probe kills 23; CSV 345-360 k: kill 22, `r` back > 0, D 800 since 328 k. Reading for "does H act or is it seed noise?": reproduced on 540/541/542 = the credit on H first extinguishes the gesture (gain −0.002 to −0.004 vs −0.010 for 539 s3); the gesture returns when the trigger becomes selective and the dump resorbs (541 s5-s9, 542 s11-s13), variable date = seed. corr(ΔH, yaw reduction) 542: −0.00 / 0.02 / 0.03 / 0.06 (s5/s9/s11/s13), 541 ≈ 0 ⟹ the gesture's return is not explained by ΔH. Not settled: whether H makes aim hold after D 800 (539 lost it at s12, single run).
- 2026-11-20 (537 k, s16, s19; D 950 since 438 k): D 850 → 900 → 950 passed without brutal extinction (kill 21-22 to 430 k, where 541 fell to 5), then slow slide: fire 0.8 → 0.06-0.2/s, kill 22 → 8-9 at 510-535 k, CSV hit 0.17 → 0.23, `r` stays > 0. Probes 550 s13 / s16 / s19: aligned p_fire < 40 m 0.54 / 0.45 / 0.17, off-axis 0.11 / 0.030 / 0.0095 (same en-bloc slide as 541), mean armed p_fire 0.063 → 0.024; hit/shell 16 / 20 / 28 %, shots < 5° hit 38-47 %; dump gone (empty 80 s, disarmed 1 %); turret gain fell back −0.008 → −0.004 → −0.002; probe kills 23 / 20 / 12.8 (side 950: 8.5). **The 0.05 target finally acts** (`alpha_fire` 1e-4 → 4.5e-4 from 450 k) but does not hold the trigger: α too small against the advantage. Judge nothing before s9-s12; the target is judged only at the D 800 → 850 passage and only if p_fire falls toward 0.05; compare at equal D (bruit-de-seed-comparer-a-d-egal).

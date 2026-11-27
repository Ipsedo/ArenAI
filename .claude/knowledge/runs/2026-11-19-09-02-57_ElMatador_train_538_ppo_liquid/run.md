# train_538_ppo_liquid

| | |
|---|---|
| machine | ElMatador |
| user | samuel |
| launch | 2026-11-19 09:02:57 |
| commit at launch | `cc4ee96b` — add new graphic parameters to GUI |
| commits ≤ 90 min after | none |
| diff at launch | unknown (identity recorded after the run ended) |
| raw data | `outputs/train_538_ppo_liquid` on ElMatador |

## Established facts

- 2026-11-19 09:03: run 538 = 537 + `--noise_correlation 0` (sole difference; same training binary of 2026-11-19 07:31): white-noise control of 537. Probe binary `build/arenai_agent/arenai_agent_probe` of 2026-11-19 09:33 (recompiled after adding export fields to `AimGeometry`, no inference diff). Probes s1 (27 k, D 250) and s3 (81 k, D 350), sides 250/550, controls 537 s1/s3, 534 s3.
- 2026-11-19 ~10:20 (verdict of the control, probes/save_3.md): **white noise ≥ coloured noise at equal config.** s3: yaw < 5° 6.9-7.0 % (537: 5.4-5.8), corrective 0.58-0.59 (0.57), aligned shots 22-23 % (13-14), hit/shell 14.3/7.9 % (12.3/6.5), trigger ratio 1.13-1.23 (present without coloured noise), p_al 0.75/0.75, shots/s 2.21/1.78, σx 0.10/0.15, empty 4.0/2.8 s, disarmed 35/51 %. s1: corrective 0.561/0.553, corr −0.26/−0.23, yaw < 5° 4.5/4.5 %, dwell 0.32/0.35 s, command 1.13/1.24°/decision. CSV at equal index (537 / 538): 27 k (D 250): kl 0.039/0.008, clip 0.31/0.11, skip 0/0, kill 6.9/13.7, hit 0.070/0.094, fire 1.32/1.18; 66 k (D 300): kl 0.069/0.011, clip 0.36/0.16, skip 0.05/0, kill 19.7/23.2, hit 0.115/0.117, fire 1.11/1.66; 80 k (D 350): kl 0.066/0.014, clip 0.38/0.16, skip 0.08/0, kill 19.4/22.8, hit 0.111/0.109, fire 1.35/1.78; identical D milestones. Trigger selectivity at s3 comes from batch 2400 or the linear command, not the noise. Gaps of seed-noise size, one run per arm; same direction everywhere. 538 left running.

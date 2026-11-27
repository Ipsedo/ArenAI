# Decoding alignment from features, 522 save_68 (1 836 k)

Question: does the network know where the target is, and does the trigger exploit what it knows?

## Method

`arenai_agent_probe --features` (run binary, 2026-11-15 20:07), 4 episodes, 32 tanks, stochastic. Per decision: liquid output read by the heads (128) and CNN output (384, spatial softmax). save_68 sides 550 (91 k decisions) and 1000 (137 k); save_0 side 550 (213 k) as untrained control. Linear probes trained on 3 episodes, evaluated on the 4th, rotating (script `decode_gpu.py`, torch on GPU). Decisions at step ≥ 10. "In field": nearest enemy at |yaw| < 39.5°, |site| < 22.5°, < 200 m. "Fine": additionally |yaw| and |site| < 10° and nearest = best target. MLP probes (2×256) generalise worse than linear across episodes (negative R²): only linear retained. A numpy ridge probe on 12 k decisions gives same figures ±0.02.

## Results (linear probe)

| target | features | save_68, 550 | save_68, 1000 | save_0 (untrained) |
|---|---|---|---|---|
| signed yaw, in field (R²) | liquid / CNN | −0.01 / −0.08 | +0.02 / −0.06 | −0.07 / +0.04 |
| signed yaw, target < 100 m (R²) | liquid / CNN | +0.01 / +0.02 | +0.06 / −0.01 | −0.07 / +0.07 |
| fine yaw (R²) | liquid / CNN | +0.04 / −0.09 | +0.08 / −0.06 | −0.28 / −0.29 |
| left or right, \|yaw\| > 3° (AUC) | liquid / CNN | 0.58 / 0.56 | 0.63 / 0.58 | 0.55 / 0.67 |
| site, in field (R²) | liquid / CNN | +0.09 / +0.01 | +0.28 / +0.38 | +0.42 / +0.46 |
| fine site (R²) | liquid / CNN | +0.22 / +0.16 | +0.28 / +0.28 | −0.04 / +0.11 |
| aligned < 2° & target < 100 m (AUC) | liquid / CNN | 0.86 / 0.82 | 0.945 / 0.92 | 0.59 / 0.67 |
| aligned < 2° vs 2-10°, target < 100 m (AUC) | liquid / CNN | 0.76 / 0.71 | 0.81 / 0.80 | 0.49 / 0.48 |
| control: μ canon_x (R²) | liquid | 1.00 | 1.00 | 1.00 |
| control: shell ratio (R²) | liquid | −0.01 | −0.05 | +0.03 |

What the policy itself does (AUC of p_fire, armed decisions): aligned < 2° & target < 100 m: 0.83 (550) / 0.94 (1000) / 0.51 (save_0); aligned < 2° vs 2-10°: 0.66 / 0.70 / 0.50.
Correlation between target yaw and μ canon_x command: −0.15/−0.16 in field, −0.19/−0.24 fine (save_0: −0.08/−0.05). Site and μ canon_y fine: −0.35/−0.39.

## Established

1. The network does not know which side the target is on. Signed yaw undecodable (R² −0.09 to +0.08), left/right AUC 0.56-0.63 for chance 0.50. Untrained CNN gets 0.67. Same as 2026-11-03 on 470 (left/right 0.68), on the best model to date.
2. It knows a near target is on the axis: "aligned < 2° and target < 100 m" decodes at 0.86/0.945; control 0.59-0.67. A detector without sign: presence of a tank near image center.
3. The trigger exploits almost all of that detector: p_fire AUC 0.83/0.94 vs 0.86/0.945 decodable. On fine discrimination (< 2° vs 2-10°) a gap remains: 0.66/0.70 exploited vs 0.76/0.81 decodable, and the decodable itself is far from 1.
4. Site better perceived than yaw in fine aim (R² 0.22-0.28, control ≈ 0) and the command follows (corr −0.35 to −0.39). Site-in-field R² is not learning: control does better (0.42-0.46, chassis pitch effect).
5. Shell ratio, present in proprio, not decodable from the liquid output (R² ≤ 0), as on 2026-11-03.

## Limits

Linear probes miss highly nonlinear codings (MLPs did not do better but generalise badly here); label = nearest enemy (with several in field the network may track another); closed loop (a controller corrects the error part it perceives, reducing what remains decodable; but turret within 5° only 30 % of time, a perceived slowly corrected error would stay decodable); 4 episodes per file.

## Reading

The bottleneck is upstream of the trigger: the target's lateral position is not in what the network extracts from the image. The turret cannot be brought onto the target; alignment comes from sweeping and chance (turret < 5° 25-33 % of time over twenty runs), and the trigger fires when the "tank at center" detector lights up. Makes coherent the frozen acquisition (near enemy in field 38 % of time), the trigger gaining nothing under oracle aim, and the common plateau 510-522.
Gap with the 2026-11-03 reading: undecodability had been attributed to event rarity. Here events are dense (kill 19-21 per episode, hit 0.38-0.46) and yaw stays undecodable.

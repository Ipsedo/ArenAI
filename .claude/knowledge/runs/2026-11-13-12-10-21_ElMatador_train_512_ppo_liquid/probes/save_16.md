# Probe 512 save_16 (432 k, D 1000) vs 510 save_16, same rule

D = 1000 reached at 429 k by 512 and at 428 k by 510: save_16 is the first save at D = 1000 in both runs.

`build/` binary of 12:05, with the code of 68b9697d (respawn delayed 10 s in place). Both checkpoints launched in `--force`, for the same reasons as save_9:
- 512: false dating diff;
- 510: only diff = respawn rule.

Both policies therefore play under the same rule. 4 episodes per side, stochastic, 32 tanks.

| run | side | p_arm | ratio | <5°-only ratio | empty_s p10/p50/p90 | %empty | %disarmed | shots/s | hit/sh | h<5° | shots<5° | enrich | turret<5° | σ cx/cy | p_zoom |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 510 | 550 | 0.09 | **13.6** | **7.1** | 3.4/**7.9**/20.8 | 74 % | 26 % | 0.59 | **24.9 %** | **34.6 %** | 60 % | 2.60 | 32.6 % | 0.54/0.40 | 0.55 |
| 512 | 550 | 0.17 | 6.2 | 5.3 | 2.3/6.0/21.1 | 79 % | 39 % | 0.76 | 20.0 % | 30.5 % | 53 % | 2.18 | **38.8 %** | **0.12/0.12** | 0.75 |
| 510 | 1000 | 0.05 | **16.9** | 5.1 | 3.1/**11.5**/75.1 | 80 % | 22 % | 0.37 | **20.8 %** | **30.8 %** | 56 % | 2.63 | 26.1 % | 0.60/0.46 | 0.50 |
| 512 | 1000 | 0.10 | 11.2 | 4.8 | 2.4/8.2/31.7 | 85 % | 37 % | 0.54 | 15.9 % | 26.4 % | 52 % | 2.25 | **32.8 %** | **0.28/0.20** | 0.71 |

Aligned & <40 m counts: 510 376/330, 512 475/168. 512's ratio at 1000 is therefore noisy. <5° only: 12-22 k.

**Heads:**
- **512, movement saturated:** μ dir_x saturated in 30-40 % of states (|μ| 1.1-1.3, mean −1.1/−1.2), i.e. full chassis rotation on one side only.
- **510:** dir_y saturates (8 %), i.e. forward driving.

**CSV at 432 k:**

| | Hc | kl | skip_canon | ev |
|---|---|---|---|---|
| 510 | 0.11 | 0.05 | 0.25 | 0.59 |
| 512 | **−0.22** | **0.21** | 0.38 | 0.74 |

## Reading

- **Trigger:** 510 stays ahead on selectivity (ratio 13.6/16.9 vs 6.2/11.2). 512 fires about 2× higher off-axis (p_off 0.068/0.045 vs 0.033/0.024). Save_9's gap persists at equal D.
- **Economy:** 510 holds its 1st reserve longer (7.9/11.5 s vs 6.0/8.2 s), with fewer disarmed steps (26/22 % vs 39/37 %). In both runs, 74-85 % of tanks still empty their 1st reserve.
- **Precision:** 510 better (hit/shell 25/21 % vs 20/16 %; h<5° 35/31 vs 31/26).
- **Aim:** 512 better on turret (39/33 % vs 33/26 %), as at save_9. Only axis where it leads.
- **New fact, σ canon:** 512's σ canon collapsed (median 0.12/0.12 at 550, 0.28/0.20 at 1000) vs 0.54-0.60 for 510. CSV goes the same way: Hc −0.22 vs +0.11, kl 4× higher, skip_canon 0.38. 512 converges tighter on canon and movement, with a tenser KL regime.
- **CSV misled:** its lower `fire` for 512 (0.19 vs 0.67/s at 485 k) came from the 10 s disarmed, not a more economical trigger. Under the same rule, 512 fires more than 510 (0.76/0.54 vs 0.59/0.37 /s).

## Established / noise

- **Established (two saves, two sides, same rule):** at equal D, 512's policy is not more economical than 510's. It is less selective, less precise, and holds its reserve less long. Delayed respawn did not bring the hoped gain.
- **Constant (save_9 and save_16):** 512's turret is better.
- **One seed per run:** reserve gaps stay in seed-noise range. Ratio and hit/shell gaps are consistent on all four measures.
- **σ canon / Hc collapsed in 512:** cause not established (respawn rule or seed). To watch (cf. Hc waves of 496/500).

## Next

- Comparison at save_28 no longer needed to settle delayed respawn. It would only show whether 512's σ canon drop resolves or degrades aim.
- Training-rule choice for Samuel: return to 510's rule (immediate respawn, cut at empty), or keep 512 for train/game gap despite lagging metrics.

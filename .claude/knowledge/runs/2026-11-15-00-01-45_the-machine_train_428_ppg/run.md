# train_428_ppg

| | |
|---|---|
| machine | the-machine |
| user | samuel |
| launch | 2026-11-15 00:01:45 |
| commit at launch | `0e5093a1` — backtrack higher resolution : real problem is in discrete target entropy value |
| commits ≤ 90 min after | none |
| diff at launch | unknown (identity recorded after the run ended) |
| raw data | `outputs/train_428_ppg` on the-machine |

## Established facts

_Run 428 PPG (0e5093a1, vs 427: discrete_target_proba 0.05→0.01 + γ 0.998, confounded) — α_fire/zoom at floor all run; switch 560-620k: fire 0.9→0.2-0.4/s, hit 0.17→0.26, kill 22→14-17; probe save_27: ratio 8.6/21.9 but p_al halved, deterministic mute_

Run 428 PPG launched 2026-11-15 00:01 on 0e5093a1 (high confidence), analysed at 739k (running).
Vs 427 (d1bb4020): discrete_target_proba 0.05 → 0.01 and gamma 0.995 → 0.998, **confounded**; ev now per optimisation batch.
D milestones identical to the index for 424 to 428 (curriculum at max speed) → compare at D=1000 at equal index.

- α_fire/α_zoom at floor 1e-4 from 64k (427: α_fire up to 0.017) → discrete constraint inert.
- 640-740k vs 423/427: hit 0.232 (vs 0.20/0.15), miss 54 m (vs 67/89), fire 0.44 (vs 0.83/1.09), kill 16.9 (vs 23).
- Probe save_27 (729k): ratio 8.6 / 21.9 (423: 4.9 / 7.5), p_off 0.017 / 0.007, p_al 0.15 (423: 0.36 / 0.29),
  hit/shell 0.27, h<5° unchanged 0.35, aligned shots 65 % (vs 47-50 %), %empty 0.49 / 0.39 (vs 0.70), disarmed 1-3 %,
  deterministic: fires in 4-7 % of aligned states.

**Why:** hit gain = shot selection (off-axis extinguished, dump removed), not ballistics; trigger globally low → fewer kills.
**How to apply:** do not read the hit rise as better aim; watch p_al and fire (< 0.15/s = extinguished trigger); γ vs discrete-target attribution requires a separate run. See run 423.

**Probe save_49 (1323k, 2026-11-15)**: h<5° 0.43 / 0.47 (1st precision gain per aligned shot of the lineage, ≤ 0.37 before), turret<5° 0.375 / 0.332,
hit/shell 0.29; zoom abandoned (p_zoom 0.02-0.03, CSV Hd 0.07 from 1.18M, α_zoom ≈ floor); chassis steered to one side (μ dir_x +1.05)
→ turns in circles; ratio 7.0 / 11.1 (flat off-axis floor 0.02-0.04 beyond 10°), deterministic still mute (3-7 %).

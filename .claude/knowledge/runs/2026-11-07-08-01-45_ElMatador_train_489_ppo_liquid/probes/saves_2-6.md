# Probes 489 save_2 to save_6 (54-162 k) — 2026-11-07

Run 489 (C 0.03, p_init 0.5), probes at D=250, 4 episodes, armed decisions at step ≥ 10. Probe CSV: scratchpad `probe_489_save2.csv`.

| save | k | p <5° | p 5-20° | p >20° | gap | reserve emptied | hit/shell | hit <5° |
|---|---|---|---|---|---|---|---|---|
| 2 | 54 | 0.55 | 0.52 | 0.55 | 0.00 | 2.8 s | 13.8 % | 31 % |
| 3 | 81 | 0.47 | 0.43 | 0.46 | +0.01 | 3.5 s | 17.4 % | 38 % |
| 4 | 108 | 0.46 | 0.44 | 0.49 | −0.03 | 3.4 s | 12.7 % | 23 % |
| 5 | 135 | 0.52 | 0.50 | 0.54 | −0.02 | 2.9 s | 16.4 % | 28 % |
| 6 | 162 | 0.56 | 0.53 | 0.57 | 0.00 | 3.0 s | 16.2 % | 29 % |

- Zoom 0.76 → 0.90 (only discrete head that learns). 420-480 shells/episode, 15 % aligned. CSV at 169 k: kill 2.5-3.2, hit 18 %, net per shell +0.003, but Hc 0.20 → 0.035, kl 0.02, skip 2.8 % (start of 488's pattern, with events this time).
- save_2 detail (54 k): p_fire 0.54 / 0.55 / 0.52 / 0.55 (< 5° & < 40 m / < 5° / 5-20° / > 20°), 0.545 at empty reserve, 0.57 at spawn — flat and at init; emptied 125/128 at 2.8 s median, 5.4 fire actions/s; 97 % of tank-steps at empty reserve; hit/shell 13.8 % (30.8 % aligned < 5°, 16.7 % at 5-20°, 2.3 % > 20°), 15.5 % of shots aligned; zoom 0.75-0.78.
- **Hd ≈ H(zoom).** `discrete_action_mask` (torch_converter.cpp) masks only fire; metric Hd = sum of masked entropies / nb masked, so fire weighs only its armed fraction (3-30 %). Hd 0.06-0.20 of 488 and 0.30 of 489 = the saturating zoom, not the trigger. Read p_fire at the probe only.
- Five flat points with ~500 aligned decisions per probe (detectable gap ≈ 0.07): at mean-zero force (net ≈ 0) the fire head does not move at all, neither in level nor contrast, although the reward contrast (+0.03 aligned vs −0.02 off-axis per shell) exists and shots are not lacking. The advantage signal-to-noise is the lock, not C.
- The save_8 gate will not change the reading; next lever = amplify ×k the per-shell terms (cost C and damage hit/received) in TrainEnvironment, not in the model (training shaping), kill/death unchanged, k = 3; to validate by Samuel.

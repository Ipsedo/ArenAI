# train_480_ppo_liquid

| | |
|---|---|
| machine | ElMatador |
| user | samuel |
| launch | 2026-11-05 14:57:45 |
| commit at launch | `fc5b0444` — add trained ppo liquid |
| commits ≤ 90 min after | none |
| diff at launch | unknown (identity recorded after the run ended) |
| raw data | `outputs/train_480_ppo_liquid` on ElMatador |

## Established facts


- 2026-11-06: 478 config unchanged + `actor_epochs` 2 (critic 4), launched 2026-11-05 14:57 in `outputs/train_480_ppo_liquid` (repo root, not arenai_agent/outputs). Read 2026-11-06 00:05 at 910 k steps, save_every 108 k.
- 2026-11-06 CSV vs 478 at equal index: skip **0 all run** (478: 0 → 14 %), kl 0.004-0.005 i.e. 10× under target (478: 0.008 → 0.029); but hit 7-8 % (478: 10 %), kill 0.6-0.9 (478: 2-2.7 from 200 k), D 250 until 300 k then 300, 350 at 800 k (478: 450). Hc 0.20 → −0.07 at 600 k, same slope as 478 ⟹ continuous entropy fall is not due to the KL regime. Hd 0.53-0.58. CSV fire 0.45/s constant. Trough 450-650 k (kill 0.2-0.4) then recovery 650-750 k (kill 1.5-2, hit 9 %).
- 2026-11-06 probe (side 450, 4 ep., stoch) per checkpoint — p_fire(can_fire, step ≥ 10) by angle, empty reserve:
  - save_1 (108 k): 0.30 flat (quantiles 0.15-0.35, 0 % > 0.5), empty at 4.5 s.
  - save_3 (324 k): 0.45 flat (spearman −0.03), empty 2.9 s; turret < 5° 9.4 % (478 save_3: 11.2 %).
  - save_5/6 (540-648 k): dump 0.75 flat, empty 2.1 s, **turret broken** (mu_canon_x +0.65 = constant rotation, < 5° 3-5 % of time, hit/shell 2.5 %) = the 478 trough at 620-670 k.
  - **save_8 (864 k)**: p_fire 0.70 under 5°, 0.65 at 5-10°, 0.31 at 10-20°, **0.17 at 20-45°**, spearman −0.25, stable in each time slice; deterministic identical (0.70 / 0.19). Shots at < 10° = 32.6 % of shells for 14.8 % of armed decisions (478 save_8: 29.5 % vs 29.4 %, save_12: 27.9 vs 25.9 ⟹ no enrichment on 478). **First selectivity measured since 470.** Coupled to zoom: corr(p_fire, p_zoom) 0.905; in zoom mode (p_zoom > 0.3) p_fire 0.88-0.95 whatever the angle, outside zoom 0.31 (< 5°) vs 0.10 (20-45°); 61 % of decisions at < 5° are in zoom mode vs 9 % at 20-45°. Turret still rotating (mu +0.54, < 5° 5.8 %; 478 save_8: 8.5 %): strategy = sweep + « zoom + fire » when the enemy crosses the reticle. But reserve emptied for 99 % of tanks, median 3.5 s (478: 2.3 s), unarmed 78 % of time; hit/shell 5.8 % (= 478), hits/tank 1.12 (1.08), shots/tank 17.9.
- 2026-11-06 why: 10 Hz decisions × 10 shells: any p_fire floor off-axis > ~0.03 empties the reserve in < 30 s; a floor at 0.17 empties it in 3-6 s, selectivity ×4 changes nothing to ammo. The centred reward does push against off-axis firing (expected gain < b) but the gradient is weak and slow (appeared between 650 and 864 k). Epochs 2 kept skip at 0 but under-uses rollouts (kl 10× under target): slower learning than 478, not more stable for the turret (broken 540-650 k like 478). Read selectivity via the share of shots < 10° vs share of decisions < 10° (enrichment), not spearman alone; entering « zoom+fire » mode is the sign the CNN detects the centred enemy. Scripts: scratchpad session 60e24df7 (probe480/, analyze_probe.py).

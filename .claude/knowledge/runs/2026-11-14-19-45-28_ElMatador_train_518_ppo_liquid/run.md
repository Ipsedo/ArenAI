# train_518_ppo_liquid

| | |
|---|---|
| machine | ElMatador |
| user | samuel |
| launch | 2026-11-14 19:45:28 |
| commit at launch | `e554371e` — update readme |
| commits ≤ 90 min after | none |
| diff at launch | unknown (identity recorded after the run ended) |
| raw data | `outputs/train_518_ppo_liquid` on ElMatador |

## Established facts

- 2026-11-14: Samuel's choice for 518: resolution alone = 512×256, 515's CNN + tail block 128→128 (GN 2/4/8/16/16, output 384), LTC 128, sensors 64, zoom still removed; defaults changed for ppo/ppo_liquid/ppg, release rebuilt 18:46, not committed. At 18:48, on Samuel's suggestion the 128→128 block was removed: 518 = exact 515 CNN (4 blocks, GN 2/4/8/16, output 384) on 512×256 ⟹ 16×32 map before spatial softmax (cell ≈ 2.5°), receptive field 31 px ≈ 4.8°; LTC 128, sensors 64, zoom removed. Only architecture change vs 515: resolution. Release rebuilt, agent tests green, not committed.
- 2026-11-14 20:45: 512×256 track ended: throughput measured at saves (save_0 → save_1 = 27 k): 515 19 min (~85 k/h), 516 45 min, 517 57 min, 518 > 42 min (~27 k/h) ⟹ 512×256 = 2.4-3× slower (5-min trials underestimated: first rollouts only). 518 stopped at save_0. Samuel: zoom restored (state d1bb4020: 2 discrete actions, initial_zoom_proba 0.5, probe p_zoom + `--zoom on|off`), vision 128×256, CNN 3 blocks 3→32, 32→64, 64→128 (GN 4/8/16, map 16×32, output 384) for ppo/ppo_liquid/ppg; LTC 128, sensors 64. Release rebuilt, tests green, not committed. Game models reload again.

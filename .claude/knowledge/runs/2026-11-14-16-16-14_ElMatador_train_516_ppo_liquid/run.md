# train_516_ppo_liquid

| | |
|---|---|
| machine | ElMatador |
| user | samuel |
| launch | 2026-11-14 16:16:14 |
| commit at launch | `e554371e` — update readme |
| commits ≤ 90 min after | none |
| diff at launch | unknown (identity recorded after the run ended) |
| raw data | `outputs/train_516_ppo_liquid` on ElMatador |

## Established facts

- 2026-11-14: Config of 516 decided by Samuel after the "forced zoom: resolution is not the bottleneck" finding (zoom on does not make shots hit): main.cpp defaults vision 256×512 for all agents (sac keeps its 4 blocks ⟹ 16×32 map); ppo/ppo_liquid/ppg vision_channels 3→8, 8→16, 16→32, 32→64, 64→128, group_norm_nums 1,2,4,8,16, final map 8×16, output 384 unchanged; zoom removed for the agent only (ENEMY_NB_DISCRETE_ACTION = 1, `to_actions` sets zoom_button false, `initial_zoom_proba` removed from the 4 agents (CLI + factory), metric alpha_zoom gone; the player keeps zoom); probe: `--zoom` and column p_zoom removed, probe_metrics.py returns NaN without p_zoom. The 3 models in resources/trained_models (2 discrete heads, 128×256) no longer load (Samuel's choice, replace after retraining); checkpoints ≤ 515 not probeable with the then-current binary.
- 2026-11-14: 512×256 trial (before zoom removal, head block in liquid only): throughput −9 % vs 515, GPU peak 12.1/16 GB. 515 stopped at 380 k (save_14). Nothing committed; gamma 0.998 still an uncommitted default. 516 differs from 515 by resolution + block + zoom removed; expected gain = range 100-200 m.
- 2026-11-14 17:00 (revision after 30 k of 516, ev distorted by micro-batch 256): block moved to the tail, 3→16 … 128→256, GN 2/4/8/16/32 (ppo, ppo_liquid, ppg); ppo_liquid neuron_number 256, hidden_size_sensors 128. 5-min trial (gpu batch 256): GPU peak 11.1 GB, ≈ 17.6 idx/s vs 25.5 (first 516 trial) and ≈ 28 (515) ⟹ throughput −37 % vs 515 (only 3 rollouts). 516 stopped by Samuel. CSV ev is computed per GPU micro-batch: not comparable across configs with different gpu_batch_size. At 8.2 % hit at 25-37 k (see 517), 516 had hit 8.2 %, kill 16.

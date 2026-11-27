# PPO liquid train 452 — 32 tanks throughput (2026-10-29)

Host: ElMatador (memory notes); raw data no longer on disk.
Source note date: 2026-10-29. Measured with Release binary of 13:57, 150 s per config, GPU RTX 4070 Ti Super 16 GB.

| config | collection | train / rollout (300 steps) | throughput | tank-steps/h |
|---|---|---|---|---|
| 16 tanks | 135 steps/s | ~10 s | ~86 k steps/h | 1.37 M |
| 32 tanks (batch 512 or 1024, gpu 512) | 67 steps/s | ~20 s | ~44 k steps/h | 1.42 M |

- Real run 452 (32 tanks): 31.7 k steps/h over 16:54→17:09, vs 54 k for 451 (16 tanks). Only the expected ×2: each step renders 32 visions instead of 16 and each rollout trains on 9 600 rows instead of 4 800. Training phase ~80 % of time (GPU 80-100 %), VRAM 11 GB stable (peak 13 GB): no memory thrash, gpu_batch 512→256 not needed for memory. Episode reset recreates the 32 Vulkan renderers (6.5 s at 32 tanks, 2.4 s at 16), constant share ~13 %.
- Pitfall: `train_452` was created 11:46 but relaunched 16:54 in the same folder (config.json, save_0/*.pt rewritten, metrics.csv truncated at start). Date a run with `stat` of config.json, never the folder or save_0 mtime — otherwise a false throughput is computed (1 500 steps/h here).
- CSV `index` counts env steps, not samples: at 32 tanks curves look 2× slower per index while experience collected per hour is identical. Compare 16 vs 32 tanks in tank-steps (index × nb_tanks) or wall time.

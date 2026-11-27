# Probe 511 save_1 (27 k, D 300) vs 510 save_1

Run 511 = 510 + removal of respawn/truncated at empty reserve (commit deca02d4), identical config. `build/` binary of 13/11 09:13, commit deca02d4.
- 511: no inference diff.
- 510 save_1: launched in `--force`, only diff = respawn removed from `train_environment.cpp`. Both checkpoints thus probed under the same rule (no respawn): two policies compared in the same environment.

4 episodes per side, stochastic, 32 tanks.

| run | side | p_arm | ratio | hit/sh | h<5° | shots<5° | empty_s p50 | %empty | %disarmed | shots/s | turret<5° | σ cx/cy | p_zoom |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 510 s1 | 550 | 0.76 | 1.02 | 12.8 % | 21.2 % | 33 % | 2.3 | 97 % | 94 % | 0.077 | 36.1 % | 0.16/0.15 | 0.51 |
| **511 s1** | 550 | **0.66** | 1.01 | 14.1 % | 28.2 % | 30 % | **2.5** (p90 5.8) | 97 % | **86 %** | 0.089 | 33.0 % | 0.09/0.10 | 0.71 |
| 510 s1 | 1000 | 0.76 | 1.01 | 7.6 % | 13.6 % | 28 % | 2.3 | 99 % | 94 % | 0.069 | 24.5 % | 0.16/0.15 | 0.51 |
| **511 s1** | 1000 | **0.65** | 1.00 | 7.4 % | 16.1 % | 25 % | **2.5** (p90 3.0) | 99 % | 93 % | 0.071 | 27.2 % | 0.09/0.10 | 0.70 |

511 aligned & <40 m counts: 127 (side 550) and 46 (side 1000). Ratio therefore non-significant, but flat anyway here.

## Reading

- **Reserve**: no notable lengthening. 1st reserve empty goes from 2.3 to 2.5 s (p50), 97-99 % of tank-episodes empty in both cases. Only the tail moves at 550 (p90 5.8 s), and disarmed-step share falls from 94 to 86 %.
- **Trigger**: flat in both runs (aligned/off-axis gap < 0.01). 511 just has a lower level (p_arm 0.65 vs 0.76), enough to explain the +0.2 s. Deterministic: fires everywhere (p > 0.5).
- **511 CSV**: `s` ≈ 5-9 s (510: 40-70 s) and `fire` = 0. Artefacts of the new rule: `s` averages only deaths, which all happen in the first seconds of fire, then everybody is disarmed until `max_episode_steps` (probed median life 180 s). `fire` window falls in this period. Kills/episode 5 vs 15 at same index, for the same reason.
- **Canon**: σ 0.09/0.10 vs 0.16/0.15, with CSV `skip_canon` rising (0.31 at 27 k, 0.39 at 36 k; 510: 0.09). To watch: resembles 509 save_3's σ trough.

## Established / noise

- Established: at 27 k, the new rule has not yet produced shell economy. Dump in 2.5 s as in 510, flat trigger.
- Possible onset (one seed, 4 episodes): p_arm −0.1 and disarmed −8 points at 550.
- Open point confirmed: about 90 % of episode steps are spent dry and eventless, losing useful sample throughput.

## Next

- 511 save_3 (81 k) and save_5 (135 k) vs 510 save_3/5 (reserve 2.3-2.5 then 3.4/3.0 s): does the reserve go beyond 510's?
- Follow `skip_canon` and σ canon in CSV.

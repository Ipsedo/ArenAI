# Probe A on 433 save_17 (CNN 4 blocks + spatial softmax) — 2026-11-24

Binary: worktree `arenai_wt_spatial_softmax` (`7b612934` + fields `canon_yaw`/`canon_pitch` of `EnemySighting` + `--oracle_target network`). 3 episodes per side, `--oracle --oracle_fov 79 --oracle_policy_fire --oracle_target network`. Policy and geometric oracles: probes of the other PC (same save, binary without the option). Reading: `lecture_s17.py`.

| hits/tank-s (550 / 1000) | 433 s17 (D 800) | 552 s41 (D 1000) |
|---|---|---|
| policy | 0.144 / 0.031 | 0.275 / 0.119 |
| geometric aim, policy trigger | 0.128 / 0.049 | 0.527 / 0.365 |
| target chosen by the network map | 0.264 / 0.091 | 0.605 / 0.318 |
| ratio network / policy | ×1.8 / ×2.9 | ×2.2 / ×2.7 |
| shells/tank-s under network target | 0.80 / 0.26 | 1.88 / 0.99 |

Agreement with « nearest in view »: 53-83 % of designations.

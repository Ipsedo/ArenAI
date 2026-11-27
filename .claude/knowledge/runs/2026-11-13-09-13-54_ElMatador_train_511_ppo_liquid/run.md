# train_511_ppo_liquid

| | |
|---|---|
| machine | ElMatador |
| user | samuel |
| launch | 2026-11-13 09:13:54 |
| commit at launch | `deca02d4` — no respawn at empty shells reserve |
| commits ≤ 90 min after | `3fa18b07`, `c43516da` |
| diff at launch | unknown (identity recorded after the run ended) |
| raw data | `outputs/train_511_ppo_liquid` on ElMatador |

## Established facts

- 2026-11-13: Run 511 = 510 under the game rule at empty reserve: no respawn / no truncation at empty (see memory note on empty-reserve game rule without respawn). Probe save_1 (27 k, 511 vs 510 save_1, same binary; 510 save_1 probed under the same rule in `--force`: 2.3 s): no net lengthening of reserve (p50 2.5 s, 97-99 % empty). Disarmed 86 % vs 94 % at 550. σ canon 0.09/0.10 vs 0.16/0.15, and CSV `skip_canon` 0.31 → 0.39 (510: 0.09). In 511's CSV, `s` (mean death duration) and `fire` (end-of-episode window) are blind: deaths come early, then ~170 s of disarmed zombies — do not read them. Watch σ canon and skip_canon (509-type trough). Analysis `outputs/train_511_ppo_liquid/probes/save_1.md`.
- 2026-11-13: save_3 (81 k, D 400), 510 save_3 re-probed under same rule: reserve empty p50 2.4/2.3 s vs 2.3/2.2 (no gain); ratio 1.13/1.22 vs 1.05/1.03 (onset, p_arm 0.72-0.75); h<5° 36/20 % vs 24/10 %, hit/shell 21 % vs 15 % at 550; σ canon 0.29-0.36 (save_1 trough resolved). Old 510 save_3 probe archived in `probes/binaire_1e171bbf_avec_respawn/`. Analysis `probes/save_3.md`.
- 2026-11-13: save_5 (135 k, D 450), 510 save_5 re-probed under same rule: 511 empties faster (p50 2.3/2.2 s vs 3.1/3.0), trigger stayed flat and rose (p_arm 0.80-0.82, ratio 1.06/1.09) while 510 takes off (ratio 1.58/1.52, p_arm 0.47-0.58); precision caught up (h<5° 34/29 vs 37/24). Hypothesis: everyone dry in ~3 s ⟹ disarmed without danger (life 180 s), dump = collective equilibrium. Judge at save_9 (243 k); fallback = delayed respawn (run 512). Analysis `probes/save_5.md`.

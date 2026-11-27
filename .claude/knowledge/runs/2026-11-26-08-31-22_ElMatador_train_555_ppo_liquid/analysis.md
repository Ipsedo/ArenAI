# Run 555 — ppo_liquid, panorama aggregated by the mean

Launched 2026-11-26 08:31, running; read at 19:10, index 502 k, 47.6 k steps/h (1.52 M tank-steps/h). Effective commit `a3e0b27d` (committed 2 min before launch, training binary of 08:31; high confidence).

## What the run tests

Config ≡ 554. Code: one line, `arenai_agent/src/networks/panorama.cpp`, panorama map aggregation `amax(2)` → `mean(2)` (other commits docs / harness / desktop). Reference: 554, one seed each side.

## CSV (training rules)

D milestones: 300 @ 48 k, 400 @ 114 k, 500 @ 181 k, 600 @ 233 k, 700 @ 281 k, 800 @ 328 k, 850 @ 366 k, then blocked at 850 to 502 k. 554: 600 @ 208 k, 800 @ 305 k, 850 @ 328 k, 950 @ 404 k.

| slice (555, D 850) | kill | hit | fire | miss (m) | s (s) |
|---|---|---|---|---|---|
| 375-390 k | 17.6 | 0.24 | 0.39 | 48 | 52 |
| 420-435 k (trough) | 14.4 | 0.23 | 0.33 | 39 | 71 |
| 480-495 k | 18.5 | 0.31 | 0.59 | 43 | 50 |
| 554 at D 850 (328-365 k) | 21.0 | 0.18 | 0.61 | 69 | 46 |
| 554 at D 950 (480-495 k) | 17.8 | 0.27 | 0.41 | 59 | 57 |

- Trough at constant D from 390 to 435 k (kill 17.6 → 14.4, life 52 → 71 s), then recovery to 18.5 with hit 0.31 (run record; 554 max 0.29 at D 950).
- Profile different from 554 at D 850: hit higher, miss shorter (43 m vs 69 m), kill lower.
- Optimisation: kl 0.010-0.011, skip < 1 %, ev 0.73-0.78, α at floor 1e-4 since ~300 k, Hc −0.2 to −0.3.

## Game-rules matches

save_15 (405 k, in the trough): under 554 s15 (touches/tank 1.53 / 3.95 vs 4.14 / 6.62). save_17 (459 k): ≡ 554 s17 in combat (5.39 / 9.45 vs 5.07 / 7.62), stays armed longer (time dry 0.27 / 0.47 vs 0.43 / 0.60), 552 s41 level. Details matches/save_17.md.

## Panorama: encoder and trunk

Enemy AUC per cell on the labelled bench (0-60 / 60-100 / 100-200 m): actor s8 0.71 / 0.62 / 0.57, s17 0.73 / 0.65 / 0.58, untrained 0.63 / 0.59 / 0.54; critic s8 0.77 / 0.61 / 0.55, s17 0.89 / 0.75 / 0.58 (554 s17 max: 0.80 / 0.67 / 0.58); supervised 0.95 / 0.96 / 0.90.
Trunk h at save_17 (probe side 1000, 4 episodes, linear probe, test = episode 3), left/right of nearest out-of-field target: 0.54 (40-90°, < 100 m), 0.48 (> 90°, < 100 m), 0.45-0.51 at 100-200 m — chance, ≡ 554 s17 (0.58 / 0.51). `features/save_17_decode.txt`.

## Diagnosis

- Established: the 390-435 k trough is transient, in CSV and in game (s15 → s17).
- In the noise: D delay vs 554 (one seed; 550 reaches D 950 at 680 k); advantage at side 500.
- Established: at save_17 out-of-field bearing is not in the trunk and the actor's panorama encoder has not progressed since save_8; the critic's has (enemies < 100 m).
- Not decided: cause of the trough; why D stays at 850 with a mean kill of 18.5 (the criterion bears only on episodes played at D, kills/tank > 0.5).
Options/markers: decoding out-of-field bearing in the trunk on save_17 (`decode_panorama.py`, probe side 1000, binary `a3e0b27d`): says whether the mean makes the panorama enter the policy (~10 min); encoder on save_17 (`features/decode_encoder.py`, ~10 s): follow-up of the save_8 measure; passage of D 900; match at D 1000 entry vs 554 at same D; approach (cos velocity/target, armed time within 100 m) vs 554 s17 (0.19).

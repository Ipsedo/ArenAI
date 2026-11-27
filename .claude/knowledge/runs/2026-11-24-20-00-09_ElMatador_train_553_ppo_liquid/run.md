# train_553_ppo_liquid

| | |
|---|---|
| machine | ElMatador |
| user | samuel |
| launch | 2026-11-24 20:00:09 |
| commit at launch | `95f1812c` — spatial aim action (cell + offset) for ppo, ppo_liquid and ppg |
| commits ≤ 90 min after | none |
| diff at launch | unknown (identity recorded after the run ended) |
| raw data | `outputs/train_553_ppo_liquid` on ElMatador |

## Established facts

- 2026-11-24: run 553 = 552 + spatial aim action (`aim_cell` on an 8×16 grid, Gaussian-tanh offset, duals α cell/dx/dy; spatial aim code of run 552's note). Launched 2026-11-24 20:00 (commit `95f1812c`). D 1000 reached at 509 k. 552 → 553 is clean: the aim action is the only change (vs 510 confounded: reserve max 20 vs 30, different credit and reward, modified environment between `1e171bbf` and `95f1812c`).
- 2026-11-25 **Game verdict (Samuel, 25/11, save_36): bad.** The reserve empties very fast and the canon then stays pointing down. Probe s36 (972 k): 1st empty reserve median 5.4/5.8 s stochastic and 5.9/6.2 s deterministic (sides 550/1000); p10 2.2-2.4 s; 62-80 % of lives empty the reserve. 552 s41: 18/31 s, 37-51 %. Canon pointing down once dry is not measured: the training and probe environment respawns at empty reserve. "From the first tenths of a second" is not reproduced by the probe (p10 ≈ 2.2 s). Why: in training a dry tank is respawned full (`train_environment.cpp:127-131`) with a 0.25 penalty only; the game does not. Kill, hit and tour<5° measured in these conditions say nothing about the game. Apply: do not cite 553 as an improvement. Not established: the explanation of α_aim_dy (probe s36: $\bar H_x<-1$ while α_dx falls, open inconsistency).
- 2026-11-25 (game-rules match, s36, matches/save_36.md): dry at 10 s 0.30/0.48 of living (552 s41: 0.06/0.08), 5 matches of 8 ended « all dry » on side 500; touches/tank 5.1/8.3 ≡ or under 552 (4.8/9.3); canon < 5° from a target 0.33 vs 0.12-0.18 but no touch gain. Probe re-read on first life: touches/tank 9.4/5.5 vs 26.0/21.9 over all lives ⟹ more than half of the training touches came from refills when dry. Dry-canon-down not reproduced (−1°). Details: see etalons-verdict-jeu (game_labels.md). **Spatial aim and spatial softmax removed from code 2026-11-25** (Samuel's decision): `develop` back to 550's code. No 553 match binary kept (`build/` rebuilt): redo in a worktree at `89153e36` to replay its checkpoints.

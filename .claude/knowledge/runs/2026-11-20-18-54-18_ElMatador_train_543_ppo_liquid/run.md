# train_543_ppo_liquid

| | |
|---|---|
| machine | ElMatador |
| user | samuel |
| launch | 2026-11-20 18:54:18 |
| commit at launch | `cc4ee96b` — add new graphic parameters to GUI |
| commits ≤ 90 min after | none |
| diff at launch | unknown (identity recorded after the run ended) |
| raw data | `outputs/train_543_ppo_liquid` on ElMatador |

## Established facts

- 2026-11-20: run 543 (ppo_liquid, launched 2026-11-20 18:54:17, uncommitted tree above b83da585; `cmake-build-release/arenai_agent/arenai_agent_train` dated 18:54; `train_environment.cpp` carries `EMPTY_RESERVE_PENALTY = 0.5f`: good confidence, not provable by git) = 541 + constant empty-reserve penalty 0.5 (before: D/1000, i.e. 0.25 at D 250 and 0.8 at D 800); `config.json` identical to 541 (`discrete_target_proba` 0.01); one gap. Mechanical consequence: penalty ×2 at D 250, ×1.1 at D 450, equal at D 500, then lower than 541 (×0.63 at D 800); the zone the change targets (trigger holding at D 750-950) not reached at first reading. Details in analysis.md and probes/save_8.md, probes/save_13.md.
- 2026-11-20 (pt at 176 k): trigger not extinct, dump not resorbed (see analysis.md). Probe s13 (351 k, D 700 in CSV at 374 k): 543 reproduces 541's closing (s14 to s16) one to three saves earlier; failure criterion set for save_16 (aligned close < 0.2, cadence < 0.15) already met at save_13; possible passing reopening like 541 s20-22 (which did not hold, reclosed at s24).

# train_489_ppo_liquid

| | |
|---|---|
| machine | ElMatador |
| user | samuel |
| launch | 2026-11-07 08:01:45 |
| commit at launch | `b5309399` — reward redesign : damage fractions, aim potential, fire cost, no starvation |
| commits ≤ 90 min after | none |
| diff at launch | unknown (identity recorded after the run ended) |
| raw data | `outputs/train_489_ppo_liquid` on ElMatador |

## Established facts


- 2026-11-07: launched 08:01: 488's code with `AIMED_FIRE_COST_SCALE` 0.03 instead of 0.05, p_init fire/zoom 0.5. (see docs/decisions.md: fire cost / miss cost removed 2026-11-07)
- 2026-11-07 CSV at 72 k: kill 2.1-3.0/episode (488 at same index: 1.2-3.3), hit 16-18 %, s flat 6-7 s, kl 0.01, skip 0, Hd 0.69 → 0.50 (but Hd = mean fire+zoom, unusable alone for p_fire). Per-shell balance (residual r − r_hit − r_rcv − r_aim): cost 0.025, hit gain 0.019-0.021 + kill 0.004 ⟹ **net ≈ 0** (−0.004 at 10 k, +0.001 at 43 k, −0.0025 at 70 k). 488 at equal index: net −0.02 then −0.006 at the peak.
- 2026-11-07 probe save_2 (54 k, D=250): p_fire 0.54 / 0.55 / 0.52 / 0.55 — **flat and at init**; reserve emptied 125/128 at 2.8 s median, 5.4 fire actions/s; 97 % of tank-steps at empty reserve; hit/shell 13.8 %; zoom 0.75-0.78 (rising from 0.5). Why: the fixed-point reasoning holds (p settles where gain = cost) but the dump at D=250/32 tanks already hits 14-17 % thanks to the turret (31 % aligned), i.e. exactly the equilibrium hit of C 0.03: the dump IS the fixed point. Window between dump (C ≤ 0.03) and extinction (C 0.05, 488 crashed from p 0.15) is very thin, and the low-p fixed point is unstable (KL regime once events thin out). The reward contrast exists (+0.03 aligned vs −0.02 off-axis per shell) but the head does not read it in 54 k: the level of p is not a question of C. Stop tuning C to look for selectivity; the metric that counts = p_fire aligned − p_fire off-axis at each save (probe), not kill. Proposed: let run to save_5 (135 k) to see whether a slightly negative net moves p, before any change.
- 2026-11-07: probes save_2 to save_6 (54-162 k): p_fire 0.45-0.55 flat vs angle at each save, aligned/off-axis gap null (−0.03..+0.01), fire head has not moved from init in 160 k; Hd of CSV ≈ zoom entropy. Table in `probes/saves_2-6.md`. Next lever proposed (to validate by Samuel): amplify ×3 the per-shell terms in TrainEnvironment (see run 490).

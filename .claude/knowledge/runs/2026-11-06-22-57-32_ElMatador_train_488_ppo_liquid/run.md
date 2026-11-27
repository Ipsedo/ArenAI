# train_488_ppo_liquid

| | |
|---|---|
| machine | ElMatador |
| user | samuel |
| launch | 2026-11-06 22:57:32 |
| commit at launch | `b5309399` — reward redesign : damage fractions, aim potential, fire cost, no starvation |
| commits ≤ 90 min after | none |
| diff at launch | unknown (identity recorded after the run ended) |
| raw data | `outputs/train_488_ppo_liquid` on ElMatador |

## Established facts


- 2026-11-07: launched 2026-11-06 22:57; same code as 487 (fire cost = 0.05 × (1 − φ) at decision), only difference `initial_fire_proba` and `initial_zoom_proba` 0.5 (edited by Samuel in liquid_ppo_hyperparams.h, uncommitted). (see docs/decisions.md: fire cost / miss cost removed 2026-11-07)
- 2026-11-07 trajectory: 0 → 110 k: kill 1 → 4.2/episode, hit 12 → 22 %, miss 17.7 → 15.2 m, Hd 0.65 → 0.39 (p_fire 0.5 → 0.13, monotone from start). 113 k: skip appears; 113 → 165 k: kill 4.2 → 0, Hd → 0.19, r_hit → 0 exactly at 188 k. Then corpse: Hc 0.05 → −0.54, skip 0.38, kl 0.03, Hd 0.06-0.10; D 250 → 300 → 250 = artefact of a « hit » window frozen without shots.
- 2026-11-07 probe save_4 (108 k, the peak): p_fire 0.153 / 0.151 / 0.150 / 0.135 for < 5° & < 40 m / < 5° / 5-20° / > 20° — **flat**, no selectivity; reserve emptied 111/128 at 11.7 s median; hit/shell 25.6 % (37 % aligned, 27 % at 5-20°, 11 % > 20°): the gain comes from the turret, not the trigger. Save_6 (162 k): p_fire 0.000 everywhere, 21 shots over 218 k tank-steps. Probes: scratchpad `probe_488_save{4,6}.csv`.
- 2026-11-07 per-shell balance at the peak (residual r − r_hit − r_rcv − r_aim): mean φ 0.20, cost 0.040, hit gain 0.029 (≈ 0.13 per hit × 22 %), kill ~0.006 ⟹ net −0.005. The mean shot never changed sign, p_init 0.5 only bought 20 episodes instead of 5 (487).
- 2026-11-07 why: 4th confirmation (465, 471, 479, 487, 488) that the level of p_fire follows the sign of the expected mean shot, the shape of the cost changes nothing while the head does not read alignment. But 488 shows a negative feedback: when p falls, hit/shell rises (12 → 22 %) ⟹ a fixed point exists where gain = cost if C is under the attainable ceiling. Equilibrium hit ≈ (0.8 C − 0.006) / 0.13: C 0.05 → 26 % (peak 22-26 %, razor's edge, missed), C 0.035 → 17 %, C 0.03 → 14 %, C 0.025 → 11 % (≈ random spam at D=250, back to dump).
- 2026-11-07 **Decision (Samuel):** C = 0.03 (`AIMED_FIRE_COST_SCALE` in train_environment.cpp, false comment removed, build/ recompiled, uncommitted) for run 489; recentring the cost (C × (φ̄ − φ)) leaves the hit gain alone in mean (> 0) ⟹ dump like 483; recalibrate C ≈ 0.03 aims at a p_fire stabilising where hit/shell ≈ 14-17 %. Gate: Hd ceasing to decrease (not monotone to 0), fire > 0.3/s and kill held at 200 k; probe save_4 and save_8.

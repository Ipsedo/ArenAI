# train_490_ppo_liquid

| | |
|---|---|
| machine | ElMatador |
| user | samuel |
| launch | 2026-11-07 10:45:28 |
| commit at launch | `b5309399` — reward redesign : damage fractions, aim potential, fire cost, no starvation |
| commits ≤ 90 min after | none |
| diff at launch | unknown (identity recorded after the run ended) |
| raw data | `outputs/train_490_ppo_liquid` on ElMatador |

## Established facts

- 2026-11-07: Change vs 489 (decided by Samuel after probes 489 save_2-6: fire head frozen at init, aligned/off-axis gap null; coded in `arenai_agent/src/core/train_environment.cpp`, not committed at coding time, `build/` + agent tests ok): amplification ×3 of per-shell terms: `DAMAGE_REWARD_SCALE = 3.f`, `AIMED_FIRE_COST_SCALE = 0.09f` (C 0.09). Bonus `(k − 1) × (hit + received)` added per decision with shaping and cost; CSV `r_hit` / `r_rcv` report the amplified values so `r − r_hit − r_rcv − r_aim` stays the fire cost. Kill/death unchanged. Game-model reward untouched (training shaping only).
- 2026-11-07: Hypothesis (SNR): at net ≈ 0 per shell the fire head moves neither in level nor in contrast although the reward contrast exists (+0.03 aligned vs −0.02 off-axis) and shots abound (450/episode); per-shell signal is drowned by advantage noise (kill/death ±1, critic ev 0.4-0.7). ×3 keeps net ≈ 0 (level unchanged), triples contrast (+0.09 / −0.06), learning ~k² faster in samples. Does NOT fix the dump mechanically (only via selectivity if it emerges); mean also ×3 more sensitive (net 489 +0.003 → +0.009); damage = 3 × kill in priorities. Gate save_4 (108 k): aligned/off-axis gap ≥ 0.07 at probe and reserve > 3 s; null gap ⟹ SNR hypothesis falls, bottleneck = the fire head itself.
- 2026-11-07: Verdict (gate failed). Probes save_2-5 (54-135 k): p_fire 0.47 / 0.485 / 0.45 / 0.41, flat vs angle at ±0.02 on all 4 saves, identical empty-reserve or armed ⟹ fire head independent of state; dump at 3.3-3.4 s, 125-127/128 reserves emptied. Turret progresses meanwhile (hit/shell 17 → 23 %, shots < 5° 15 → 22 %, aligned hit 40 %), zoom too. CSV: better than 489 up to 54 k (kill 4.1 vs 2.8) then same level (kill 2.3-3.5, D 250), net fire ≈ 0, Hc 0.21 → 0.06, kl 0.019, skip 1 % at 148 k. SNR hypothesis refuted: tripling contrast changed nothing.
- 2026-11-07: Measured cause: **the fire head receives only 2-3 % of decisions** (mask outside armed states) because the dump empties the reserve in 3 s of 180 s; ~200 armed decisions per rollout, ~30 aligned. Dump ⟹ no data ⟹ head at init ⟹ dump: closed loop, the reward never enters it. Led to run 491 (reserve-empty truncation).

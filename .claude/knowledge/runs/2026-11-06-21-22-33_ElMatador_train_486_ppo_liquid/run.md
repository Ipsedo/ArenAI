# train_486_ppo_liquid

| | |
|---|---|
| machine | ElMatador |
| user | samuel |
| launch | 2026-11-06 21:22:33 |
| commit at launch | `b5309399` — reward redesign : damage fractions, aim potential, fire cost, no starvation |
| commits ≤ 90 min after | none |
| diff at launch | unknown (identity recorded after the run ended) |
| raw data | `outputs/train_486_ppo_liquid` on ElMatador |

## Established facts


- 2026-11-06: launched 21:22, commit `b5309399`: reworked reward = damage as fraction of health (±), kill +1 / death −1, FIRE_COST 0.03 at fire, aim-potential shaping (scale 0.3, Cauchy 2.5 m + 3°), economy 10 / +2 per hit / max 20, no starvation (truncation at 180 s).
- 2026-11-06 first 11 episodes (20 k steps, save_0 only): CSV `fire` (window 256 steps = 25.6 s) is sampled at 90 s and 180 s of each 180 s episode: it reads 0.00 everywhere, the dump of the first 10 s never appears. Read the dump through the cumulative reward decomposition: residual = r − r_hit − r_rcv − r_aim ≈ −0.03 × shots − suicides (kills/deaths between tanks sum to zero).
- 2026-11-06 residual: 12.7 → 22 shells/tank in the 1st half of each episode, ≈ 0 in the 2nd (reserve empty for the 32 tanks from 90 s, +2 refill chain extinguished). The 10 shells leave at 1/s = initial_fire_proba 0.10 at 10 Hz (1st line: 4 tanks of 32 fire at decision 0).
- 2026-11-06 damage reported per shell: 0.008 (ep. 1) → 0.030 (ep. 10) = FIRE_COST reached within 10 episodes; cumulative hit/shot 6.6 % → 14 %, kill 0 → 1.2/episode, miss 19.8 → 15.2 m. The comment « ~0.02 at D=250 » underestimates the density at 32 tanks. With refund +2 and the kill share, the random shell is ≥ free: nothing pushes to retain the shot. Hd rises 0.33 → 0.42 (Bernoulli 0.10 → ~0.15-0.20 on fire and/or zoom), kl 0.004-0.010, skip 0: the trigger drifts up, not down.
- 2026-11-06 why: the uniform fire cost is selective only if EV(random shell) < cost; at D=250 with 32 tanks the density makes the random shot profitable in 10 episodes (see ammo-dump-optimum; run 479 for the other edge: cost > EV ⟹ extinction). Do not read CSV `fire` to judge the dump under a 180 s episode; probe save_1 (108 k, ~22:50 on 2026-11-06): p_fire vs angle at step ≥ 10, empty reserve at which second. (see docs/decisions.md: fire cost / miss cost removed 2026-11-07)

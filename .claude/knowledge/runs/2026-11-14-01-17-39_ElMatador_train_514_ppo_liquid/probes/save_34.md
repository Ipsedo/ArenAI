# Probe train_514_ppo_liquid / save_34 — stochastic vs deterministic

- Index ≈ 918 k, D = 1000. Same binary as previous saves (`--force`, a·|a| present). Sides 550 and 1000, 4 episodes, economy 10/3/30.
- Goal: know whether the dump measured in stochastic also exists in game, where the trigger fires when p > 0.5.

## Summary

| mode | side | ratio | hit/sh | h<5° | shots<5° | shots ≥ 20° | empty_s (p50 / p90) | %disarmed | shots/s | turret<5° |
|---|---|---|---|---|---|---|---|---|---|---|
| stoch | 550 | 6.6 | 20.0 % | 29.8 % | 53 % | 17.5 % | 9.8 / 32 | 12.7 % | 1.40 | 30.3 % |
| **determ.** | 550 | n/a | 22.1 % | 33.3 % | 55 % | 18.7 % | **13.8** / 37 | **6.0 %** | 1.25 | 28.8 % |
| stoch | 1000 | 12.3 | 18.8 % | 32.0 % | 48 % | 20.8 % | 15.2 / 44 | 10.6 % | 0.82 | 22.8 % |
| **determ.** | 1000 | n/a | 22.8 % | 32.6 % | 61 % | 16.2 % | **28.0** / 118 | **2.9 %** | 0.41 | 26.6 % |

In deterministic, the tank fires in 34-38 % of aligned states (<5°) at 550 and 10-21 % at 1000. It fires off-axis (≥ 20°) in 7.6-7.9 % of states at 550 and 2.2-3.7 % at 1000.

## Reading

- **In game, the dump is much smaller**: reserve lasts 1.4× (550) to 1.8× (1000) longer, disarmed-step share is divided by 2 to 4 (6 % / 3 %), and cadence by 2 at 1000. Part of the dump measured in training is thus Bernoulli sampling noise.
- **But off-axis leakage does not come from the draw alone**: in deterministic, 16-19 % of shells still leave at over 20° (1-2 % hits), as many as in stochastic (17-21 %). These shots are μ decisions (p > 0.5 in 2-8 % of off-axis states), not noise. Deterministic reduces volume, not the waste proportion.
- **save_34 vs save_20-23 (stochastic)**: turret reaches 30.3/22.8 %, 514's best point, at 510 save_28's level (28.8/22.7). Ratio, reserve and hit/shell stay within save_20-23's range. Reserve has not degraded despite CSV cadence rising.
- **New**: movement μ saturated (dir_x |μ| > 1.5 in 13.5 % of states at 550, dir_y 4.4 %), and zoom almost abandoned (p_zoom 0.04-0.11 vs 0.25-0.42). Trigger/zoom correlation falls from 0.86 to 0.33-0.40.

## Conclusion

- The reserve problem is half as big in game as in training, but not null: about one shell in six stays wasted off-axis by policy decision.
- To act on the policy-caused part, γ (horizon) and economy leads remain relevant. The draw-caused part vanishes by itself in game.
- To watch: movement saturation (dir_x 13.5 %), recalling 501's « μ at bounds ».

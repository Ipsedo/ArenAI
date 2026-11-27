# train_485_ppo_liquid

| | |
|---|---|
| machine | ElMatador |
| user | samuel |
| launch | 2026-11-06 18:57:27 |
| commit at launch | `d12273ef` — hit reward baseline on opportunities : EMA of predicted hit over armed decisions |
| commits ≤ 90 min after | none |
| diff at launch | unknown (identity recorded after the run ended) |
| raw data | `outputs/train_485_ppo_liquid` on ElMatador |

## Established facts


- 2026-11-06 19:10 **Decision (Samuel: « on part sur l'option 2 »)**, follow-up to run 484 (slow extinction): the hit is no longer paid at impact but at the fire decision, on the ray predicate (HIT_REWARD if the predicted trajectory stops on a living enemy body), centred by the same `hit_baseline` on opportunities; kill raw and refund stay on the real hit. Samuel dismissed the curriculum hypothesis (probe at fixed side 250, drop began at D=250, 481/482 extinguished at 250, and the curriculum advances on real hit that rises when firing less = a consequence) and chose option 2 over the diagnostic run at fixed D (`--final_spawn_width/height 250`, lever 1, never launched) and ×k amplification (lever 3, never tried).
- 2026-11-06 mechanism targeted: before, the negative part (aim under the bar) fell exactly on the fire decision, the positive part (real hit 0.3) arrived 1-2 s later, diluted: with γλ = 0.945 over ~15 decisions the fire decision recovered only ~45 %, the rest credited intermediate non-fires ⟹ a successful hit also taught « do not fire ». Also reward (real hit) and bar (predicted hit) did not measure the same thing. Now both parts are exact, measured on the same quantity as their bar ⟹ mean zero by construction for a blind policy; +0.37 max in an aligned state, −0.03 off-axis. Not corrected: if the policy does not perceive the aligned state, no net force ⟹ p_fire stays near init = dump. Raw kill stays delayed (leak toward non-fires, rare, left). Estimated chance 1/2; last clean lever of the « centring reference » line.
- 2026-11-06 code (uncommitted, 4 files on top of 484's 8, Samuel runs pre-commit + commit): `jolt_enemy_tank.h/.cpp`: `mutable float hit_to_pay`; in `tick`, at the aim_pending site: `prediction = predict_shot(tracked, tanks)`, `aim_to_pay += scale × aim`, `if (prediction.hit) hit_to_pay += hit_reward`; in `get_reward`: `detail.hit = hit_to_pay`, impact only pays `kill` (`if (has_hit && has_killed)`). `train_environment.cpp`: `rewards[i] -= (aim_mean + hit_mean) × nb_fired_shells` (no more nb_landed term). Tests: `HitSplitsIntoAimHitAndNoKill` (hit = HIT_REWARD at fire, 0 at impact, total = aim + hit of the shot), `MissedShellPaysOnlyAimTerm` (hit 0 at fire at 27° off-axis), `AimOpportunityMatchesTheShotItPredicts` (paid hit = hit_opportunity). Model 82, core 27, agent 2280. 4 min GPU check (484 still on GPU): r_hit 0.0009-0.0013 (= predicted hit at fire, ≈ 0.3 × fire 0.04/step × ~8 %), b 0.020-0.029, r_aim 0.0005, fire 0.3-0.5, real CSV hit 6-10 % (column unchanged, it drives the curriculum), kill 0.2 at 2.7 k, ev −2.6 transient, nothing abnormal.
- 2026-11-06 gate for save_1 (108 k, ~1 h 20): CSV fire ≥ 0.4/s until 100 k (484: 0.56, then 0.37 at 130 k); probe: p_fire facing (< 5°) > 0.3 and > 484 save_1 (0.16), off-axis stable or falling, no spawn reflex (p_fire 1st second ≈ after 10 s), reserve emptied < 72 %. If facing < 0.2 and flat ⟹ perception, stop stacking (auxiliary head refused). If p_fire ≈ init everywhere and empty reserve < 4 s ⟹ dump = policy blind to the signal.

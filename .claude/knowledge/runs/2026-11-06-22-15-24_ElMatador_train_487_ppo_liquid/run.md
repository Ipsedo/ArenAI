# train_487_ppo_liquid

| | |
|---|---|
| machine | ElMatador |
| user | samuel |
| launch | 2026-11-06 22:15:24 |
| commit at launch | `b5309399` — reward redesign : damage fractions, aim potential, fire cost, no starvation |
| commits ≤ 90 min after | none |
| diff at launch | unknown (identity recorded after the run ended) |
| raw data | `outputs/train_487_ppo_liquid` on ElMatador |

## Established facts


- 2026-11-06 evening **Decision (Samuel, after 486 analysis; option A, location 1)**: the uniform fire cost (FIRE_COST 0.03, model) replaced by a cost indexed on predicted aim, C × (1 − φ) with C = 0.05, computed in the training environment (`AimedFireCost`); the model only pays the game's objective. Cost of a shell = 0.05 × (1 − φ(s)) paid at the fire decision, φ = Cauchy aim potential (2.5 m + 3°) of the predicted trajectory, same as the potential shaping; φ(s) read in `PotentialShaping::potentials()` before the shaping `step`. Code: `arenai_agent/src/core/aimed_fire_cost.{h,cpp}`, wired in `train_environment.cpp` (constant AIMED_FIRE_COST_SCALE). FIRE_COST and `RewardDetail::fire` deleted from the model: model reward = damage ±, kill, death only. No CSV column for the cost (r = residual, see run 486). Samuel set save_every to 5 episodes (27 k steps) in main.cpp on 2026-11-06. (see docs/decisions.md: fire cost / miss cost removed 2026-11-07)
- 2026-11-06 calibration table (470 for P(hit), 486 for damage 0.15/hit): off-axis φ 0.05 net −0.045; 5-20° φ 0.15 net −0.03; aligned < 5° and < 40 m φ 0.7 net +0.06; predicted trajectory on target φ 1 cost 0 net +0.08. Why: the advantage of firing = EV_hit(s) − cost(s); with a fixed cost the trigger gradient is a scalar (dump or extinction: 465/471/479), with a cost in (1 − φ) its sign depends on the state. No EMA bar (481's ratchet came from the self-referential bar). Gate at save_1 of the next run: p_fire facing ≥ 3× p_fire off-axis at step ≥ 10, fire ≥ 0.2/s (no extinction), reserve non-empty at 5 s. Known risk: far shot at high φ free but unlikely (tolerance 3° × distance).
- 2026-11-06 22:50 (p_init 0.05, save_every 27 k) **balance: extinction, not dump.** Reward decomposition (valid to ~21 k steps; after, the r window of 691 k samples is full and the cumulative residual is worthless): episodes 1-5 = 13-15 shells/tank in the 1st half (init 0.05 at 10 Hz = 0.5/s, reserve empty at ~20 s), then 10 → 3 → 1 shells/tank at episodes 6-11; Hd 0.20 → 0.10 (p ≈ 0.02). Probe save_1 (27 k, 2 episodes, D=250): p_fire 0.000 in all alignment classes, including < 5° and < 40 m (n = 1729), 26 shots over 101 k tank-steps (0.003/s), 0/64 reserves emptied, 3 hits of 23 shots (2 of the 3 under 5°).
- 2026-11-06 mechanism: during the fall, mean φ ≈ 0.10-0.14 ⟹ mean cost 0.044 against mean gain 0.15 × 7 % ≈ 0.01: the mean shot is worth −0.03; aligned states are 1.7 % of armed decisions and the fire head cannot yet distinguish them (CNN from scratch) ⟹ the gradient follows the sign of the mean, p falls everywhere, then no more samples = ratchet at 0. The state-dependence of the cost is invisible while the policy does not read the state. 486 (uniform cost 0.03 ≈ EV) showed the opposite sign: p rose. The razor's edge is the mean, not the shape of the cost. Same family as runs 471, 479.

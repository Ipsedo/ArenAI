# train_498_ppo_liquid

| | |
|---|---|
| machine | ElMatador |
| user | samuel |
| launch | 2026-11-09 16:22:47 |
| commit at launch | `564f2656` — initial discrete proba 0.5 for ppo and ppg |
| commits ≤ 90 min after | none |
| diff at launch | unknown (identity recorded after the run ended) |
| raw data | `outputs/train_498_ppo_liquid` on ElMatador |

## Established facts

- 2026-11-09: Run 498 (09/11 16:22) = 497 with `initial_sigma` 0.15 (start on the constraint Hc −0.5). Read at 79 k, probe save_2 (54 k, side 250). **Regulator does its job**: Hc −0.44 ± 0.04 (min −0.50, max −0.35) all run, σ canon 0.15-0.17 at probe. Continuous α non-zero on 22-39 % of rows, mean 0.003-0.006, max 0.06: bang-bang at the frontier (integral ~0, proportional on minibatch noise). α fire/zoom ≡ 0 (Hd 0.6 ≫ 0.2). kl 0.009, skip 0, ev 0.25-0.37.
- 2026-11-09: **Worst agent since 439**: CSV hit 4.5 % (497: 6.8; 496: 8.0 at 30-40 k), kill 3.5 → 1.2 → 2.2, D 250 at 79 k. Probe: turret <5° 7.1 % (497: 10.6; 496: 14.4), median angle 29° (19 / 16), hit/shell 7.0 %; trigger flat (ratio 1.04), dump 3.2 s, 76 % disarmed.
- 2026-11-09: Why: 494-497 all learned the turret during free σ descent 0.30 → 0.20 (80 k); pinning σ at 0.15 from step 0 removes that exploration phase. Of the 4 runs: the two good ones (494: fixed 0.001; 496: log α 1e-3 → 1e-4) have a small constant bonus **and** free σ from 0.3; 497 had no bonus, 498 no free σ. PID not at fault; operating point « floor at working level » is. Entropy floor must sit below the working zone (σ 0.10-0.12 ⟹ Hc −0.9 / −0.7), where 494/496 collapsed, with σ_init 0.3; and a constant bonus 0.001 under the PID (α = max(α_min, PID)) to recover 494's regime. Do not propose a target at current level again.
- 2026-11-09: Decision by Samuel (evening): back to 496's code (log α + Adam, target −1, σ_init 0.3); PID and `initial_sigma` 0.15 in `git stash` (« PID Stooke alphas (runs 497-498) »), tree = HEAD `564f2656`, `arenai_agent_train` rebuilt. Mechanism of 496's waves (read per 100 k slices): log α falls ~7 log units / 100 k steps while Hc > target (windup, CSV shows 0 below 5e-7), then ~100 k to rise when σ collapses ⟹ Hc plunges to −1.4/−1.6 (σ ~0.07), skip 0.15-0.33; integrator cycle at bounded speed. σ is a state head clamped `LOG_SIGMA_MIN` −5 (σ 0.007) / MAX 2 (`constants.h`). Proposals made: clamp log α (anti-windup), σ floor in head, KL-adaptive actor lr instead of skip (led to run 499).

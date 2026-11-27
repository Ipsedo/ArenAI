# Why the agent sprays instead of aiming — reward decomposition and failed fire cost

Runs: 344, 345, 346

Observation by Samuel in game after `train_344`: « il ne tire même pas vers moi ». Not a bug: the agent correctly maximises the reward as written.

**Measured behaviour.** Continuous μ stuck at the bounds (Hc measured −0.456 at σ = 0.326, below the theoretical value at |μ| = 1, −0.395). The turret is under **velocity control** (`enemy_handler.cpp`, `ENEMY_TURRET_RADIAL_VELOCITY = π` rad/s): saturated μ ⇒ **turret spins continuously, one revolution per 2 s**. Trigger: Hd = 0.363 ⇒ requested fire probability ≈ 0.88, `fire` = 1.47/s, limited by shell reserve not by policy. Precision `hit/fire` ≈ 2.3 %. The ~6-7 kills come from 32 tanks spraying simultaneously.

**Reward decomposition** (per tank per second, run 344): aim Gaussian **+0.123/s (76 % of gross positive)**, hits + kills +0.038/s, death −0.036/s, hits received −0.003/s. The code comment said the Gaussian was « an order of magnitude under the hit bonus »: true per shell (0.5 vs 1.0), **false in expectation** — 0.087 vs 0.028 per shell, i.e. 3.1× above, because hits are too rare.

**Failure of the fire cost (`train_345`, `fire_cost = 0.2`).** Calibration error: the 0.115/shell given as « income of a random shot » was the income of the **trained** 344 policy. A fresh policy earns only **0.059/shell**. Cost = 3.4× income, shooting lost from step one, fastest path to better reward = « stop shooting » — learned in ~120 rollouts (fire probability 2.7 %, `hit/fire` 0.00 %). Absorbing state: fewer shots → less aim signal → degrading aim. `discrete_entropy_coefficient` 1e-2 did not hold the trigger.

**Basic tension: a constant fire cost cannot do both.** Bootstrap needs c ≤ ~0.06, anti-spam needs c > 0.115, and shot income *grows with training*. Would need annealing. The original comment in `jolt_enemy_tank.cpp` (shell reserve rather than per-shot penalty) was right, and 345 confirmed it.

**Not at fault** (checked line by line): GAE (`dones` mask bootstrap and recursion, normalisation masked by `valids`), the rollout buffer, γ = 0.997 / λ = 0.98, and credit delay — a shell travels ~8 m/frame, so resolution in 6 frames at 44 m, not 40.

**Retained fix (`train_346`)**: `ratio = 1.5 * miss / sqrt(distance)` instead of `10 * miss / distance`. Tolerance was proportional to range (15 m at 150 m, three tank widths), making long-range crowd fire the cheapest income. The sqrt version keeps 4.4 m at typical engagement distance (44 m — 500×500 arena, 32 tanks) and drops to 8 m at 150 m. Gain vs 344: unchanged at 44 m, ÷8.7 at 100 m, ÷58 at 150 m; a real hit still pays 0.90.

**Next lead if `hit/fire` stays flat**: the only aim signal is paid at shell death, never during. A dense immediate signal (canon angle / nearest enemy, paid every frame) would give the turret a direct gradient. Not validated by Samuel.

(see docs/decisions.md: dense shaping removed 2026-07, fire cost removed 2026-11-07 / refused 2026-11-09, shell reserve validated 2026-08-06)

See comparisons/2026-08-17_ppo-entropy-regulation-known-gaps.md for metric reading.

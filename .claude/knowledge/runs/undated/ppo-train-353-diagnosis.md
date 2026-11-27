# PPO train 353 — shell damage fix (2026-08-19)

Host: ElMatador (memory notes); raw data no longer on disk.
Source note date: 2026-08-19. Reading script then: `ppo_report.py`.

First run under fix `1764289b` (one `on_contact` per body pair instead of one per manifold point). Two effects, not one: damage no longer multiplied, **and** shell refill no longer multiplied (`nb_shells += 5` once per impact). `fire` drops 1.50/s → 1.10/s and is entirely set by the stock ceiling (0.667 regen + 5 × hit rate ≈ 0.98/s).

- Death threshold is not 10 hits but 5 on the same part: `JoltTank::is_dead()` is an `any_of` over 7 `life_items` (chassis 10 HP; turret / canon / 4 wheels 5 HP). Scattered fire never kills. Central diagnostic metric (absent from CSV): **hits/kill** (theoretical min 5), driven by **gaussian/shell**.
- Reward decomposition: logged `r` excludes deaths (`train_environment.cpp:129`), so `r*30 = gauss/s + hits/s + kills/s − 0.1 × hits_received/s`. Under fix + `miss_distance_scale` 1.5: gaussian 0.067/shell vs hits 0.061/shell (~1.1×; run 344: 3.1×). Best balance the project had.
- `hit/fire` does not measure learning: 2.0 % (349-351) → 5.7 %, but already 3.3 % at rollout 0 (targets no longer vaporise in one shell, arena stays populated = mechanical). Flat 0.055-0.062 over 3 200 rollouts, `corr(hits/shell, kill) = +0.02`. What moves is alignment finesse: `corr(gauss/shell, kill) = +0.75`.
- Discrete head inert: Hd collapses 0.58 (p = 0.30, `initial_fire_proba`) → 0.065 (p = 0.988) near rollout 2300, never recovers (neither during the trough kill 8.4 → 1.6, rollouts 2400-3000, nor the full recovery kill 1.6 → 7.4, rollouts 3000-3200). A mechanism built on it (stock caps throughput, trigger picks the frame, low Hd = less shot timing) is refuted. Do not restart from `discrete_entropy_coefficient` without new evidence.
- Everything is on the continuous head, with a loop: trough then recovery, σ 0.38 → 0.66, Hc −0.15 → +0.18, `skip` 0.66 → 0.26, `clip` 0.28 → 0.19, hits/kill 65 → 11. Mechanism: low σ ⇒ same step gives larger log-ratio ⇒ `kl` up ⇒ `skip` up ⇒ the entropy bonus (inside `actor_loss` after the skip `return`, `ppo_trainer.cpp:198-217`) applies only 35 % of the time ⇒ σ drops further. Self-sustaining both ways.
- `skip` alone is not a lever: `corr(skip, kill) = +0.08` in level, `+0.22` in first differences; a loop member.
- σ target: the two best phases (peak rollout 1500, recovery rollout 3200) are both at **σ ≈ 0.65**, never 1.1. σ ≈ 1.15 of 349/350 = 87 % of uniform on [−1, 1] with saturated μ = spraying, which won **because the bug made spraying optimal**. Any performance comparison with 349/350 is void.
- Reading pitfalls: `816ea671` introduced KL trim, so `kl`/`skip` of 349-350 (old tail-inflated estimator) are incomparable with 351+. `fire`/`hit` windows are 256 samples ≈ 8.5 s: one line means nothing, need ≥ 100 rollouts.
- Confound: three variables changed between 351 and 353 (fix, `miss_distance_scale` 6 → 1.5, `α_c` 1e-2 → 2e-3) ⇒ only intra-run evidence usable.
- Read order: σ → (`clip`, `skip`) → Hc against its theoretical curve, then gaussian/shell and hits/kill.
- Related topics: entropy regulation known gaps; agent does not aim.

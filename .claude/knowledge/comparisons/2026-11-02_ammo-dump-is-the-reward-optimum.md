# Ammo dump is the reward optimum (run 467)

Runs: 465, 467 (also 437, 474 cited)

2026-11-02, run 467 (10 shells, +10/hit, cost 0.01, empty −0.5, γ 0.99 at 10 Hz) — emptying the reserve in ~10 s is the computed optimum of the reward: shot = +0.06 expectation (aim 0.068 for a 26 m miss = shot bonus), −0.5 at empty cheaper than living disarmed (−0.3/hit, −1 killed), γ removes 40 % from a shell kept 5 s; no penalty on the trigger can ration while miss does not move.

Run 467 (`outputs/train_467_ppo_liquid`, launched 2026-11-02 10:26, 340 k decisions, D 250 → 550): fire 1.1-1.5/s, hit 7-11 %, miss 21 → 26.5 m (never falls), s 12-15 s, r_aim 0.0047 > r_hit 0.0033 per step, r_rcv ≈ −(r_aim + r_hit) (symmetric melee, zero sum).

**Shot arithmetic (jolt_enemy_tank.cpp, tol = 10 m + 6°·D).** At D=550 tol = 68 m: a 26 m miss pays 0.1·exp(−26/68) = 0.068, a perfect hit 0.1, a 50 m miss still 0.048. The aim reward is a shot bonus, not a precision bonus, and the first source of income. Expectation per shot (aim + hit)/fire ≈ 0.06 vs cost 0.01 ⟹ every shot is net positive.

**Why −0.5 at empty weighs nothing.** Last shell, γ 0.99 at 10 Hz: firing ≈ +0.06 − 0.91·0.5·γ^15 + 0.09·(refill 10 shells ≈ +0.5) ≈ −0.28; keeping = living disarmed ≈ hits received (−0.004/step) + killed (−1) before 30 s ≈ −0.5. Dying empty is the cheapest exit. Same logic as run 437 (−1: « flat penalty »). And γ: a shell kept 5 s loses 40 % of its value.

**Why a fire cost cannot do the job.** Cost is constant, gain depends on state (density, distance). 0.01 changes nothing (467); 0.03 cuts shots as soon as density drops and extinguishes the trigger (465). No intermediate value: structural.

**Conclusion.** Rationing = waiting for a better shot than the current one; only valuable if the policy recognises a good shot, i.e. if it aims. Constant miss ⟹ all shots equal ⟹ sooner is better. The dump is a symptom of comparisons/2026-08-17_agent-does-not-aim.md.

**Rule:** never propose a penalty/cost on the trigger nor a reserve overhaul to treat the dump. Decide first with an offline probe on a checkpoint (p_fire and turret action as a function of canon ↔ nearest enemy angle): flat ⟹ perception lock (comparisons/2026-09-27_candidate-architecture-locks.md); dependent ⟹ reward shape (shot bonus). See comparisons/2026-10-27_ammo-economy-break-even-threshold.md, run 465, long cooldown refused. (see docs/decisions.md 2026-10-25 long cooldown refused; 2026-11-07 fire cost removed; 2026-11-09 fire-rate constraint refused)

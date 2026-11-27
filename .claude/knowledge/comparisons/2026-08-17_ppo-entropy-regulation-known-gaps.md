# PPO entropy regulation — windup 355, canonical PID, signed α after 375, review 375-379, 383, λ ≥ 0 (Stooke), per-dimension targets

Runs: 336-343, 344, 355, 358, 359, 367, 368, 374-379, 383, 405, 407

Series 336-343: dual ascent on PPO entropy coefficients collapsed in every possible way (limit cycle, windup, α at floor). Closed 2026-08-17 by **constant coefficients**. SAC keeps its dual ascent, intact. Later sections: 2026-08-19/20 return to per-action dual ascent, PID Lagrangian, 2026-08-27 discrete gain, 2026-08-30 signed α, 2026-08-31 review 375-379, 2026-09-01 constant factor 383 and λ ≥ 0 with constant target, 2026-09-21 per-dimension targets (ppo_liquid) after run 407 camping. (see docs/decisions.md 2026-09-01 target-entropy annealing refused; 2026-11-08 entropy coefficient replaced by automatic α per action)

**What decided it (`train_343`, bonus fully removed).** σ collapsed 0.31 → 0.027, log-ratio exploded (`kl` 0.017 → 0.65, 32× the `1.5 × target_kl` threshold), early-stop bit on almost all minibatches and the actor was left with **1 gradient step per rollout instead of 112**. Kills 4.5 → 0.6. Irreversible lock: low σ → huge KL → early-stop → no step to leave low σ.

**The diagnosis of 342 was wrong on one key point**: the PPO surrogate does have a downward restoring force on σ, and it is strong. The bonus was not inflating σ — it was preventing its collapse. With constant α there is a real equilibrium (`∂H/∂log σ ≈ 1` pushes up, the surrogate pulls down), unlike dual ascent where α itself diverged.

**Measured calibration** (α_c sweeps by the dual ascent of 340/341): below ~5e-4 σ drifts down, above ~1.2e-3 it rises. **α_c = 1e-5 is indistinguishable from zero** — the « plateau » read on 342 was a descent over too few rollouts. Retained: `continuous_entropy_coefficient = 2e-3`, placing σ* ≈ 0.4-0.5.

**Result (`train_344`)**: first run of the series to clearly progress — peaks 6.07 → 7.36 kills, stable troughs ~4.2, best 100-rollout window 7.39 (no previous run above 5.5 at this stage). The run crossed the KL threshold around rollout 900 (`kl` 0.037) **and came back out**: σ rose, `clip` with it. That is the difference with 343.

**Metric reading pitfalls.**
- `clip` low means nothing alone: recorded **only after** a successful `optimizer->step()`. Low + `kl` low + σ large = diffuse policy, all fine (`corr(σ, kl) = −0.69`, a large σ mechanically crushes `clip` and `kl`). Low + `kl` high = skipped minibatches, frozen actor. The pair matters, never `clip` alone.
- `kl = inf` = underflow of `exp(log_ratio)`, never fixed (`SIGMA_MIN = 1e-4`). 219 rollouts of 2010 in 344. Freezes the actor on those rollouts.
- **`Hc` betrays μ saturation**: compare measured Hc to the theoretical truncated-normal value for the measured σ and |μ| = 0. An Hc below the |μ| = 1 curve means means are stuck at the bounds. See comparisons/2026-08-17_agent-does-not-aim.md.

**Missing metrics, proposed three times, never added**: `σ_min` (distinguishes uniform collapse from a single recalcitrant dimension) and `|μ|` **per dimension** (Hc is a mean over the 4 dimensions, so cannot say whether turret or chassis). (see CLAUDE.md: no training metric added without validation)

## Return of dual ascent, per-action (`5f41fc18`, run 355, 2026-08-19/20)

Constant coefficients were replaced by an α **per continuous action dimension**, as **unbounded** `AlphaParameters` — SAC uses `ClampedAlphaParameters`. Run 355 reproduced the windup identically: α_c starts at 1e-3 and decays **exponentially, one e-fold per ~190k steps**, because the error `H − H_target` stays positive for 2.8M steps (natural σ ≈ 0.5 vs `target_sigma = 0.1`). α_c falls below 5e-7 at 1.59M — 4000× under the constant 2e-3 that worked. Then σ falls to 0.057, `clip` 0.09 → 0.46, `kl` 0.006 → 0.17, `skip` up to 0.52, kills 11.8 → 0.58. The regulator's time constant was of the order of the whole run: it never regulated.

**The entropy target at current μ is correct** — do not « fix » it. `truncated_normal_entropy(mu.detach(), target_sigma)` cancels the truncation term on both sides, so the constraint does mean « σ ≈ target_sigma » and not « H ≈ constant ». With a fixed target at μ=0, a policy at σ = 0.1 but μ stuck at the bounds reads 0.69 nat missing and inflates σ for nothing. End-of-355 numerical check: σ = 0.099 and Hc = −1.57, vs `H(μ=±1, σ=0.1) = −1.577`. The regulator was exactly at its setpoint. **Comparing Hc to the μ=0 reference is the trap** (the reading that led to a wrong « moving target » diagnosis).

**SGD does not replace the clamp.** Windup is structural: dual ascent = pure integrator, `log_α += lr·(H−H_target)` in SGD as `≈ lr·sign(…)` in Adam. As long as the error has one sign, it drifts without bound both ways. On 355 the error was ≈ +1.4 nat, so SGD would even have fallen 40 % faster than Adam. What SGD fixes is something else: Adam's scale invariance, giving the same step for 0.001 nat as for 2 nats, so α never settles.

**Fixes applied 2026-08-20** (pre-commit green): `ClampedAlphaParameters(1e-3, 1e-5, 1e-1)` continuous **and** discrete; `alpha_learning_rate` 2e-4 → **1e-3** (set on the measured e-fold: ~38k steps instead of ~190k); entropy bonus that was **counted twice** outside skip (`actor_loss = -mean(bonus)` then `-= mean(surrogate + bonus)`) brought to once; `kl_exceeded` fallback = **dry skip**.

**Why dry skip.** The old fallback did a pure-entropy step on the skipped minibatch, moving π away from π_old and raising KL of the following minibatches of the same rollout — positive loop, `skip` up to 0.52. Without an actor step, π stays equal to π_old, next rollout's ratios restart at 1 and the guard relaxes by itself. **Alpha updates continue on a skipped minibatch**: their loss contains no ratio, and it is what makes σ rise.

**Per-dimension metrics still missing** (4th request). On 355, `α_c` being a mean of exponentials over dimensions, impossible to say which dimension pulled α down at end of run.

**Why:** Samuel changes one variable at a time between runs to keep the comparison clean. **Rule:** facing a collapse, read σ first, then the (`clip`, `kl`) pair, then `Hc` against its theoretical value. `v_μ` losing its signal is a consequence, never a cause.

## PID Lagrangian: the PID plugged into an integrator (`934b061c`/`93c271a1`, 2026-08-20)

First PID Lagrangian draft: `pid()` returned `u = k_p·e + k_i·I + k_d·Δe`, but the trainer made it a **loss** `-log_α · u` passed to an SGD → `log_α += lr·u`. `log_α` is thus a second integrator in series with the I term: `log_α = lr·(k_p·Σe + k_i·ΣΣe + k_d·e)`, two poles at z=1, never stable. Plus: `I` unprojected (could dive to −140 with k_i=2.5e-2 and ~14 calls/rollout) and the clamp `[1e-5, 1e-1]` lost in the move from `ClampedAlphaParameters` to `PidLagrangianAlphaParameters`. α → 0, same symptom as 355 but different cause.

**Fixed (Samuel's choice: canonical PID Stooke et al. 2020, per-minibatch cadence kept).** `PidLagrangianAlphaParameters` no longer inherits `AlphaParameters`: no parameter, no optimiser (both `*_alpha_optim` vanished from `PpoTrainer`). α **is** the controller output, and the controller runs **on log α** (Samuel's idea): `I ← clamp(I + k_i·e, log α_min, log α_max)`, `log α = k_p·e + I + k_d·max(0, H_prev − H)`, `I₀ = log α₀`. Two consequences justifying log space:
- **positivity is structural** — no clamp on the output, a gain acts multiplicatively on α whatever the decade;
- only **one bound** remains, that of the integral, which is exactly the anti-windup.

Derivative **on the measurement** not the error: the continuous target moves with μ and would give a parasitic whip. Bounds as `static constexpr` in the class: `MIN_ALPHA = 1e-8`, `MAX_ALPHA = 1`. Gains in the anonymous namespace of `ppo_trainer.cpp`: `k_p = 2e-1`, `k_i = 5e-3`, `k_d = 1`, α₀ = 1e-3. `alpha_learning_rate` (CLI) becomes **unused on the PPO side**.

**The linear trap, for the record**: without output clamp, as soon as the integral hit its floor only `k_p·e` remained, so α **negative** when H > target — the entropy bonus became a penalty. That symptom triggered the switch to log.

**Rise time from the floor is the price of anti-windup**: (log range)/(k_i·|e|). With floor 1e-8, rising to 1e-3 = 11.5 nats, ~4600 updates at |e| = 0.5, ≈ 330 rollouts (~300k steps). *Bounded* (the goal) but the same order as 355's calibrated e-fold. Two levers if too slow: raise the floor (`α_c = 1e-5` already indistinguishable from zero per 340/341) or raise `k_i`.

**The D term is a multiplicative offset, not an integrator**: recomputed every update, never accumulated, so equals `×exp(k_d·ΔH)` with ΔH the inter-minibatch noise. Clipped ≥ 0, it gives a slight constant positive bias on α (a few %), not a drift. If α_c rises for no reason, `ALPHA_K_D = 0` is the first try.

Covered by 6 tests in `tests_entropy.cpp` (`PidLagrangianAlphaParameterTest`), incl. `StaysPositiveAndFinite` and `RecoversFromSaturationWithoutWindup` locking both failure modes.

## Discrete gain not at the scale of its error (`train_368`, 2026-08-27)

The controller was not at fault — the integral clamp **is** the anti-windup, by design. What was wrong: `DISCRETE_ALPHA_K_I = 5e-4`, **10× lower** than continuous while the discrete head's error scale needs ~20× **higher**.

Structural reason: the discrete target is the binary entropy of `target_fire_proba = 0.02`, i.e. **0.098 nat**, and the whole `Hd` range is [0, ln2 = 0.693]. Errors are intrinsically O(0.03), vs O(0.5–1) continuous. The rise formula `Δlog / (k_i·|e|)` gave, with errors measured on 368: continuous 3 800 updates ≈ **61 k steps** OK, discrete 766 000 updates ≈ **12 M steps** NOT OK (factor 200). `k_p = 2e-1`, identical on both heads, is equally inert on the discrete side: at |e| = 0.03 it moves α by 0.6 %.

`RecoversFromSaturationWithoutWindup` did not cover this case — it locks `k_i = 5e-3` and `|e| = 0.5`, i.e. the **continuous** head's parameters.

**Fixes (Samuel's choice, pre-commit green)**: `DISCRETE_ALPHA_K_I` 5e-4 → **1e-2** and `MIN_ALPHA` 1e-8 → **1e-6** (shared; 340/341 already gave α = 1e-5 as indistinguishable from zero). Discrete rise reduced to ~23 000 updates ≈ **370 k steps**, in the targeted budget. New test `DiscreteGainRecoversAtItsOwnErrorScale`, failing on both assertions with the old gain.

**What it does not explain.** On 368 there are **two** breaks, not one. α_d hits the floor at 3.68 M but `Hd` falls below its target only at 5.65 M: between the two the floor is the *correct* output. Only the 2nd break (5.7 M → end: `Hd` 0.17→0.068, `h/k` 19→24, `kl` 0.015→0.024) is windup. The 1st (3.9 → 5.1 M: `ev` 0.48→0.378, `kill` 10.6→6.3, `hit` down, **aim intact** `h/k` 15.6–17.4) remains **unexplained** — the critic degrades before aim. Unexplored lead: env changed between 367 and 368 (hit +1 s/+3 s, kill +5 s/+15 s, timeout penalty `−1` → `1 − steps/max` tending to 0 at end of episode, winner bonus `+4` → `4·(kills + 0.05·hits)`), making hit farming competitive with kills. Coherent symptoms: `s_μ` +3 %, `hit` +9 %, `kill` −29 %, `h/k` +39 %. To compare: 367 rose monotonically to 11.0 kills at 5.2 M, no regression.

## Signed α as equality constraint, after the collapse of `train_375` (2026-08-30)

`train_375` (beta law): best start of the series (kill 4.8 from 180k, smoothed peak 9.4 at 1.17M), `Hc` follows the descending target in lockstep from 1.2M to 3.6M **without any help** — α_c at floor all run, the constraint `H ≥ target` never violated. At 3.65M concentration snaps back (`Hc` −0.39 → ~0 in 300k steps) and **nothing holds it**: the one-sided controller (α = exp(log_α) > 0, always a bonus) cannot push entropy down. Self-sustaining loop afterwards: fewer hits → starvation deaths (s → 30 s floor) → signal reduced to the timeout −1 → weak advantages → no pressure to re-concentrate. kill 6→1 in 1M steps, no recovery. Second defect: `Hd` dives to 0.20 vs target 0.41 at 4.35M with α_d ≈ 0 — integral at floor `log(1e-6) = −13.8` makes the controller inert for thousands of updates even under massive violation (`k_p·e ≈ 0.04` vs −13.8).

**Fix (Samuel's choice, pre-commit green, run 376 from scratch)**: PID back to **linear space with signed α** — equality constraint `H = target`. `I ← clamp(I + k_i·e, ±1)`, `α = clamp(k_p·e + I + k_d·(H_prev − H), ±1)` (`MAX_ALPHA_ABS = 1`). Bonus when H < target, **penalty** when H > target: descent locked both ways. The derivative becomes **symmetric** (no more `clamp_min`): it also brakes fast rises — the missing emergency brake at 3.65M. The integral saturates at the output bound itself, so rise from saturation takes (range)/(k_i·|e|) and the P term acts immediately.

**Not a regression to the pre-log bug of 08/2026**: back then, negative α was an *accident* (unclamped output + integral at −140); here output and integral are bounded at ±1 and the sign is the intended mechanism. The reasoning « positivity is structural in log » (§ canonical PID above) is the one that let the 375 collapse happen. Tests rewritten: `EntropyAboveTargetTurnsAlphaNegative`, `StaysBoundedOnBothSides` (replaces `StaysPositiveAndFinite`), `RecoversFromSaturationWithoutWindup` adapted (rise in ~360 updates instead of ~4600). Alpha modules are not serialised in saves (actor/critic/optims only), so no checkpoint-compat concern.

Finding that stays true across 374 and 375: even at kill peak, `miss` never moved from ~30 m — gains come from cadence/proximity, not precision. See comparisons/2026-08-17_agent-does-not-aim.md.

## Cross review 375-379: gamma 0.997 + high continuous entropy (2026-08-31)

Analysis of `train_379_ppo` (trunc-normal, gamma 0.99, λ 0.98, warmup 7.5M) at 1M steps: decline already begun — kill 6.3 → 3.3 and hit ÷2 over the last 110 k-steps, **with no optimisation-side signal** (clip 0.155 stable, kl 0.011 stable, Hc stuck to target, fire/miss constant). Same early-collapse profile as 375 and 378. Aim decline is silent in the update metrics.

**Two clean proofs, by controlled pairs:**
- `378_ppo` vs `378_ppo_trunc-normal-alpha-negative` (only gamma/λ differ): moving to gamma 0.99/λ 0.98 **triples the clip fraction from the 1st window** (0.045 → 0.125), collapses v (0.18 → 0.05), degrades ev (0.37 → 0.25) and kills. gamma 0.99 = horizon ~100 steps ≈ 3.3 s at 30 Hz — positioning credit leaves the horizon when a kill needs 5 hits.
- `378_trunc` vs `379` (only warmup differs, 3M vs 7.5M): long warmup helps a bit (5.2 vs 4.3 kills at 1M) but does not compensate gamma.

**The only run that progresses durably is `377` (beta uniform, gamma 0.997)**: kill 3.4 → 10.6, holds 9.5 at 2.8M steps — and its Hc stayed 0.55-0.59 all run *because the controller could not track the descending target*. Pattern across the 5 runs: each time continuous entropy is effectively pushed down, performance declines; the only spared run is the one where regulation failed.

**Trunc-normal entropy scale on [-1,1] (μ=0)**: H(σ=0.3) ≈ 0.21, H(σ=0.5) ≈ 0.57, H(σ=1) ≈ 0.68, ceiling ln 2 ≈ 0.693. The old init 0.2 corresponded to σ ≈ 0.3, far from 377's winning regime.

**Fixes (validated by Samuel, pre-commit green, 2026-08-31)** in `ppo_hyperparams.h`: gamma 0.99 → **0.997**, gae_lambda 0.98 → **0.99**, `continuous_target_entropy_init` 0.2 → **0.55**, `initial_sigma` 0.3 → **0.5** (policy starts on its target, no α re-inflation transient at start).

**`continuous_target_entropy_final` −1 → 0.2 (Samuel's choice, 2026-08-31), signed α kept.** −1 was the SAC heuristic (−dim(A), Haarnoja 2018) transplanted out of context — PPO literature uses a fixed small bonus (0.01 Atari, 0 MuJoCo), never a schedule toward determinism. −1 ⇔ σ ≈ 0.09 = death zone documented 3× (343: σ 0.025/kl 1.3 frozen; 355: σ 0.069/kills 0.5; 359: σ 0.166 still 12.5 kills but clip 0.27, at the edge of the lock). Operating zone of all good runs at peak: **σ 0.4–0.8** (Hc measured −0.1 to +0.3); the best run of all time, `train_358` (peak 14.4, end 12.8, never collapsed), ran at σ ≈ 0.77. With signed α, final 0.2 is a *rail*: σ pinned ≈ 0.3 at end of warmup. Known residual risk: around 4-5M the cosine imposes σ ≈ 0.33-0.35, slightly under the golden zone — if the run plateaus at this stage with Hc stuck on target and healthy kl/clip, **raise the target**, do not look elsewhere. Note: post-368 kills not strictly comparable to runs ≤ 367 (env reward change between 367 and 368).

## The under-calibrated « constant factor » backtrack (`train_383_ppo_constant-entropy`, 2026-09-01)

Commit `8378a100`: back to a fixed-coefficient entropy bonus (no α/PID at all), but **CosineAnnealing** init → final over `target_entropy_warmup_steps` (3M). Run 383: continuous 1e-3 → 1e-4, discrete 1e-3 → 1e-4 (discrete lowered 10× by the uncommitted diff of `ppo_hyperparams.h`).

**The run replays exactly the collapse predicted by the 340/341 calibration** (« below ~5e-4 σ drifts down; 1e-5 indistinguishable from zero; the constant that worked = 2e-3 »): even the 1e-3 init is below the retained 2e-3, and the 1e-4 final is 5× under the drift threshold. Clear chronology in `metrics.csv` (1 row ≈ 900 steps, annealing done at 3M ≈ row 3350):
- 0 → 2000: normal rise, kill peak 8.7 around 1500; Hc follows annealing 0.47 → 0.25.
- **~3500 (= end of annealing): Hc turns negative** (min −0.19 ⇔ σ ≈ 0.2), clip 0.13 → 0.19, kl 0.013 → 0.021. Kills begin to decline (8 → 5).
- 5300 → 6700: starvation loop identical to 375 — kill 4.5 → 0.2, `s` 38 → 30 (timeout 30 s without hit), miss 21 → 29, `v` 0.10 → 0.03, r turns negative. Hc rises slightly at the very end (bonus/surrogate equilibrium exists even at 1e-4) but the reward signal is already dead.
- Discrete side: Hd 0.59 → 0.10 from row 2500 then 0.01-0.03 at end, fire 1.15 → 0.88 — the 10× lower discrete factor made the fire head quasi-deterministic very early.

Lesson: the « constant » scheme validated 2026-08-17 was not « any fixed coefficient », it was **2e-3 precisely**. Annealing toward 1e-4 recreates the α→0 of dual ascent, open-loop this time.

## Back to true Lagrangian λ ≥ 0, constant target (2026-09-01, Samuel's request)

After `1651296c` (PID Lagrangian + **ConstantTargetEntropy** — « target entropy annealing is destructive »), Samuel switched the PID to **inequality constraint H ≥ target**: canonical projection Stooke et al. 2020, in linear space (no return to log and its floor inertia): `I ← clamp(I + k_i·e, 0, 1)`, `α = clamp(k_p·e + I + k_d·d, 0, 1)`. `MAX_ALPHA_ABS` → `MAX_ALPHA`. The post-375 signed α protected against re-inflation during a **descending target**; with a constant target that rail no longer exists, the surrogate pulls H down and the floor suffices. Tests rewritten: `SustainedOvershootParksAlphaAtZero`, `StaysWithinZeroAndMax`; output/integral saturate at 0 → the P term acts from the first update after the flip. Gains unchanged (k_p 2e-1, k_i 5e-3/1e-2, k_d 1, α₀ 1e-3).

Reading of hyperparams in floor regime: the target is no longer a setpoint but a **lower bound** — initialise the policy *inside* the feasible region (H_init ≥ target), else α transient at start. `continuous_target_entropy = 0.4` ⇔ σ_floor ≈ 0.4 = low edge of the golden zone; `initial_sigma = 0.5` (H ≈ 0.57, feasible); `initial_fire_proba = 0.4` (H ≈ 0.67, feasible; spraying bounded by the env rate-limit of `9e02790a`).

**`continuous_target_entropy = 0.4`: well placed, not « optimal » (never swept).** At μ = 0, 0.4 nat ⇔ σ ≈ 0.37 (low edge of golden zone), but the constant target is compared to the real H(μ, σ): with saturated μ (Hc of good runs at peak = −0.1 to +0.3), the effective floor pins σ toward ~0.5 and **α_c will be active at peak** — intended, it reproduces in closed loop the winning « 2e-3 constant » regime (375-379: pushing Hc down kills performance, the historic error has always been targets too low). Run reading grid: α_c ≈ 0 + σ in zone 0.4-0.8 = harmless floor; Hc stuck at 0.4 + α_c at stop + healthy kl/clip + suffering perf = only case where going down to 0.3 is discussable; σ under 0.4 despite active α_c = gains problem, not target.

## Per-dimension targets on the ppo_liquid side (2026-09-21, validated by Samuel after run 407)

The collapse of 407 (see run 407 note: acceleration unlearned, camping) played out **above the floor**: Beta target −0.88 ⇔ κ ≈ 100, the camping locked at κ 25-80 without ever violating the constraint. And raising the global target is forbidden by the canon: in healthy 405_zoom, canon_x legitimately rose to κ 72-97 (H ≈ −0.85). Hence: `continuous_target_entropy` becomes a **per-dimension vector** (default `{-0.2, -0.2, -0.88, -0.88}` = direction x/y at κ ≲ 25, canon x/y unchanged) and `discrete_target_entropy_factors` a per-action vector (default `{0.2, 0.2}`, behaviour unchanged — no discrete collapse observed on 407: fire H 0.68, zoom H 0.40 at save_30). α were already per dimension; only the floor height becomes differentiated. Size TORCH_CHECK in the LiquidPpoTrainer ctor; `std::vector<float>` added to the variant of `cli_fields.h` (JSON on CLI). The **ppo** module (feed-forward) keeps its scalars — deliberately not migrated. Beta scale (mode 0.5): H(κ=10) = +0.21, H(κ=25) = −0.21, H(κ=100) = −0.89.

**The discrete floor must stay low: `discrete_target_entropy_factor = 0.2`** (0.139 nat ⇔ p ≈ 0.026), not 0.5. Samuel's argument, validated: at 30 fps, even p = 0.01 fires within 10 s with ~95 % probability — the floor does not serve to guarantee fire events (they abound), only to prevent 383-type locking (Hd 0.01-0.03, dead gradient), and 0.139 suffices. Above all: the PID error is on the **mean** of Hd over the minibatch, and a correctly differentiated policy (p ≈ 0.9 target in front 30 % of the time, p ≈ 0.02 otherwise) has a mean Hd ≈ 0.17 nat — a floor at 0.347 would harass it permanently and only allow uniformly hesitant heads. 0.2 is also coherent with the historical scale `target_fire_proba = 0.02` (0.098 nat) on which `DISCRETE_ALPHA_K_I = 1e-2` was calibrated post-368.

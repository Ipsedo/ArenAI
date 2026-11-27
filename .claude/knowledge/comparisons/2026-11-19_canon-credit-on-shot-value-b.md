# Canon credit on the shot value H (option B) — one-step credit, linear head of the critic

Runs: 539, 540, 541 (credit canon on H head), 542

2026-11-20 state: one-step canon credit on H = linear head of the critic (`to_shot_value`, auxiliary BCE ×0.1), 3rd CNN removed, ppo/ppg/ppo_liquid; uncommitted, never launched beyond a 3-min trial (run 540 stopped at 147 k, not analysed).

Option B validated by Samuel 2026-11-19 (« Oui, code B ») after run 539 (up-to-date image) and coded the same evening, uncommitted, compiled in `build/` only. Choices validated: separate network on the image alone with no gradient to the policy, label = hit alone, canon_x/canon_y credited only on the one-step variation of this value.

- **Code (first version):** `networks/shot_value.{h,cpp}` (agent CNN + head 128), `agents/shot_value_learner.{h,cpp}` (FIFO of in-flight shells per tank, ring of 32 768 labelled shots, AdamW 3e-4, 16 steps of 256 per rollout, horizontal mirror), `ppo_objective` (PolicyLogProbs, GroupKlStop next to KlStop), `LiquidPpoTrainer` (aim_advantages = normalised (γ·H(s′) − H(s))·(1 − cut), H evaluated before learning the rollout's shots). `on_transition` receives `shot_outcomes` [nb_tanks, 3] (fired, landed, hits) from `TrainTankEnvironment::last_step_shot_outcomes()`; ppo, ppg, sac ignore it.
- **CSV:** `skip_canon` and `shot_value` (cross-entropy on each rollout's new shots, before learning), CSV only. Options `--shot_value_*`.
- **9-min trial** (next to 539): runs, kl 0.005-0.009, skip 0; `shot_value` 0.26 → 0.13-0.17 for ~4.5 % hits (base rate ≈ 0.18). Throughput overhead not measured.
- **Consequence:** `LiquidPpoTrainer::load` requires `shot_value.pt` ⟹ `arenai_agent_sigma_gradients` no longer loads earlier checkpoints (no backward-compat for old checkpoints). Probe `arenai_agent_probe` untouched.

**Why:** only separate canon credit held ~20 kills at D 1000 (aim-synthesis-2026-11-19-the-circle); Φ refused because it biases aim; with the up-to-date image (comparisons/2026-11-19_image-one-decision-late.md), one-step credit read on the image is 0.100 offline.

**Test plan:** judge the run on kills at fixed side 550 and at D 1000, hit/shell and dwell holding after save_8, vs 539 (probe kills 24 / 20 / 25.5 / 23 / 21 from s3 to s12, dwell 0.62 → 0.24 s). Known risk: H learns only where the agent fires; offline it credits nothing beyond 100 m.

**Contested 2026-11-20:** Samuel discovers H has its own CNN (3rd encoder, after actor and critic): « j'ai jamais dit oui à ça ». The « separate network » noted as validated was not validated in his eyes — only « Oui, code B » is found in the 2026-11-19 session. Lesson: have every new architecture piece (network, encoder) explicitly validated, not only the principle.

**Redone 2026-11-20 (validated point by point by Samuel, then interactive code review):** `ShotValue`/`ShotValueLearner`/options `--shot_value_*` deleted. H = `to_shot_value` (Linear only, init like `to_value`) at the output of LTC (`LiquidCritic`) or of `head` (`ValueFunction`), trained in `backward_critic`: MSE + `SHOT_VALUE_COEF` 0.1 × mean BCE of the batch's labelled shots (constant, no CLI). Canon credited on normalised γ·σ(H(s′)) − σ(H(s)) (`shot_value_advantages`), rest GAE, `GroupKlStop`, CSV `skip_canon` + `shot_value` (in-sample BCE per critic minibatch, not comparable to 540). `agents/shot_credit.{h,cpp}`: `ShotOutcomes` (struct fired/landed/hits), `ShotLabeller` (lives in the 3 rollout buffers, delivering `rollout.shot_labels`), BCE, credit. Trainers: `evaluate_critic` / `compute_gae` / `compute_aim_advantages` separated. `KlStop` and the one-advantage surrogate removed. Samuel's review choices: no common class for the loss (independent trainers by philosophy), a single shot_credit file, critics' `value()` not renamed, critic tests only. (see docs/decisions.md 2026-11-22: separate aim credit removed)

# Presence head on the CNNs: 8×16 enemy-presence map (coded 2026-11-23)

Runs: 550, 551, 552 (first run), 536 (cited)

2026-11-23 — auxiliary head « enemy presence map » coded on the actor and critic CNNs of the 3 trainers, validated piece by piece, uncommitted. (outcome: removed 2026-11-25 after game matches vs twin 550, see docs/decisions.md; superseded by the panorama mask loss 2026-11-27)

Coded 2026-11-23 after the bimodal-engagement analysis (comparisons/2026-11-23_turret-bimodal-per-pass-engagement.md: the CNN does not extract one visible target in two). Samuel lifted his refusal of the auxiliary head (« si le cnn n'apprend pas au final je suis ok »). Uncommitted; build/ compiled, ctest 5/5, 3 tests `PresenceTest`, real 5-min trial without error, map checked by eye (`outputs/comparisons/2026-11-23_origine_balayage/carte_entrainement_controle.png`).

Pieces validated one by one:
- target: 8×16 map (after CNN block 4, `PRESENCE_BLOCKS`), 1 on all cells covered by a living enemy, in view (free ray within 8 m), footprint 7 m × 3 m;
- on both CNNs (actor and critic), the three trainers (ppo, ppo_liquid, ppg), permanent;
- Adam optimiser separate per CNN, `--presence_learning_rate` 3e-4, one pass per rollout in blocks of the carrier network's GPU size, after the PPO updates;
- loss: simple binary cross-entropy, unweighted (bench: ≡ weighted);
- projection: render matrix. Public contracts modified with his agreement: `AbstractRenderer::last_view_projection()` (lifted from `AbstractPlayerRenderer`), `EnemyVisionThreadPool::read_view_projection(int)`, `BaseTanksEnvironment::read_view_projections()` (protected).

Code: interface `networks/presence_model.h` (`PresenceModel`: `presence_logits`, `presence_parameters`) implemented by ConvolutionNetwork, Actor, ValueFunction, LiquidActor, LiquidCritic — validated by Samuel in place of a `vision()` getter that Claude had added without asking; `networks/vision.{h,cpp}` (`presence_grid_size`), `agents/presence_learner.{h,cpp}`, `TrainTankEnvironment::presence_maps`, `AbstractStepCollector::on_presence` (SAC: no effect), field `presence` of the 3 buffers. The probe has `--presence` and `--sightings`. Metric `pres` (head loss, actor and critic combined) added at Samuel's request 2026-11-23, bar and CSV. Checkpoints ≤ 551 no longer load (extra `presence` parameter in the CNN). The Python ExecuTorch conversion script was not verified with this extra parameter.

**Test plan:** first run = to read with `reaction.py` / `suite.py` (share of engaged passes at < 60 / 60-100 m, reference 551: 40 / 19 %, i.e. ~49 / 20 % net of background) and `decodage.py`. Non-recentring once stopped is a second defect this head does not target.

**Variant planned by Samuel (2026-11-23):** remove blocks 5 to 7 and `PRESENCE_BLOCKS`, presence head on the final 8×16 map, spatial softmax for the policy reading. He codes it himself on his other PC, with Claude's help there; offline bench proposed here, declined. Warnings given: 2026-11-16 bench (4 blocks + softmax: left/right 0.51 even supervised, comparisons/2026-11-16_cnn-bench-spatial-softmax.md), run 536 lost (spatial softmax, without a·|a|), softmax without presence signal nor size; the combination softmax + presence supervision was never measured.

**Three-readings bench (2026-11-23, `banc_lectures.py`, report §10, requested by Samuel without touching code)**: horizontal position of the nearest visible enemy, unseen episode, left/right 0.86-0.96 for the three readings (4 blocks + spatial softmax, 4 blocks + flattened, 7 blocks flattened); with blocks 1-4 trained by presence alone: 0.90-0.95. The 0.51 of softmax in the 2026-11-16 bench does not reproduce. 7 blocks flattened most precise at ≥ 60 m (0.74-0.89° vs 1.1-2.5°), softmax least precise at < 60 m (2.0-2.4°). Supervised reading: does not say what PPO will do with it. (see docs/decisions.md 2026-11-03 auxiliary head lifted 2026-11-23, removed 2026-11-25)

# Decisions register

Audience: Claude. Normative. Loaded by `rl-hypothesis` (step 2) and before any proposal of a
remedy, lever or design. Each entry is a decision of the project owner. **Refused** ⟹ never
re-proposed unless new evidence is stated first and the owner re-opens it. **Lifted** ⟹ the earlier
refusal no longer binds, within the stated scope. Add an entry the moment a lever is refused or
lifted. Public file: no names, no quotes from conversations — state the reason and scope only.

| date | lever / design | status | words / scope |
|---|---|---|---|
| 2026-07 | stack: Android/ExecuTorch dropped, OpenGL → Vulkan offscreen vision, Bullet → Jolt, RmlUi GUI | validated | desktop only |
| 2026-08-17 | training-only signal in `arenai_model` (e.g. starvation timer in the proprioception) | refused | module boundary beats the RL gain |
| 2026-09-01 | target-entropy annealing (cosine, warmup schedules) | refused | destructive in every run 08-22 → 09-01; constant target / PID Lagrangian only |
| 2026-09-23 | Beta distribution for continuous actions | removed | tried 08-29 and 09-17, needs a very high concentration; tanh Gaussian since 11-08 |
| 2026-09-24 | absolute turret/canon position command; stiffer turret motor | refused | tried 09-23 and 11-11 (run 421), reverted twice: aim is a delta (velocity) command |
| 2026-10-21 | agent at 10 Hz (`action_repeat` 3) | validated | |
| 2026-10-25 | long fire cooldown (1.5 s) to ration shells | refused | gameplay: the agent must learn restraint itself |
| 2026-10-28 | full-batch PPO update | refuted | runs 368/369: critic does not learn (ev ≤ 0); minibatches required (run 364, ev 0.69) |
| 2026-10-29 | IMPALA CNN agents | removed | 2.2× slower, no hit gain (run 364); replaced by PPG |
| 2026-11-03 | alignment / target angle / reticle bit in the proprioception (any information already visible on screen) | refused | no duplicated information: alignment is already in the vision |
| 2026-11-03 | auxiliary head / privileged label on the CNN | refused, then **lifted 2026-11-23** | lifted for the enemy-presence head only; any other auxiliary target is a new G3 question |
| 2026-11-06 | slow shell regeneration | refused | pure scarcity kept |
| 2026-11-09 | fire-rate constraint: fixed or adaptive fire cost, Lagrangian/PID on shell consumption | refused | restraint comes from the event reward and scarcity |
| 2026-11-10 | CNN perception as the cause of poor aim (perception probe, bigger CNN) | ruled out | perception works; look on the credit side |
| 2026-11-11 | changing the reward to fix SAC; non-potential aim reward | refused | the reward is shared by all agents; SAC fixes stay in SAC |
| 2026-11-12 | gun laying (canon stabilised against hull motion) | validated | |
| 2026-11-16 | training under assisted / oracle aim (assisted, weaned, mixed) | refused | oracles stay measurement tools in the probe |
| 2026-11-18 | asymmetric critic: privileged inputs (enemy geometry, yaw/pitch/distance) to the critic; any brute-force feature | refused | find why the network does not extract it from the image |
| 2026-11-22 | hard σ floor / faster entropy regulator to compensate | refused | M6: no ad-hoc rule |
| 2026-11-24 | separate aim credit on a potential Φ | refused | Φ fixes the target in place of the event reward |
| 2026-11-24 | aim gradient through Q(s,a) or latent dynamics (SVG) | refused | same information as PPO |
| 2026-11-25 | enemy-presence auxiliary head on the actor and critic CNNs (`db9a825a`, run 552) | removed | after game matches against its twin 550: no gain on game outcomes, canon held low (−14 to −21° armed) on early/mid saves, ~340 k index late; one training seed. Re-opening needs new evidence |
| 2026-11-25 | spatial softmax (`7b612934`) + spatial aim action cell + offset (`95f1812c`, run 553) | removed | more armed time on target but no more hits than 552, reserve dumped at s8 and s36; only ever measured together with the presence head. Design doc `docs/spatial_aim.md` lives in git history at `95f1812c` |
| 2026-11-25 | second vision: full-turn low-resolution panorama (4 views of 90°, 512×32, turret frame, own tank not drawn) encoded by a circular 3-block CNN, in actor and critic of ppo, ppo_liquid, ppg and sac | validated | vision stays the main observation, panorama as a lidar; pieces validated one by one (`docs/neural_architecture.md`). Enemy bearing/distance as scalar inputs was the alternative set aside |

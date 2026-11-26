# Decisions register

Audience: Claude. Normative. Loaded by `rl-hypothesis` (step 2) and before any proposal of a
remedy, lever or design. Each entry is a decision of the project owner. **Refused** ⟹ never
re-proposed unless new evidence is stated first and the owner re-opens it. **Lifted** ⟹ the earlier
refusal no longer binds, within the stated scope. Add an entry the moment a lever is refused or
lifted. Public file: no names, no quotes from conversations — state the reason and scope only.

| date | lever / design | status | words / scope |
|---|---|---|---|
| 2025-10 | training loop native in C++ on LibTorch, no Python in the loop | validated | |
| 2026-02-22 | hybrid action space: continuous (drive, aim) + discrete (fire) | validated | Bernoulli heads [fire, zoom] since 09-18 |
| 2026-03-03 | Gumbel-softmax for the discrete actions | removed | replaced by the exact expectation over discrete actions |
| 2026-07 | dense shaping added to the main reward ("shoot in aim", quality score, softmax over enemies) | removed | back and forth from 2025-10 to 2026-07 |
| 2026-07 | stack: Android/ExecuTorch dropped, OpenGL → Vulkan offscreen vision, Bullet → Jolt, RmlUi GUI | validated | desktop only |
| 2026-08-06 | shell reserve (scarcity) in place of a miss cost | validated | recharge on hit only since 10-23 |
| 2026-08-17 | training-only signal in `arenai_model` (e.g. starvation timer in the proprioception) | refused | module boundary beats the RL gain |
| 2026-08-26 | PPO family as the current agent | validated | SAC removed 09-18, rewritten 11-10 as a factorised hybrid agent |
| 2026-09-01 | target-entropy annealing (cosine, warmup schedules) | refused | destructive in every run 08-22 → 09-01; constant target / PID Lagrangian only |
| 2026-09-06 | `ppo_liquid` agent (recurrent liquid cell, TBPTT) | validated | |
| 2026-09-06 | curriculum on the spawn zone | validated | driven by hit ratio, then by kills per tank since 11-18 |
| 2026-09-23 | Beta distribution for continuous actions | removed | tried 08-29 and 09-17, needs a very high concentration; tanh Gaussian since 11-08 |
| 2026-09-24 | absolute turret/canon position command; stiffer turret motor | refused | tried 09-23 and 11-11 (run 421), reverted twice: aim is a delta (velocity) command |
| 2026-10-20 | fire action masked when the tank cannot fire | validated | |
| 2026-10-21 | reward normalisation (running mean / std) | removed | return normalisation (`ReturnScaler`) validated instead |
| 2026-10-21 | agent at 10 Hz (`action_repeat` 3) | validated | |
| 2026-10-25 | long fire cooldown (1.5 s) to ration shells | refused | gameplay: the agent must learn restraint itself |
| 2026-10-28 | full-batch PPO update | refuted | runs 368/369: critic does not learn (ev ≤ 0); minibatches required (run 364, ev 0.69) |
| 2026-10-29 | IMPALA CNN agents | removed | 2.2× slower, no hit gain (run 364); replaced by PPG |
| 2026-11-03 | alignment / target angle / reticle bit in the proprioception (any information already visible on screen) | refused | no duplicated information: alignment is already in the vision |
| 2026-11-03 | auxiliary head / privileged label on the CNN | refused, then **lifted 2026-11-23** | lifted for the enemy-presence head only; any other auxiliary target is a new G3 question |
| 2026-11-06 | hit reward from the shell's closest miss distance (07-21) | removed | replaced by damage as a fraction of health ±1, kill/death ±1 |
| 2026-11-06 | starvation (frame budget, timer), win / last-survivor reward | removed | |
| 2026-11-06 | slow shell regeneration | refused | pure scarcity kept |
| 2026-11-07 | fire cost / miss cost in the reward | removed | at least 10 attempts since 2025-12; see the 11-09 refusal |
| 2026-11-08 | entropy coefficient: fixed, learned (08-10), PID Lagrangian (08-20), fixed again (10-11) | replaced | automatic α, one per action |
| 2026-11-08 | state-independent σ (free log σ parameter per axis; tried 10-18 and 10-31) | removed | σ is a state-dependent head, soft-clamped since 11-11 |
| 2026-11-09 | fire-rate constraint: fixed or adaptive fire cost, Lagrangian/PID on shell consumption | refused | restraint comes from the event reward and scarcity |
| 2026-11-10 | CNN perception as the cause of poor aim (perception probe, bigger CNN) | ruled out | perception works; look on the credit side |
| 2026-11-11 | SAC: n-step returns (10), state encoder shared critic → actor (detached) | validated | SAC only |
| 2026-11-11 | changing the reward to fix SAC; non-potential aim reward | refused | the reward is shared by all agents; SAC fixes stay in SAC |
| 2026-11-12 | gun laying (canon stabilised against hull motion) | validated | |
| 2026-11-14 | 512×256 vision + one more CNN block | removed | the cause was the discrete target entropy |
| 2026-11-16 | training under assisted / oracle aim (assisted, weaned, mixed) | refused | oracles stay measurement tools in the probe |
| 2026-11-18 | asymmetric critic: privileged inputs (enemy geometry, yaw/pitch/distance) to the critic; any brute-force feature | refused | find why the network does not extract it from the image |
| 2026-11-21 | coloured AR(1) exploration noise + canon σ cap 0.25 (`81b4d6c0`) | removed | removed in `0137ae77` |
| 2026-11-21 | terminal state at the arena limits; received-hit direction in the proprioception | validated | |
| 2026-11-22 | aim potential Φ (potential-based shaping) | removed | tried at least 8 times since 2025-10-28 |
| 2026-11-22 | empty reserve as death (10-25), penalised terminal state (10-30), no respawn (11-13, reverted) | removed | current rule: refill + constant penalty 0.25 under curriculum |
| 2026-11-22 | value bootstrap on the last state at the episode time limit | validated | an empty reserve cuts the value target like a death (11-08) |
| 2026-11-22 | separate aim credit (aim advantage, split log-probs, per-group KL stop; in 11-12, out 11-16, back 11-21) | removed | single policy optimisation group; see the 11-24 refusal |
| 2026-11-22 | hard σ floor / faster entropy regulator to compensate | refused | M6: no ad-hoc rule |
| 2026-11-24 | separate aim credit on a potential Φ | refused | Φ fixes the target in place of the event reward |
| 2026-11-24 | aim gradient through Q(s,a) or latent dynamics (SVG) | refused | same information as PPO |
| 2026-11-25 | `arenai_agent_match`: judging a checkpoint under the desktop game's rules | validated | only instrument supporting a game verdict |
| 2026-11-25 | enemy-presence auxiliary head on the actor and critic CNNs (`db9a825a`, run 552) | removed | after game matches against its twin 550: no gain on game outcomes, canon held low (−14 to −21° armed) on early/mid saves, ~340 k index late; one training seed. Re-opening needs new evidence |
| 2026-11-25 | spatial softmax (`7b612934`) + spatial aim action cell + offset (`95f1812c`, run 553) | removed | more armed time on target but no more hits than 552, reserve dumped at s8 and s36; only ever measured together with the presence head. Design doc `docs/spatial_aim.md` lives in git history at `95f1812c` |
| 2026-11-25 | second vision: full-turn low-resolution panorama (4 views of 90°, 512×32, turret frame, own tank not drawn) encoded by a circular 3-block CNN, in actor and critic of ppo, ppo_liquid, ppg and sac | validated | vision stays the main observation, panorama as a lidar; pieces validated one by one (`docs/neural_architecture.md`). Enemy bearing/distance as scalar inputs was the alternative set aside |

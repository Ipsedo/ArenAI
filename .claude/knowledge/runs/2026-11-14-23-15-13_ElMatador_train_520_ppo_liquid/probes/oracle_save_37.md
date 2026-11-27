# Oracle on 520 save_37 (999 k): where the score is lost

Binary `build/arenai_agent_probe` (23:58). Sides 550 / 1000, 4 episodes, stochastic, economy 10/3/30, `--oracle_fov 80` (camera field). Oracle commands the 32 tanks: servo gain 0.3 on nearest enemy in field, ballistic drop compensated, no motion lead, fires if |error| < 1°. ⟹ lower bound of the ceiling, not the ceiling.

| variant | yaw | site | trigger | shots/s | hit/sh | **touches/tank-s** | %disarmed | tour<5° | median dist. aligned shots |
|---|---|---|---|---|---|---|---|---|---|
| a | agent | agent | agent | 1.31 / 0.60 | 24.5 / 23.1 % | **0.320 / 0.140** | 5 / 3 % | 31 / 24 % | 72 / 83 m |
| b | oracle | agent | oracle on yaw alone | 1.17 / 1.20 | 24.0 / 15.8 % | **0.282 / 0.190** | 16 / 28 % | 49 / 47 % | 89 / 118 m |
| c | oracle | oracle | oracle | 1.52 / 1.70 | 38.6 / 21.2 % | **0.586 / 0.361** | 1 / 5 % | 60 / 63 % | 138 / 174 m |

## Reading

- Bottleneck in the agent, not the game: even a simple oracle permits 1.8× (550) and 2.6× (1000) the policy's touches; the real ceiling is higher (no lead, 1° tolerance).
- A perfect yaw is useless without a good site: perfect yaw + agent's site (b) = −12 % at 550 and +36 % at 1000 although turret aligned 1.6-2× more often. The c − b gap (+0.30 / +0.17) is the site's and of a trigger that accounts for it. Confirms the 507 oracle finding (2026-11-12): "elevation (hausse) is the main bottleneck". The stabilizer did not lift it.
- The agent's trigger compensates its site: fires at close targets (median 72-83 m vs 138-174 m for oracle) where a site error costs little. With a trigger seeing only yaw, b wastes shells (16-28 % disarmed steps). The policy learned to choose its occasions for lack of elevation control.
- Agent aligns turret in only 24-31 % of decisions vs 60-63 % for oracle: yaw remains the second half of the gap, but earns nothing until the site follows.

## What this plan does not separate

- b's trigger ignores the site ⟹ confounds "agent site" and "trigger blind to site".
- Clean separation needs a 2×2×2 plan (each link agent or oracle). Two probe options were missing: agent trigger under oracle, and agent yaw with oracle site (`probe.cpp` change, Samuel to validate).
- The site error of column `fire_pitch` is geometric, without ballistic drop (≈ 0.3-0.7° at 70-170 m): do not compare its slices across variants with different distances.

## Proposed step 2: why the site?

Perception (is site error decodable from features?), command (does μ_canon_y respond to site error and chassis pitch?) or credit (is the PPO site gradient correlated with the corrective gesture?). Existing tools: `--features`, probe columns `pitch`/`canon_relative_elevation`/`chassis_pitch`, `arenai_agent_sigma_gradients`.

# 2×2×2 plan (yaw, site, trigger; A = agent, O = oracle), 2026-11-15

Probe modified (uncommitted): `--oracle_policy_yaw` and `--oracle_policy_fire` added to `--oracle_policy_pitch`. Oracle trigger now requires yaw **and** site within 1°: old arm b (trigger blind to site) no longer reproducible. Binary `build/` rebuilt from 0e5093a1 + this change. Raw: `factorial_save_37.txt`.

Touches/tank-s (550 / 1000):

| arm | yaw | site | trigger | touches/s | shots/s | hit/sh | tour<5° |
|---|---|---|---|---|---|---|---|
| AAA | A | A | A | 0.320 / 0.140 | 1.31 / 0.60 | 24.5 / 23.1 % | 31 / 24 % |
| OAA | O | A | A | 0.295 / 0.106 | 1.10 / 0.46 | 26.9 / 23.3 % | 51 / 49 % |
| AOA | A | O | A | 0.348 / 0.158 | 1.29 / 0.67 | 27.0 / 23.8 % | 31 / 32 % |
| AAO | A | A | O | 0.124 / 0.099 | 0.20 / 0.18 | 63 / 55 % | 38 / 31 % |
| OOA | O | O | A | **0.320 / 0.150** | 0.93 / 0.52 | 34.5 / 28.9 % | **84 / 92 %** |
| OAO | O | A | O | 0.200 / 0.181 | 0.50 / 0.61 | 40 / 29 % | 56 / 55 % |
| AOO | A | O | O | 0.169 / 0.126 | 0.31 / 0.29 | 55 / 43 % | 38 / 32 % |
| OOO | O | O | O | **0.586 / 0.361** | 1.52 / 1.70 | 38.6 / 21.2 % | 60 / 63 % |

## Reading

- No single link is the bottleneck: making one link perfect earns nothing (yaw −0.03, site +0.02-0.03, trigger −0.04 to −0.20). Even two perfect links are insufficient: OOA (perfect aim, agent trigger) = 0.320/0.150 = policy level although turret aligned 84-92 % of time. Only the perfect trio doubles/triples the score.
- The three links are co-adapted. The agent's trigger fires at approximate alignments (~5°) and compensates by volume (24 % touches per shell). Its aim rarely reaches 1° on both axes: a demanding trigger (AAO) fires only 0.2 shot/s at 55-63 % touches. System sits in a **local optimum of coordination**: marginal gain of each link, others fixed, is zero or negative ⟹ PPO has no gradient to leave it. Consistent with identical ceiling of all runs since 510 whatever the settings.
- The agent's trigger does not recognize the alignment produced by the oracle: in OOA the turret is within 1° of the enemy 24-26 % of armed time and p_fire there is only 0.13/0.07. In normal play p_fire is 0.43-0.50 when the agent is aligned (< 5°, < 40 m) ⟹ its fire decision does not rest on visible alignment alone.

## Trigger hypotheses, to test

- H1 (parallax): camera at (0, 4, −20) aims at the point at 100 m. Oracle aligns the cannon axis, not the image center, while the trigger learned a visual cue. Test: OOA with `--oracle_camera_center`; if p_fire at alignment rises toward 0.4, the trigger reads an image cue distorted by parallax.
- H2 (gesture memory): the trigger fires at the end of its own correction sequence (liquid state, proprioception), not on what it sees. Test: if H1 insufficient, decode alignment from features (`--features`) and compare p_fire depending on whether the turret was just commanded by the agent.

## Decision implied (Samuel)

A local optimum of coordination is not left by tuning a single link (entropy, target, canon credit): what runs 510-520 did. Must change what makes precision profitable for the trigger, or what links the trigger to real alignment. E.g.: an observation or camera where real alignment is legible (camera on cannon axis proposed 2026-11-12); training with assisted then weaned aim so the trigger learns to exploit precise aim; trigger-only fine-tune under oracle aim (tool `arenai_agent_aim_finetune`) to check it can learn to exploit precision.

# Parallax test and trigger dynamics (2026-11-15)

OOA with `--oracle_camera_center` (oracle site on image center instead of cannon axis):

| arm | shots/s | hit/sh | touches/s | p_fire aligned 50-150 m | p_fire at 1st aligned step |
|---|---|---|---|---|---|
| AAA (agent) | 1.31 / 0.60 | 24.5 / 23.1 % | 0.320 / 0.140 | 0.59 / 0.58 | 0.46 / 0.44 |
| OOA | 0.93 / 0.52 | 34.5 / 28.9 % | 0.320 / 0.150 | 0.12 / 0.13 | 0.18 / 0.18 |
| OOA camera centered | 1.12 / 0.59 | 29.7 / 26.6 % | 0.332 / 0.156 | 0.16 / 0.16 | 0.28 / 0.26 |
| OOO (oracle fires) | — | — | — | 0.47 / 0.42 | 0.44 / 0.43 |

"Aligned" = |yaw| < 2° and |site + drop| < 1° on nearest enemy, armed state.

- H1 (parallax): real but minor effect. Centering the target in the image raises p_fire 0.12 → 0.16 and cadence 15-20 %; far from the agent's 0.58. Explains at most a quarter of the gap.
- The trigger fires in self-sustaining bursts. Normal play: p_fire 0.63 in the second after a shot, then 0.10-0.19; in prolonged alignment it rises 0.46 → 0.79. Under OOO the oracle fires continuously and the agent's trigger stays "in burst" (0.45 at alignment, no decay). Under OOA the agent fires little, the burst does not start: 0.18 at alignment start then decay to 0.06.
- Reading: the fire decision rests mainly on an internal engagement state (having fired recently: reload seen in proprio, liquid memory, own aiming gesture) and only secondarily on visible alignment. This is the co-adaptation mechanism: more precise aim does not ignite the burst, so is not exploited, so is not rewarded.

## Not established

- Which signal sustains the burst: cooldown and reserve in proprio, or liquid memory? Proposed test: neutralize cooldown/reserve proprio inputs in the probe (new option, to validate), then compare p_fire after a shot and under OOA.
- Causal link with the ceiling (burst prevents learning precision) is a hypothesis. Test: trigger-only fine-tune under oracle aim; if it learns to fire on visible alignment, OOA must rejoin OOO.

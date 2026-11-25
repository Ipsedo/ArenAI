# Decisions register

Audience: Claude. Normative. Loaded by `rl-hypothesis` (step 2) and before any proposal of a
remedy, lever or design. Each entry is a decision of Samuel's. **Refused** ⟹ never re-proposed
unless new evidence is stated first and Samuel re-opens it. **Lifted** ⟹ the earlier refusal no
longer binds, within the stated scope. Add an entry the moment Samuel refuses or lifts something.

| date | lever / design | status | Samuel's words / scope |
|---|---|---|---|
| 2026-10-25 | long fire cooldown (1.5 s) to ration shells | refused | « ôte-toi ça de la tête le cooldown de 1.5s » — gameplay, the agent must learn restraint itself |
| 2026-11-03 | alignment / target angle / reticle bit in the proprioception (any information already visible on screen) | refused | « je veux pas dupliquer de l'info, l'alignement est déjà présent dans la vision » |
| 2026-11-03 | auxiliary head / privileged label on the CNN | refused, then **lifted 2026-11-23** | lifted for the enemy-presence head (« si le cnn n'apprend pas au final je suis ok »); any other auxiliary target is a new G3 question |
| 2026-11-06 | slow shell regeneration | refused | Samuel keeps pure scarcity |
| 2026-11-09 | fire-rate constraint: fixed or adaptive fire cost, Lagrangian/PID on shell consumption | refused | « je ne veux pas de cette contrainte de cadence à la con » — restraint comes from the event reward and scarcity |
| 2026-11-16 | training under assisted / oracle aim (assisted, weaned, mixed) | refused | « Je n'aime pas le fait d'entraîner sous visée assistée » — oracles stay measurement tools in the probe |
| 2026-11-18 | asymmetric critic: privileged inputs (enemy geometry, yaw/pitch/distance) to the critic; any « brute-force » feature | refused | « pas du tout fan de rajouter des features en mode bourrin » — find why the network does not extract it from the image |
| 2026-11-22 | hard σ floor / faster entropy regulator to compensate | refused | M6 (« pas de règle à la noix ») |
| 2026-11-24 | separate aim credit on a potential Φ | refused | Φ fixes the target in place of the event reward |
| 2026-11-24 | aim gradient through Q(s,a) or latent dynamics (SVG) | refused | « on l'oublie » — same information as PPO |
| 2026-11-25 | enemy-presence auxiliary head on the actor and critic CNNs (`db9a825a`, run 552) | removed | Samuel chose « La retirer » after game matches against its twin 550: no gain on game outcomes, canon held low (−14 to −21° armed) on early/mid saves, ~340 k index late; one training seed. Re-opening needs new evidence |
| 2026-11-25 | spatial softmax (`7b612934`) + spatial aim action cell + offset (`95f1812c`, run 553) | removed | Samuel chose « Les retirer »: more armed time on target but no more hits than 552, reserve dumped at s8 and s36; only ever measured together with the presence head. Design doc `docs/spatial_aim.md` lives in git history at `95f1812c` |
| 2026-11-25 | second vision: full-turn low-resolution panorama (4 views of 90°, 512×32, turret frame, own tank not drawn) encoded by a circular 3-block CNN, in actor and critic of ppo, ppo_liquid, ppg and sac | validated | « je préfère garder la vision comme principale observation », « comme un lidar » ; pieces validated one by one (`docs/neural_architecture.md`). Enemy bearing/distance as scalar inputs was the alternative he set aside |

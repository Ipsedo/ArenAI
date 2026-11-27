# Deferred idea: expose in-flight shells + cooldown in the observation (via the canon class)

Runs: 418 (re-proposal), none specific

Samuel's decision (2026-07-26): do NOT add in-flight shell state / cooldown to the proprioception via `BulletEnemyTank::get_proprioception()` + `ActionStats` (proposal made while debugging PPO convergence). He wants to think about integrating it into the canon class (`canon.cpp`), where cooldown and fired shells have their natural place.

**Context**: the shell reward falls at shell death, seconds after the shot, and nothing in the observation carries the state of in-flight shells → the PPO critic cannot predict these returns (POMDP). Touching `ENEMY_PROPRIOCEPTION_SIZE` breaks existing checkpoints.

**Rule**: if the subject returns (critic not converging, flat value), recall this lead and the sketched mock-up (K slots relative pos/vel + flag + cooldown fraction), but anchor it on the canon side, not the tank side. Do not re-propose the ActionStats variant.

**State at 2026-09-30**: cooldown and ammo ARE observed (`get_proprioception` ends with `nb_shells/max_shells` and `curr_cooldown_frame/fire_cooldown_frames`). Still unobserved: **in-flight shells** (relative position/velocity, where they will fall). Re-proposed after 418 then **rejected by Samuel**: the shot is visible at the frame (cooldown, image), the LTC can remember it — do not bring it out as an explanation for low ev. (see docs/decisions.md 2026-08-17: training-only signal in arenai_model refused; 2026-11-03 information duplicating the screen refused)

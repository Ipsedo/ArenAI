---
name: ppo-train-390-diagnosis
description: "Run 390 (ppo_liquid, p(fire) init 0.5, read at 78k steps) — 10 shells leave in ~3 s, hit flat 0.065, miss 20 m; recommendation initial_fire_proba 0.1 (floor 0.05 because of 30 s starvation); Hd dominated by zoom, unreadable for fire"
metadata:
  node_type: memory
  type: project
  originSessionId: 0f29c4ea-1ad8-4117-bb16-b86533e1d8a5
  modified: 2026-11-01T18:18:08.649Z
---

Run `outputs/train_390_ppo-liquid` (2026-11-01 17h16, read at 78k steps, to compare with [ppo-train-389-diagnosis](ppo-train-389-diagnosis.md)).
Binary compiled after the uncommitted edit out_of_shells -0.5, so probably -0.5 in this run.

- hit 0.065 → 0.047 when D goes to 350, miss 20-22 m, Hc 0.21 → 0.14, ev 0.2-0.5.
- Computation: decision 10 Hz + cooldown 5 frames → mean interval 0.1·(1+1/p) s. p=0.5: 10 shells in
  3 s; p=0.1: 11 s; p=0.05: 21 s; p=0.03: 34 s, beyond starvation (30 s).
- **Hd is not readable for fire**: mean over unmasked decisions, zoom dominates there.

**Why:** Samuel asked whether the actor should start at p(fire) ≤ 0.1. Hypothesis: less correlated shots; not
a proof, and it does not directly address aim.

**How to apply:** test with `--initial_fire_proba 0.1` (existing flag, does not change the default). In game
the action is p > 0.5: at 0.1 the agent may never fire until the state raises the logit by +2.2.

## From ElMatador memory

Distinct run from the 2026-11-01 one above (this 390 dates from early September 2026). Note dated 2026-09-05, run still going (3089 metrics rows). Related: 388, 389, 387.
- Commits `b7c72f69` + `2bc422e5`: aim = 0.1·(q − 0.5), win +2 only if kill, starving −2 vs death −1, final spawn back to 1000 m.
- **~2× slower than 387 AND 388 at equal index** in a comparable or smaller zone (row 1000: 2.9 kills vs 5.7/5.2). r_aim constant −0.001/step: baseline q = 0.5 requires miss ≤ √d/1.5 (≈ 9 m at 200 m, ≈ 13 m at 400 m) while mean miss is 40+ m ⇒ nearly every shell pays negative. This is the inert fire cost already documented: trigger held by the discrete-entropy PID (fire stays ~1.0/s); the −0.5 offset does not change the q shaping gradient (∂(q−b)/∂q = ∂q/∂q), it only adds negative bias and variance. Removing win +2 without kill also dries up early positive signal (early "wins" were survivals without kill).
- Commit message of `b7c72f69` says « aim reward = 1.5 * (reward - baseline) » but the code keeps hit_reward_scale = 0.1 (1.5 is miss_distance_scale, unchanged).
- **Decision (Samuel, 2026-09-05): baseline cancelled** — the aim reward must **never** penalise firing, only reward a well-executed shot. Back to pure `0.1·q` (member `aim_quality_baseline` removed from jolt_enemy_tank).
- **Adaptive curriculum implemented (2026-09-05, validated by Samuel)**: `SpawnCurriculum` (arenai_agent/src/core/), ADR style on the spawn side. Per-episode draw: 20 % probe at bound D, 80 % uniform [500, D] (anti-forgetting). Controller: hit/fire pooled over 24 probe episodes, hysteresis 0.02/0.04, ±25 m, clamp [initial, final], D starts at initial. Failure if a window has no fire. CLI `--curriculum_*`, CSV columns `side` (played side) and `D` (bound), fixed seed 1337. Old linear per-episode schedule deleted.

# PPO liquid train 407 — acceleration unlearned, final checkpoint = immobile camper (2026-09-21)

Host: ElMatador (memory notes); raw data no longer on disk.
Source note date: 2026-09-21 (`train_407_ppo_liquid`, launched 2026-09-20).

Only config diff vs 405_zoom: deeper/wider CNN (7 convs 3→128 + GroupNorm instead of 6 convs 3→64). Tanks do not move with save_30. Deterministic probe of checkpoints (random obs, Beta mode):
- save_11→22 (~1.2→2.4 M): deterministic accel rises to +0.5/+0.64 — the agent drives.
- ~2.3-2.6 M (save_23): continuous-entropy collapse (Hc 0.27 → −0.03, κ explodes), kill at max (11.6 at D=500) = local optimum of camping. α_c (PID) raises Hc back to 0.3 but accel never returns: 0.6 → 0.18 → 0.15 → ~0.
- save_28-30: second collapse underway (κ steer 69, canon_y 81) with accel ≈ 0. save_30 = immobile camper that pivots and fires (fire p = 0.58 ⇒ always fires in deterministic).
- CSV metrics are blind: hit/kill stable, curriculum keeps going (D 500→650) because camping passes the hit-ratio threshold. No speed/displacement metric exists.
- Env differences in force for 407 but not 405_zoom (commit `566c18c0`): objects spread over the whole 2000×2000 map (before: inside tank spawn zone ⇒ less nearby cover) and ZOOM_MAGNIFICATION 4→2. Candidate causes together with the widened CNN. 405_zoom (6 convs → 64 ch) did not regress.
- Tool: C++ probe loading `actor.pt` that outputs mode/κ/probas per dim; recompilable from scratchpad, link against cmake-build-release/ (build/ may be stale: it once contained an uncommitted "double conv" version of ConvolutionNetwork).
- Related: entropy regulation known gaps; event-based reward rework.

---
name: ppo-train-378-diagnosis
description: "run 378 ppo (32 tanks, 10 shells, regen 5) — hit = random level; shells pass through the terrain (discrete Jolt at 250 m/s) and fly 20 s, so the tank lives ~18 s without shells; learned P(fire) 0.98, fire/Hd metrics blind"
metadata:
  node_type: memory
  type: project
  originSessionId: 8911f436-4c10-46c0-b776-0ffffe091aef
  modified: 2026-10-30T07:39:25.011Z
---

Measured 2026-10-30 on save_0 / save_3 (6 episodes each, temporary harness outside repo):
- hit/fire 0.049 (init) vs 0.045 (final), miss 16.7 vs 17 m: no competence; D frozen at 250 because the random baseline (~0.05) falls between curriculum_ratio_low and high.
- Life ≈ 20-22 s with last shot at 2.5-3.7 s; 83-87 % of decisions with empty reserve; median time after last shot = 20.0 s = ShellItem lifetime.
- ~82 % of the aim reward arrives after 19 s (out of GAE reach).
- Terrain probe: at 250 m/s, contact 39 % at 0°, 18 % at −5°, 2 % at −45° (MotionQuality Discrete default, one-sided mesh, 8.3 m/frame).
- P(fire|can_fire) 0.5 → 0.98: each shot yields ≈ +0.035 at no cost; initial_fire_proba is not the cause.
- Misleading metrics: fire capped by the reserve, Hd dominated by zoom on empty-reserve rows, Hc = mean of σ (canon_x 0.03, canon_y 0.92).
- Death on empty reserve waits for `!shell_in_flight` since ca4583a5 (2026-10-25): all runs since are concerned. Corrects [ppo-train-371-diagnosis](ppo-train-371-diagnosis.md): s≈23.5 was not 20 shells / 0.85 fire/s but a fast dump + ~20 s wait.
See ppo-actor-audit-2026-10-29, jolt-migration-fidelity.

2026-10-30: LinearCast on ShellItem committed (7ed8a411). Measured at init policy: terrain 100 % contact in probe, but time after last shot still 11.8 s (median 13.6), 22 % of aim reward after 19 s, empty reserve 73 %, hit 0.068. 4 s lifetime not applied. Barrel fully down = shell created under the flat ground (spawn 10 m in front of the barrel). The model tests' resource reader does not read PNGs; on a cube, no tunnelling even in Discrete.

---
name: sac-train-412-diagnosis
description: "SAC run 412 (2026-11-08, α₀ 0.01, otherwise = 411) read at 39.5k steps: Q ≈ 0.9 instead of 12, fire holds 0.4-0.5/s, Hc leaves the max (2.73 → 2.46-2.52, σ ≈ 0.65), hit 0.038 → 0.0445 still below random floor; α ÷e per ~33k steps unchanged"
metadata:
  node_type: memory
  type: project
  originSessionId: 51b10db1-fb50-4f0f-81d1-b1b5b214142f
  modified: 2026-11-08T19:10:47.825Z
---

Run `outputs/train_412_sac`, only difference with [sac-train-411-diagnosis](sac-train-411-diagnosis.md): `initial_alpha` 0.01. Read 2026-11-08
at index 39.5k (≈1.27 M transitions, 90 min). Comparison at equal index (33k) 411 → 412:
Q 10.0 → 0.80; Hc 2.73 → 2.51; fire 0.30 → 0.46 shots/s; hit 0.039 → 0.044; s 40 → 62 s; kill 0.44 → 0.44.

- Q = still α·H/(1−γ) in shape (peak 0.93 at 20k then follows α), but 13× smaller: the anti-fire
  survival bonus (≈ γ·V) went from ≈10 to ≈0.9, and fire no longer slides to 0.
- **Hc drops from 2.72 to ≈2.5 from 24k**: σ leaves max entropy (0.9 → ≈0.65 per axis);
  first sign that ∂Q/∂a weighs on the actor. Target 0.51 still far.
- hit rises slowly and steadily 0.029 → 0.0445 but stays below the random floor 0.05; miss 22.5 m flat.
  No measurable aim yet. D frozen at 250.
- kill/life 1.0 → 0.44 in both runs, independent of α: not an entropy effect.
- α 0.01 → 0.0031 at the same pace as 411 (lr 3e-5); ≈1.5e-4 expected around 100k. Risk to watch:
  Hc must stop at 0.51 (Lagrangian), not collapse.
- Units verified: `s` = seconds of life (frames at 30 Hz), `fire` = shots per tank-second; one life spans
  several reserves (respawn on empty reserve, 401b7874).

**Why:** confirm that the entropy/reward scale was 411's lock.

**How to apply:** let it run; re-read around 100k steps (α ~1e-4): expect Hc → 0.51 and hit > 0.06.
`gradient_steps` > 1 remains the next option to propose (UTD 1/32).

**Follow-up (2026-11-08 evening):** 412 stopped by Samuel at 45k (Hc 2.43, hit 0.046, miss 22.8 m: nothing converged, ≈445 steps/min).
Run 413 launched 20:30: reduced network (CNN 16/32/64, MLP 128/64), **batch 8192, gpu_batch 1024**, gradient_steps still 1
→ **43 steps/min, 10× slower** (GPU 100 %), same number of updates per env step, same α decay (÷e / 33k steps ≈ 13 h).
Diagnosis: the update budget is the lock (UTD 1/32, 33k updates in 75 min), not gradient variance;
lever = `gradient_steps` > 1 with batch 256-512. See run 413 if written.

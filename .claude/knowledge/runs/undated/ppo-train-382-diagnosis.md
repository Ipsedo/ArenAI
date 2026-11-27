---
name: ppo-train-382-diagnosis
description: "Run 382 (vanilla ppo, 32 tanks, no aim reward, out_of_shells terminal) — curriculum rises 250→500 on the random floor (threshold 0.06 < random hit 0.05-0.06 at D=250), Bernoulli bias unchanged (p≈0.5) but state-dependent logits saturated (Hd 0.15), σ frozen 0.5, 0 kill in game"
metadata:
  node_type: memory
  type: project
  originSessionId: 78e69fb6-b9f9-4692-83f0-49ca42cd00a5
  modified: 2026-10-31T01:48:47.373Z
---

Run `outputs/train_382_ppo` (2026-10-30/31, vanilla ppo, 32 tanks, entropy c 0.01 / d 0.001,
initial_fire_proba = initial_zoom_proba = 0.5, curriculum ratio_low 0.03 / ratio_high 0.06,
probe_window 24, boundary 0.2). Tested in game (save_4): no aim, no kill.

- **Curriculum false positive**: at D=250 with 32 tanks, a near-random policy already gets
  hit/fire ≈ 0.05-0.06 (steps 1k-10k). Threshold 0.06 is crossed around 8k steps without any aim
  learning; D rises 250→500 while hit stays 0.06-0.10 and miss goes from 20 to 26 m.
  The hit/fire signal measures target density, not aim.
- **Bernoulli head**: the bias does not move (sigmoid(bias) ≈ 0.49/0.52 in all saves,
  orthogonal init 0.01 + logit bias), but weights grow (norms 1.9/1.4) → state-dependent,
  saturated probabilities (Hd 0.68 → 0.15). fire 5.9/s at init → 0.5-0.7/s afterwards:
  the agent rations its 10 shells over ~15 s. Episodes s = 13-30 s = out_of_shells death (-0.3),
  max 180 s. In game = argmax (sample=false): fires as soon as p > 0.5, see [ppo-train-326-diagnosis](ppo-train-326-diagnosis.md).
- **σ frozen** at init: σ bias 2.6 → σ ≈ 0.5 on [-1,1], like [ppo-train-371-diagnosis](ppo-train-371-diagnosis.md).
  The mean μ receives no usable signal → in game (action = mean) no aim.
- kill 0.2-1.4 / episode for 32 tanks ≈ nothing.

**Why:** Samuel asked whether the Bernoulli init at 0.5 was the cause; it is not the lever
(bias stays at 0.5, logits are driven by state). The real blockers: curriculum thresholds below the
random floor, and frozen σ.

**How to apply:** before reading any curriculum, measure the random hit/fire at initial D
and set ratio_low/high above it. Read head biases with a small LibTorch tool
(torch::jit::load on actor.pt, no python torch in the venv).

**Follow-up (2026-10-31, validated by Samuel)**: noise lever = initial_sigma 0.5 → 0.05 (noise ≈ 0.9° yaw / 0.36° pitch per decision, below the size of a tank at 250 m: 1.5°×0.7°) + continuous_entropy_factor 0.01 → 0.001, for the 3 agents (ppo, ppo_liquid, ppg); out_of_shells -0.3 → -0.1. Perception lever (spawn 100 m / zoom) refused: « déjà convergé avec le même CNN ». Marker to read next run: miss < 15 m within the first 200k steps.

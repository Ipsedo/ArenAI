---
name: ppo-train-339-diagnosis
description: "train_339 = 337 with epochs 2 (5.5M steps, 2026-09-30) — KL/skip fixed but clip 0.15-0.2 (3× 329); curriculum stuck at D=700 from 3.6M; hit plateaus ~0.08; ev unstable; transient collapse at 4.3M"
metadata:
  type: project
---

Run analysed 2026-09-30 (Samuel's other PC, 31 steps/s, 49 h). Config = train_337 identical
except `epochs 2` (train_338 = same config, stopped at 578k). Hc targets
`{-0.2,-0.2,-0.88,-0.88}` (aim as delta), discrete factors 0.4.

- Epochs 2 did its job: kl 0.015-0.025 (< target 0.05), skip ≈ 0-2 % (max 9 %),
  vs 0.06 / 25-45 % for 337. But clip stays 0.15-0.24 (≈ 0.10-0.15 late), 3-4× 329.
- hit/s peak 0.095-0.098 at 600-900k (D 300-350, kill ~11), then ~0.08; D rises to 700
  at 3.6M and stays there 2M steps (probes under ratio_high 0.06) → competence plateau.
- ev: slice means often negative (−2.5 at 200k, −2.1 at 900k, −7 at 4.2M); 329 was
  more stable. Collapse 4.26→4.35M (hit 0.075→0.033, kill 8→1.5) with no clip/kl spike,
  recovered ~4.5M.
- Hc falls 0.45→0.09 in 200k (338: 0.48→0.24, 329: 0.60→0.36); α_c > 0 permanently →
  one dim under floor (probably displacement). Hd 0.62→0.34 (target 0.277).

Reading: 337's KL problem is solved, the ceiling comes from elsewhere (unstable critic,
early Hc collapse). Probable best checkpoints: save_6/7/8 (≈650-860k).
See [ppo-train-337-diagnosis](ppo-train-337-diagnosis.md), ppo-liquid-agent.

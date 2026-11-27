---
name: ppo-train-337-diagnosis
description: "train_337 ppo_liquid skip+large CNN (2026-09-28) — updates too violent (clip 0.33, KL > target, 25-45 % of epochs cut), hit declining after 700k; best checkpoints save_2/3"
metadata:
  type: project
---

Run analysed 2026-09-28 (commit 1604dc5c: skip Linear+LayerNorm+SiLU in parallel with the LTC
in actor and critic, 6-block CNN 8→256 channels, neuron_number 512, epochs 4,
minibatch 1024, truncated normal, aim command as **delta** — absolute was backtracked
2026-09-24 by bb66d985, see aim-absolute-position-control). 1.21M steps in 14.4 h
(23 steps/s vs 32 for train_329: network ~27 % slower).

Finding: `clip` 0.30-0.37 and `kl` 0.05-0.065 (> target_kl 0.05) within the first 100k steps,
`skip` 25-45 % — vs clip ≈ 0.05 / kl ≈ 0.005 / skip ≈ 0 for train_329 (liquid, epochs 2,
minibatch 512) and train_336 (vanilla PPO truncated normal, epochs 1). hit/s rises to 0.08
(200-300k) then falls to 0.03-0.05 after 800k, kill 8 → 2, while 329 holds 0.09 at D=350-400.
`ev` unstable (dives to −1.3, −0.5). Mean Hc ≈ 0 with α_c > 0 → at least one dim below its
target (probably displacement, −0.2). Hd 0.61 → 0.41 (target 0.28).

Reading: the policy step per rollout is ~10× too large (epochs 4 + bigger network +
feed-forward skip branch not smoothed by the LTC); the KL guard cuts too late.
Proposed (not validated): epochs 2, or lower actor lr, or skip vs CNN ablation.
Best checkpoints: save_2 / save_3 (216k / 324k). See ppo-liquid-agent,
[ppo-train-326-diagnosis](ppo-train-326-diagnosis.md).

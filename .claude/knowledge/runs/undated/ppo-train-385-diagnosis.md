---
name: ppo-train-385-diagnosis
description: "Run 385 (384 + dense aim reward re-enabled by Samuel, commit a185f109) — at 133k steps identical to 384: r_aim flat at 0.005/step (= random baseline, decreases with D), miss 22 → 24 m, hit 0.06, μ uniform on the 4 axes; the 0.1/shell-scale oracle does not stand out of the advantage noise"
metadata:
  type: project
---

Run `outputs/train_385_ppo` (2026-10-31 19h37) = 384 + `detail.aim` put back in the return of
`get_reward` (Samuel's commit a185f109, which also committed free σ / lr 1e-4 / σ₀ 0.05 /
entropy 0.001 / empty reserve −0.1). Aim reward = 0.1 · exp(−miss / (10 m + 6°·distance)),
credited at each shell's impact (≈ 0.054 per shell at 22 m miss at 250 m).

- At 133k steps: r_aim 0.0054 → 0.0042 per step, never above its value under the random
  policy (0.0044 at step 0); it drops as D rises. miss 22 → 24 m, hit 0.06-0.07,
  kill 1.5 → 0.1, curriculum 250 → 300 (32k) → 350 (129k) on the random floor, like 384.
- μ: norms 0.07 on the 4 axes at save_1 (108k), identical displacement/aim, like 384.
- σ held (Hc −1.58 → −1.64), skip ≤ 0.03, clip 0.3, ev 0.25-0.3, Hd 0.69 → 0.63.
- Order of magnitude: ∂r/∂miss ≈ 0.0015 per metre per shell; returns are dominated by
  death −1, received −0.3, empty reserve −0.1 → the aim reward is invisible in the advantage
  (see [ppo-train-373-diagnosis](ppo-train-373-diagnosis.md): aim ×2 had already failed).

**Why:** the « aim reward as is » oracle does not discriminate: too weak to dominate the
variance of returns. Two hypotheses remain: advantage signal-to-noise ratio, or perception
(5×2 px target at 250 m, [ppo-train-345-diagnosis](ppo-train-345-diagnosis.md)).

**How to apply:** to decide, a diagnostic run where the aim signal dominates everything (aim ×10
and death/received neutralised) or a perception probe (fixed D 100 m without curriculum). Design
choice to be validated by Samuel.

---
name: ppo-train-345-diagnosis
description: "train_345 (ppo_liquid, state-independent σ, fixed entropy 0.01, BigVisionNetwork 256x128) — analysed 2026-10-18 at 11.9M steps (running): no regression, curriculum 250→850 at constant kill ~4.3, but σ rises 0.5→0.8/0.68 and kl ~3e-4"
metadata:
  node_type: memory
  type: project
  originSessionId: a79b48f0-2f23-42e5-b781-e29a3fb48841
  modified: 2026-10-18T14:47:19.849Z
---

Run launched 2026-10-16 19:02, analysed 2026-10-18 at 11.9M steps (still running, save_110).
Config = validated follow-up of [ppo-train-344-diagnosis](ppo-train-344-diagnosis.md): state-independent `sigma_logit`,
`continuous/discrete_entropy_factor = 0.01`, minibatch 2048, epochs 2, lr 1e-4, vision 256x128,
**BigVisionNetwork + CNN 8-16-32-48-64 uncommitted** (working tree at launch).

- Best curriculum run: D rises 250 → 850 without ever falling, kill stays ~4.3-4.6
  and hit ~0.075-0.083 at every stage (344: kill 2.9 at D=550 then regression). Longest D
  plateau at 700 (5.5-9.2M).
- σ (read in actor.pt via torch.jit.load) rises monotonically: displacement 0.5 → 0.80/0.77,
  aim 0.5 → 0.68/0.69 (save_110), growth slowing. Hc 0.556 → 0.627 (max ln2 = 0.693).
  Entropy 0.01 dominates the σ gradient; sampled aim stays very noisy.
- μ / discrete heads: weight norms grow (aim-yaw μ 0.01 → 0.53) → learning goes
  through the mean; discrete biases unchanged (0.41/0.26), Hd ≈ 0.68.
- kl 1e-3 → 3e-4, clip 1 % → 0.1 %, skip 0: tiny updates (wide σ ⇒ ratio barely sensitive
  to Δμ). ev 0.25-0.45 with isolated negative spikes. fire ≈ 1.1/s and s ≈ 35 constant.

Proposed (not validated): let it finish; test a recent save in game (μ mode ≠ sampled policy,
see [ppo-train-326-diagnosis](ppo-train-326-diagnosis.md)); lower only `continuous_entropy_factor`
(0.001-0.003) next run; higher lr if kl ~0.01 is wanted.

Follow-up 2026-10-19 at 16.9M (save_156, running, ~73 steps/s): D stuck at 850 from 10.0 to
15.7M (≈ 5.7M steps) then 900; kill 4.2 / hit 0.073 stable, no regression. σ keeps
rising, more slowly: displacement 0.86/0.83, aim 0.71/0.74; Hc plateaus ~0.63. kl 2-3e-4,
clip ~0.1 %. Reading: learning plateau, curriculum progress slowed ~×3.

Follow-up 2026-10-20 at 22.4M (save_207, running): D=900 since 15.7M (6.7M steps). The curriculum
rises if hit/fire of probe episodes (spawn at max D, 20 %) > 0.06 → at 900 the agent is right at
the threshold. mean hit 0.076, kill 4.05 → 4.7 (19-21M) → 4.4. aim-yaw σ nearly frozen (0.732 → 0.734),
displacement 0.90/0.87; aim-yaw μ norm 0.66 → 0.77 (save_185 → 207). kl ~2e-4, clip <0.1 %.

Follow-up 2026-10-20 at 25.0M (running): D=950 since ~24M; hit 0.073-0.083 / kill ~4.3 flat
for 1M (constant precision while D rises). kl 1.5e-4, clip 0.04 %, Hc 0.636.
Computed orders of magnitude: tank ≈ 6.5 u wide → ≈ 1000/d px at 256x128 (vertical FOV 45°,
focal 154 px); turret π rad/s = 6°/frame → aim σ 0.73 ≈ 4.4°/frame of noise vs target half-width
1.9° at 100 u, 0.6° at 300 u. CNN analysis (BigVisionNetwork): 1st conv 5x5 s4 → 8 channels
(×12 compression where 2-3 px targets live), final 4x2 map flattened = ~20° cells
→ fine localisation must be coded in channels. Proposed (not validated): continuous entropy
0.001-0.003 first; spatial softmax (x,y keypoints) on 16x8/32x16 map instead of flatten;
1st layer stride 2 + more channels; linear « enemy bearing » probe on frozen encoder
to prove whether the CNN is the bottleneck.

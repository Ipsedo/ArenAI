# PPO liquid train 417 — 416 with epochs 2 (2026-09-28)

Host: ElMatador (memory notes); raw data no longer on disk.
Source note date: 2026-09-28. `train_417_ppo-liquid_skip_larger-cnn_epochs-2`: identical to 416 except `epochs` 4 → 2. Vs 414: LTC skip actor+critic, CNN 6 blocks (2×4×256), LTC 512 — three arch changes not separable.

At 0.58 M steps (D=300):
- **D=250: hit 0.089 / kill 9.82** (414: 0.073 / 6.25; previous best at this stage 407: 7.3). **D=300: 0.098 / 11.54** (409: 9.6). Gap stable over 4 consecutive 100 k-step slices. Best start ever measured at that time.
- Healthy optimisation: kl 0.007-0.015, clip 0.11-0.18, skip 0 (416: kl 0.06, skip 0.3-0.4). Bigger network raises KL ×3-4 vs 414 at equal lr and epochs.
- `ev` 0.20-0.24, **lower** than 416 (0.4-0.5): 416's « better critic » came from 4 epochs, not skip (correction of 416).
- `miss` unchanged (22.6 vs 21.4 at D=250) despite hit +22 %.
- ~113 k steps/h ⟹ 3 M steps ≈ 26 h.
- 416 → 417 isolates `epochs`: 4 epochs with this network = skip regime that put it under 414.
- Criterion to confirm: do not conclude before D ≥ 500-600 — 409 also shone early (12.5 at D=500) then fell to 9.6 by D=600. Holds hit > 0.09 at D=600+ and beyond ~2.8 M (414 plateau)?

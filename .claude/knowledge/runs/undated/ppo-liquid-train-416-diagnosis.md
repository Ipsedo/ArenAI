# PPO liquid train 416 — skip + larger CNN + epochs 4 (2026-09-28)

Host: ElMatador (memory notes); raw data no longer on disk.
Source note date: 2026-09-28. `train_416_ppo-liquid_skip_larger-cnn` (commit `1604dc5c`), ran on another machine, metrics copied locally. config.json `output_folder` says `train_337_…` (folder renamed afterwards).

**Four changes at once vs 414**: LTC skip actor + critic; CNN 6 blocks 3→8→…→256 (final map 2×4×256 = 2048 features); `neuron_number` 128 → 512; `epochs` 2 → 4.

Measured at 1.22 M steps (D=350):
- Optimisation went from 16× under target to the stop: `kl` 0.055-0.065 (skip threshold 1.5 × 0.05 = 0.075), `clip` 0.30-0.36, **`skip` 0.27-0.43**. 414: kl 0.004, clip 0.05, skip 0.
- **Under 414 at every step**: kill 5.37/6.21/3.12 vs 6.25/7.21/3.96 (D=250/300/350), hit likewise. Stuck 0.55 M steps at D=350 (414: 0.33 M). Trough at 1.06 M: kill 0.62, `s` 30.7 (starvation floor), `v` 0.023 — signature of collapses 375/383 — then partial rebound (3.7 / 2.3).
- `ev` median 0.39-0.51 vs 0.24-0.29 for 414 — **due to 4 epochs, not skip** (417, same arch at 2 epochs: ev 0.20-0.24). `miss` slightly lower.
- ~87 k steps/h vs ~230 k for 414: 3 M steps ≈ 35 h.
- Why: with four variables nothing is attributable; the KL/skip regime (very probably `epochs` 4 + 4× bigger network at same lr) masks the arch effect. KL-lock history: 343, 355 (skip 0.52 ⟹ collapse).
- Relaunched at `epochs` 2 = run 417, which confirms the KL regime was dragging 416 down.

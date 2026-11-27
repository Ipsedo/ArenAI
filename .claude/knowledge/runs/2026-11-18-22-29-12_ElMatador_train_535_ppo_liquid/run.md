# train_535_ppo_liquid

| | |
|---|---|
| machine | ElMatador |
| user | samuel |
| launch | 2026-11-18 22:29:12 |
| commit at launch | `cc4ee96b` — add new graphic parameters to GUI |
| commits ≤ 90 min after | none |
| diff at launch | unknown (identity recorded after the run ended) |
| raw data | `outputs/train_535_ppo_liquid` on ElMatador |

## Established facts

- 2026-11-18 ~22:50: run 535 = 534 + `--actor_learning_rate 5e-5` (sole change; commit b83da585); 534 stopped ~91 k. kl ÷2.5 at the start then same plateau 0.065 (mean of a kl cut at 0.15, uninformative); skip shifted by ~40 k without changing slope (0.37 at 80 k, 0.48 at 97 k; 534: 0.58 at 80 k). CSV ≡ 534 at equal index (kill 20.8 / hit 0.100 at 80 k; 534: 20.5/0.107). Probe s3: corrective 0.61/0.61, gain −0.018, yaw < 5° 5.9/5.7 % ⟹ signal present (third time) but behind 534. Per-step kl growth with training is generic (×5-6 between 2 k and 80 k in 526/528/529 too); no loop stabilises skip (416 stable at 35-43 %, 449 went to 96 %). Compare s5/s9 to 528 s9 (dwell 0.26 s) and 529 s9 (0.53 s); any skip fix is judged against 534 s3 at same index.
- 2026-11-19 (skip archaeology, comparisons/2026-11-18_skip-kl-rise-507-535.md): 507/508 = last healthy regime (kl 0.005 → 0.011 flat, clip 0.16, skip 0, target 0.05). Actor maths identical since. 509 (factored credit): kl ×3. 510-522: the group stop masked (skip = rest = 0, skip_canon 10-30 %). 523 (removal + flat 7-block CNN) did not bring back 508: skip 0.24 at 80 k at target 0.05, kl growing ×6-7 with training. Target 0.10 (524) only delayed. CSV mean kl plateaus at ~0.6 × 1.5 × target once the stop is active. Unproven suspect for the 508 → 523 gap: the flat CNN (gamma, discrete prob, Φ ruled out by run pairs).
- 2026-11-19: KL per action measure (`arenai_agent_sigma_gradients --kl_breakdown`, uncommitted; method `LiquidPpoTrainer::kl_breakdown`): replays actor epochs without stop on a checkpoint's rollouts. KL = 94-100 % μ displacement, specific to each state; trigger/zoom/σ ≈ 0. KL at 8 steps: 534 s0 0.003 → s1 0.107 → s3 0.15 (white noise on same checkpoint 0.06); 529 s3 white 0.025. Measured cause: ‖mu.weight‖ starts at 0.02 and grows unbounded (KL per step ∝ (‖W_a‖/σ_a)²); at 81 k: 0.22-0.24 for 507/508 and 0.195 for 526 (kl flat 0.011), 0.39-0.50 for 523-535. Coloured noise = ×2-3 more at equal weights. `actor.pt` can be read in Python with `torch.jit.load` (`.venv/bin/python`), all architectures.
- 2026-11-19: probe s5 (135 k, D 400): aim stagnant (corrective 0.59/0.60, gain −0.017, yaw < 5° 6.0/5.3 %), μ saturated 17 % (s3: 1.7 %), trigger flat 0.65. CSV falling back since 110 k (kill 21 → 16, hit 0.107 → 0.081 at 171 k, D 450; 528 at same D: 24 / 0.156), skip 0.66 ⟹ the coloured-noise signal appears at s3 but does not grow under the KL regime.

# Continuous entropy bonus at 0 ⟹ σ collapses, D stuck at 250

Runs: 430, 433, 435, 439, 440 (also 429, 437)

Since 9fe8ddf9 (2026-10-24) the continuous entropy bonus is 0 ⟹ σ collapses within the first 100 k, hit plateaus at 4-5 % < curriculum threshold 6 %, D never leaves 250 (433-439); 430 with 0.001 held σ 0.45 and passed D (clean pair 430/433).

Pair 430 / 433 (only differences: cont_entropy 0.001 → 0, epochs 2 → 6, p_fire 0.4 → 0.25):

| | 430 (bonus 0.001) | 433 / 435 / 439 (bonus 0) |
|---|---|---|
| Hc 0 → 500 k | 0.54 → 0.48 (σ ≈ 0.45) | 0.53 → 0.25 / −0.14 (σ ≈ 0.3 / 0.2) |
| kl / clip | 0.004 / 0.04 | 0.02-0.07 / 0.2-0.5, skip up to 50-70 % |
| hit | 2.5 → 5.6 % at 700 k, monotone | 4-4.5 % from 200 k then plateau, then ≤ 3 % |
| D | 300 at 800 k, 450 at 5 M | 250 whole life |

Mechanism: the PPO surrogate pulls σ down (cf. run 343); without bonus σ sharpens early (fast gain 0-300 k), then falls below ~0.3 (Hc < 0.2): aim no longer explored, hit plateaus under the 6 % threshold, then drifts (μ random walk, kl at the stop). More epochs = faster collapse (433 epochs 6: Hc → 0 in 400 k; 435/439 epochs 4: in 1 M).

Samuel had already found this on 2026-10-21 (commit 10392476: « 0.0 converges to broken policy »), then the bonus was removed again 2026-10-24 (9fe8ddf9) and the factor code removed in d01d5432 (2026-10-26). The 7 following runs all have the same profile.

**Why:** the variable separating runs that leave D = 250 from those that do not, at same env (reserve 20, no regen). Ammo economy (430 vs 429) slows but does not block.

**Rule:** restore `continuous_entropy_factor` 0.001/dim (value measured on 430; 2e-3 was the vanilla calibration of 340/341 if σ still drifts under 0.3). Target operating zone: Hc ≥ 0.2 ⟺ σ ≥ 0.3, cf. comparisons/2026-08-17_ppo-entropy-regulation-known-gaps.md (golden zone σ 0.4-0.8). See runs 439, 433, 430.

**Run 440 (launched 2026-10-26 ~12 h, commit 5a2f8164)**: config = 439 + continuous **0.001** (not 0.01) + discrete 0, epochs 4, γ 0.994, p_init 0.01/0.05 unchanged. At 80 k: identical to 439 on everything, including instant σ drift (Hc 0.564 → 0.515 on both). Pitfall: `metric_window_size = 4096` updates ≈ 110 rollouts ≈ 100 k steps ⟹ the CSV trainer columns are cumulative means up to ~100 k then lagged ~50 k; de-cumulate to read the instant. Expected verdict at 200-300 k: Hc ≥ 0.5 = bonus holds (430); ≤ 0.45 = 0.001 too weak with epochs 4 (calibration 340/341: 1e-3 marginal, 2e-3 holds). Relaunched 2026-10-26 ~13 h with `metric_window_size` 256 (≈ 7 rollouts ≈ 6 k steps: trainer columns near-instantaneous, like 437) — the 80 k above is the old 440.
(see docs/decisions.md 2026-11-08: entropy coefficient replaced by automatic α per action)

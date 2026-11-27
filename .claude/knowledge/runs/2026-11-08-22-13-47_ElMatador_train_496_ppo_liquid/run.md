# train_496_ppo_liquid

| | |
|---|---|
| machine | ElMatador |
| user | samuel |
| launch | 2026-11-08 22:13:47 |
| commit at launch | `b74e338a` — add trained model for ppo liquid |
| commits ≤ 90 min after | `cf99aef4`, `564f2656` |
| diff at launch | unknown (identity recorded after the run ended) |
| raw data | `outputs/train_496_ppo_liquid` on ElMatador |

## Established facts

- 2026-11-08: Run (launched 08/11 22:13) = config 494 + two changes (commit `cf99aef4`): tanh Gaussian law and automatic entropy coefficient learned per action (log α, loss α·(H − target), Adam 3e-4, init 1e-3; continuous target −1.0 per dim, discrete target = entropy of Bernoulli(0.05)). Economy unchanged 10 / +2 / 20, respawn at empty, p_init fire 0.5. First run to pass 700 k without KL regime nor extinction; the lever that works is σ regulation, not reward. Hc not comparable to pre-tanh runs (entropy of squashed law, saturated μ count negative).
- 2026-11-09: Summary at 800 k vs 494 and 495:
  - kill 13-18/episode from 100 k (494: 4-7, 495: 10-13), r_hit 0.005 vs 0.0025; CSV hit identical (0.08-0.10): gain comes from fire rate (1.1-1.7/s vs 0.5-0.9), not precision.
  - KL regime avoided: skip peak 0.23 at 385 k (D 500 → 450) then 0.05 at 736 k, kl 0.02-0.03 (494: skip 0.54, Hc −0.45, D 900 → 850). D 750 at 744 k (494 was there at 461 k but regressed after).
  - α falls to ~0 in 30 k (Hc > target), rises only when Hc < −1.0 (canon 0.004-0.008 at 400-440 k, dir_y 0.013 at 660 k): slow regulator (~150 k to bring Hc −1.4 → −1.0) but effective. Probe save_15 (405 k): σ canon 0.081/0.043 = collapse in progress; save_29 (783 k): 0.154/0.117. Exactly the σ↔skip loop that killed 494, interrupted this time.
  - Trigger: save_1 and save_15 = dump (p_fire 0.73 flat, empty in 2.3 s); save_29 = first selectivity (0.37 aligned vs 0.21 off-axis at side 750; 0.41 vs 0.26 at side 550), reserve empty in 4.5 s, 60-65 % armed decisions.
  - At side 550 and equal save, 494 stays ahead in precision: p_fire 0.53/0.20 (ratio 2.7 vs 1.6), hit/shell 18.5 % vs 10.6 %, hit of shots <5° 31 % vs 23 % — with σ 0.09 on canon_y. The entropy floor costs shot precision; buys stability. 496 turret slightly better aligned (dec. <5° 19.6 % vs 16.4 %).
  - Trigger trajectory at side 550 (ratio p_fire <5° / >20°, noise ±0.2-0.3/episode): save_15 1.05 (flat) → save_20 1.24 → save_25 1.82 → save_29 1.60 → save_30 1.54. Real progress 400-675 k then plateau 1.5-1.8 from 675 k while level rises (0.35 → 0.45, empty in 2.9 s) and σ falls again (0.18/0.09 → 0.10/0.08): « level rises, gap frozen » pattern of 493. 494 at same saves: 1.8 → 2.1 → 2.8 → 3.1 → 2.7.
  - Hc stays under target (−1.2/−1.3) at 800 k with α canon back ~1e-4, skip rises 0.05 → 0.13: to watch, not yet a regime.
- 2026-11-09: Cause of the delay (verified): log α without floor (`entropy.cpp`, `AlphaParameters`), Adam ±3e-4 per step regardless of error ⟹ windup toward 1e-7 while Hc > target, then 150 k to rise. Structural oscillations. Proposed anti-windup floor α ≥ 1e-4 + ceiling 0.1 for a run 497. **Decision by Samuel 09/11: let 496 run**, no parallel 497 (CPU/render bound); re-probe near 950 k: if ratio > 1.8 when σ returns, waves were the brake; else brake elsewhere. α fire/zoom = 0 all run (Hd > H(Bern 0.05)).
- 2026-11-09: 11:15, 1.17 M, D 850 since 1.13 M: 2nd wave resolved itself (α canon 0.006-0.008 at 1.05 M, Hc −1.3 → −0.3, skip 0.19 → 0.06). Probe side 550: save_37 (1.0 M) ratio 1.66 (σ 0.18/0.09) → save_43 (1.16 M) ratio 2.58 (4 episodes 2.1-2.8; p 0.34 aligned / 0.13 off-axis, σ 0.31/0.40, empty in 5.5 s, 0.77 shot/s, hit/shell 10.5 %). Hypothesis confirmed: trigger progresses only in windows where σ rises. Ratio = 494 save_29 (2.67) at lower level (0.21 vs 0.34), but aligned shot precision 19 % vs 31 % (high σ). α re-winds (1e-4 at 1.15 M, Hc above target) ⟹ 3rd wave coming.
- 2026-11-09: 13:40, 1.37 M: 3rd wave in progress, deeper. D stuck 850 since 1.13 M, skip 0.07 → 0.29 (run max), kl 0.03, α canon 0.0000 (wound), Hc fell back −0.3 → −1.01 only now ⟹ alphas react ≥ 150 k. Probe side 550: save_47 ratio 2.50, save_50 2.22 (episodes 1.9-2.5): no progress since save_43 (2.58), σ canon 0.31/0.40 → 0.16/0.10. hit/shell 11 % (run max), turret <5° 18.8 %, |μ| canon at bounds 2 %. Kill 14 → 5 at 1.25 M then 11.
- 2026-11-09: Structural reading: `cf99aef4` replaced ppo_liquid's constant factors with unbounded `AlphaParameters` + Adam = the design of run 355 (windup documented in memory note on PPO entropy regulation). Three oscillation causes: unbounded integrator (windup), Adam on log α = relay (fixed step regardless of error ⟹ limit cycle), target −1.0 ⇔ measured σ canon 0.10-0.16 = already the skip zone (skip > 0.15 as soon as Hc < −1.1 over 5 saves; healthy windows Hc ≥ −1.0, σ ≥ 0.16). `PidLagrangianAlphaParameters` (Stooke λ ≥ 0, linear, bounded I, immediate P) exists in `entropy.cpp` but not wired. Proposed: wire it in `EntropyAlphas` + target ≈ −0.6 (σ ≈ 0.2-0.25), gains k_p 0.2 / k_i 5e-3 / k_d 1; minimal variant = floor on log α only (led to 497 / 499).
- 2026-11-09: How to apply: read trigger and σ at probe (`analyze_probe.py` in scratchpad), never Hc alone; compare with 494 at equal side; if skip returns > 0.3 with Hc < −1.2 and α < 1e-3 the regulator is too slow (alpha_learning_rate or init).

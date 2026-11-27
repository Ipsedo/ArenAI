# train_501_ppo_liquid

| | |
|---|---|
| machine | ElMatador |
| user | samuel |
| launch | 2026-11-10 09:34:59 |
| commit at launch | `15b32ece` — move sac alpha to entropy.cpp |
| commits ≤ 90 min after | none |
| diff at launch | unknown (identity recorded after the run ended) |
| raw data | `outputs/train_501_ppo_liquid` on ElMatador |

## Established facts

- 2026-11-10: Change (decision after 500's regression; Samuel: « Oui, 0,7 »; commit `4d7d558c`): `LOG_SIGMA_MAX` 2 (σ ≤ 7.4) → ln 0.7 = −0.3567 (pre-tanh σ ≤ 0.7) in `arenai_agent/src/networks/constants.h:13`, shared by `actor.cpp` and `liquid_actor.cpp`; `LOG_SIGMA_MIN` −5 unchanged; `tests_actor.cpp` reads the constant. Only change vs 500. σ_max = 0.7 = 1/√2 = exact unimodality threshold of tanh(N(0,σ²)) (density in y ∝ 1 + y²(1 − 1/(2σ²)) near 0); checked numerically: unimodal for μ ∈ {0; 0.5; 1} up to 0.7, bimodal from 0.8 (μ = 0) and 1.0 (μ = 0.5); max Hc per dim +0.65 at σ 0.7 then decreasing (−0.01 at σ 2) ⟹ on [0; 0.7] squashed entropy is monotone in σ and the α regulator regains meaning. Alternative 1.0 (ln = 0) rejected: slight bimodality + non-monotone zone. Why: good regimes (494-500) live at median σ 0.25-0.5 (init 0.3), ceiling does not touch them; it removes only the bang-bang attractor of 500's waves (median σ 1.2-2, 33-48 % saturated actions). Compare to 500 at equal index (D 1000 at 595 k; save_20 hit/shell 22 %, reserve 26.6 s, ratio 4.66); vigilance becomes σ↔skip loop (500: skip 0.35 at 500-540 k; 494 regressed at 0.45-0.54).
- 2026-11-10: Start (12:00 at 178 k, 2 h 10 of run, ~82 k/h): config.json identical to 500; curriculum step-for-step (D 450 at 144 k as 500/499), r 2.65e-3 ≡ 500, hit 11.8 % vs 13.5 %, kill 9.8 vs 11.7, fire 0.94 vs 0.75 (noise; 500 passed D=500 at 180 k). Hc 0.43 vs 0.27 at 500: mechanical effect of ceiling (σ 0.7 = max Hc 0.65/dim), not behaviour change. Probe (2 episodes, side 550): save_3 σ_canon 0.40/0.35, dump 8.8 s, p_fire 0.54 flat; save_6 σ_canon at ceiling 0.70 in 64 %/36 % of states, |μ_canon_x| 0.38 (500 save_6: 0.16), reserve 30 s, p_fire 0.107, ratio 3.8 (500 save_6: 5.3), hit/shell 11 % vs 12 %. Pitfall: probe links recompiled `libarenai_agent.so` ⟹ σ of 500's saves is **clamped at 0.7 by the probe** (34-63 % of states at ceiling = true σ ≥ 0.7 already at 160 k); push toward high σ exists from 80-160 k in both runs; 500's waves start at 250 k (Hc −1.3 at 350 k, skip 0.3 at 450-550 k), where 501 is judged.
- 2026-11-10: First verdict (14:00, 362 k, 81 k/h, save every 20 min): 500's Hc wave (0 → −1 between 275 and 350 k, kill 12 → 3, D stuck 700) does not occur with σ ceiling 0.7. CSV windows 275-350 k: 500 Hc −0.04 → −0.98, kill 6 → 3, hit 11 → 8.6 %, D stuck 700; 501 Hc 0.31 → 0.29 (min 0.11, never negative), kill 7-12, hit 12.1-12.5 %, D 700 → 800. D 800 at 325 k vs 500's 486 k ⟹ 160 k ahead. skip 0.02-0.03, kl 0.018 (target 0.05): no KL regime. Hc drifts slowly 0.47 (210 k) → 0.29-0.32 (340-360 k). Probe (2 episodes, side 550):

| | 501 save_13 (351 k, D 800) | 500 save_18 (486 k, D 800) | 500 save_13 (351 k, wave) |
|---|---|---|---|
| σ_canon_x/y median | 0.70 / 0.70 (ceiling 88 %/70 %) | 0.26 / 0.21 (collapsed post-wave) | clamped by probe (true σ ≫ 0.7 in 90 %) |
| \|μ_canon\| mean | 0.59 / 0.60 | 0.49 | 0.44 / 0.68 |
| p_fire aligned / off-axis | 0.405 / 0.044 (ratio 9.2) | 0.648 / 0.127 (5.1) | 0.58 / 0.07 (8.3, biased) |
| hit/shell | 15.0 % | 10.6 % | 9.3 % |
| reserve empty median | 37.6 s | 16.5 s | 19.8 s |
| turret <5° | 13 % | 15 % | 12 % |

  Ratio 9.2 and reserve 37.6 s = record of all probes (500 save_20: 4.66 / 26.6 s / 22 % hit/shell). Why: bimodal attractor (pre-tanh σ > 0.7 → bimodal → Hc negative → skip) is indeed the mechanism of 500's wave; without it policy keeps σ at ceiling and carries signal through μ. Vigilance moves to Hc drift via growing |μ| with σ frozen at ceiling: α = 0.001 can push nothing (σ bounded), so if Hc continues toward 0 or below it is μ at the bounds (read |μ| > 1.5), not σ. Probe clamps 500's σ: do not compare raw σ 500 vs 501 at probe.
- 2026-11-10: Second verdict (17:00, 600 k): D 1000 at 560 k (500: 595 k) but entropy loop came back through μ: σ frozen at ceiling 0.7, |μ_canon| 1.0 with 23-28 % of states |μ| > 1.5, Hc 0.3 → −0.1, skip 0.02 → 0.22 monotone; probe save_21 below 500's record save_20 (hit/shell 12.9 vs 20 %, reserve 16.6 vs 62 s, ratio 8 vs 15). Saves ≥ 14 in old `mu` Sequential format: convert a copy with `arenai_agent/python/convert_mu_linear.py` (without --in-place, output `converted/`) and point the probe at the copied folder. CSV 425-600 k, 500 vs 501 at equal index: 501 Hc 0.26 (400 k) → −0.10 (560-600 k), skip 0.03 → 0.22 monotone, kl 0.019 → 0.026; hit 10-12 %, kill 7-10; 500: skip peak 0.35 at 500-530 k then 0.10 at 590 k; hit 13-15 % at 500-580 k. Continuous α pinned 0.001 in both runs (target Hc −1 never reached before 600 k); only α_fire moves (0.006-0.019). Probe save_21 (567 k), side 550:

| | 501 save_21 | 500 save_20 (record) | 500 save_21 |
|---|---|---|---|
| σ_canon_x at ceiling | 80 % | 15 % (probe-clamped) | 58 % |
| \|μ_canon_x\| mean / frac > 1.5 | 1.07 / 28 % | 0.43 / 0 % | 0.40 / 0 % |
| p_fire aligned / off-axis (ratio) | 0.57 / 0.072 (8.0) | 0.28 / 0.019 (15) | 0.32 / 0.035 (9.0) |
| hit/shell | 12.9 % | 20.0 % | 17.7 % |
| reserve empty median | 16.6 s | 62 s | 65 s |
| turret <5° | 17.9 % | 11.8 % | 13.2 % |

  Side 1000: 501 hit/shell 8.7 %, reserve 19.8 s; 500 save_21 11.3 %, 104 s. 501 save_13 (351 k) was at 15 % / 37.6 s: regression between 350 and 570 k. Why: ceiling removed bimodal σ attractor but not the pressure to reduce entropy; with σ bounded, policy pushes μ out of tanh's linear zone (saturated ±1 actions), Hc negative again, skip rises. Same underlying mechanism (bang-bang), other variable. 500's saves at probe have σ clamped at 0.7 by recompiled lib, their reserve is perhaps flattered. At equal D 501 does not improve 500; arbiter = hit/shell + reserve at probe, not the Hc wave. Remaining entropy-side lever = the target (−1 never reached ⟹ α inert) or `LOG_SIGMA_MIN`, not the ceiling; do not propose cooldown nor cadence constraint (see docs/decisions.md).

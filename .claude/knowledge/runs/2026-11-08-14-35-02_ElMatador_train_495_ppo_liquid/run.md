# train_495_ppo_liquid

| | |
|---|---|
| machine | ElMatador |
| user | samuel |
| launch | 2026-11-08 14:35:02 |
| commit at launch | `b74e338a` — add trained model for ppo liquid |
| commits ≤ 90 min after | none |
| diff at launch | unknown (identity recorded after the run ended) |
| raw data | `outputs/train_495_ppo_liquid` on ElMatador |

## Established facts

- 2026-11-08: Launched by Samuel 08/11 14:35 from `cmake-build-release` (binary rebuilt 14:35), ~83 k/h, saves ~27 k. Only config difference vs 494: `continuous_entropy_factor` 0.001 → 0.01 (3rd option from 494's summary, against the σ↔skip loop). Change in `liquid_ppo_hyperparams.h` in working tree, not committed.
- 2026-11-08: CSV at 174 k vs 494 at equal index (494 / 495):

| index | D | kill | hit | fire | Hc | kl | skip | clip |
|---|---|---|---|---|---|---|---|---|
| 53 k | 300 / 300 | 9.1 / 8.3 | 0.099 / 0.092 | 0.80 / 1.31 | +0.16 / +0.31 | 0.012 / 0.008 | 0 / 0 | 0.18 / 0.12 |
| 107 k | 350 / 350 | 7.9 / 13.2 | 0.096 / 0.110 | 1.05 / 1.60 | +0.10 / +0.35 | 0.012 / 0.009 | 0 / 0 | 0.18 / 0.14 |
| 161 k | 400 / 450 | 6.6 / 11.9 | 0.084 / 0.091 | 0.80 / 1.27 | +0.08 / +0.39 | 0.017 / 0.012 | 0.001 / 0 | 0.23 / 0.18 |
| 174 k | 400 / 450 | 8.1 / 12.1 | 0.096 / 0.092 | 0.92 / 1.19 | +0.06 / +0.41 | 0.017 / 0.013 | 0.007 / 0 | 0.23 / 0.19 |

  Steps: 300 at 47 k, 350 at 67 k, 400 at 114 k (as 494), **450 at 144 k (494: 180 k)**. ev 0.37-0.42 (494: 0.31-0.39). Hc rises but decelerates (0.38 at 125 k, 0.41 at 174 k).
- 2026-11-08: Probes side 250 (4 episodes, step ≥ 10, armed; script `analyze_probe.py` = `analyze_491.py`):

| save (≈k) | <5° | 5-20° | >20° | ratio | level | median reserve | armed | hit/shell | shots <5° / dec. <5° | σ canon |
|---|---|---|---|---|---|---|---|---|---|---|
| 494 s2 (54) | 0.60 | 0.60 | 0.58 | 1.03 | 0.59 | 2.8 s | 33 % | 11.9 % | 14.1 / 13.7 % | 0.29/0.28 |
| 494 s7 (190) | 0.58 | 0.56 | 0.46 | 1.26 | 0.52 | 3.3 s | 35 % | 12.5 % | 14.6 / 13.0 % | 0.26/0.25 |
| 495 s3 (81) | 0.56 | 0.60 | 0.56 | 1.00 | 0.58 | 3.1 s | 46 % | 13.0 % | 14.3 / 13.9 % | 0.36/0.34 |
| 495 s6 (162) | 0.68 | 0.61 | 0.50 | 1.35 | 0.58 | 3.2 s | 45 % | 14.3 % | 16.1 / 14.4 % | 0.41/0.37 |

  |μ| > 0.9: 0 % everywhere. 9.4 respawns/tank-episode, 1.4 shot/s at probe (494 s7: 0.87/s, 7.4 respawns). p_fire 0.64 at full reserve (spawn reflex), 0.60 at 1-2 shells: not yet decreasing with reserve (494 became so at save_14). At 174 k nothing worse than 494 and two things better (KL regime pushed back: kl ÷1.3, skip 0, clip 0.19 = expected effect of larger σ on KL ∝ Δμ²/σ²; curriculum 36 k ahead); dump intact (3.2 s vs 494's 3.3 s); σ canon rises 0.30 → 0.41 instead of falling; named risk (494 summary): σ > ~0.5 hurts long-range aim (D ≥ 600). 494's reserve exceeded 5 s only at save_11 (297 k), 9 s at save_14 (378 k): gate not readable before ~300-380 k.
- 2026-11-08: Result at 469 k (saves 11, 14, 17 side 250; save_17 side 550). **Gate failed, run behind 494.** CSV: D 500 at 180 k (494: 233 k) then **stuck 500 until 328 k, 550 since** (140 k); 494 at same index: 550 at 267 k, 600 at 328 k, 750 at 461 k. hit CSV 0.08-0.09 (494: 0.10-0.11), fire 1.1-1.5/s (494: 0.5-0.9), kill 10 (shorter D), Hd 0.66 → 0.29, Hc +0.40 flat (σ held by bonus), **skip 0-0.01 and kl 0.02: KL regime avoided** (494: skip 0.28 at 470 k), ev 0.27-0.37.

| save (≈k) | <5° | >20° | ratio | level | median reserve | hit/shell | dec. <5° | σ canon | \|μ\| > 0.9 |
|---|---|---|---|---|---|---|---|---|---|
| 11 (297) | 0.72 | 0.42 | 1.74 | 0.54 | 3.2 s | 14.6 % | 14.1 % | 0.48/0.39 | 0/1 % |
| 14 (378) | 0.89 | 0.61 | 1.45 | 0.73 | 2.6 s | 13.1 % | 15.0 % | 0.48/0.36 | 4/10 % |
| 17 (459) | 0.84 | 0.59 | 1.44 | 0.71 | 2.6 s | 13.1 % | 11.7 % | 0.49/0.30 | 4/9 % |
| 17, side 550 | 0.64 | 0.39 | 1.65 | 0.50 | 2.8 s | 6.5 % | 12.3 % | n/a | 2/10 % |

  Reference 494 save_14 (378 k): 0.34/0.14 level 0.21 reserve 9.3 s hit/shell 17.9 %; side 600: 0.23/0.09, reserve 10.7 s, hit/shell 9.5 %. p_fire 0.93 at full reserve (spawn reflex), 0.78 at 1-2 shells.
- 2026-11-08: Reading: contrast peaked at save_11 (1.74) then **level rose toward saturation** (0.54 → 0.73): dump stronger than at start (2.6 s), opposite of 494 where level fell to 0.20 and reserve lasted 10 s at same index. Consequence: CSV hit under curriculum threshold 0.08, D stuck 550. Per-aligned-shell precision intact at side 250 (26-31 %) but at range (side 550) hit/shell 6.5 % vs 9.5 % for 494 at side 600, and μ canon starts to saturate (|μy| > 0.9: 10 %) with σ 0.48 held by the bonus. Entropy 0.01 broke the σ↔skip loop but at the price of the trigger gate: large σ = dump returns. One run each side, no demonstrated mechanism; measured: same code, same reward, only the bonus differs, trigger goes the other way. 495 behind 494 on everything except skip; do not wait for D = 1000 here. Options posed: back to 0.001 and treat KL regime with actor lr 5e-5 (494's recommendation); or intermediate 0.003; no resume from save (from scratch).

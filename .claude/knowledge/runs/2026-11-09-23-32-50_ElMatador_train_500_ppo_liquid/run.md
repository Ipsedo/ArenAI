# train_500_ppo_liquid

| | |
|---|---|
| machine | ElMatador |
| user | samuel |
| launch | 2026-11-09 23:32:50 |
| commit at launch | `2e594069` — clamp alpha to 0.001 |
| commits ≤ 90 min after | none |
| diff at launch | unknown (identity recorded after the run ended) |
| raw data | `outputs/train_500_ppo_liquid` on ElMatador |

## Established facts

- 2026-11-09: Change (diagnosis from run 499, see its facts; Samuel: « fais juste le premier point » = received ×1, inflicted ×3; coded 09/11, commit `34d2928d`): `HIT_REWARD_SCALE = 3` / `RECEIVED_REWARD_SCALE = 1` replace `DAMAGE_REWARD_SCALE` in `arenai_agent/src/core/train_environment.cpp` (reward and metrics r_hit / r_rcv). Build ok, 2311 tests ok. Rejected option: no more respawn at empty (starves the fire head of data). Refused: cadence constraint (see docs/decisions.md). Why: at ×1 the received flow falls to −0.015/s: waiting 10 s costs −0.15 vs +0.41 gain from aligning, restraint becomes rational without touching exploration nor shell economy. Caution: CSV r_rcv changes scale (÷3), do not compare r / r_rcv of 500 vs 499 raw.
- 2026-11-10: Run (launched 09/11 23:32 by Samuel, `outputs/train_500_ppo_liquid/`, config.json ≡ 499 bit for bit; only change commit `34d2928d`). ~27 k per save, ~80 k/h. Probes side 550 (4 episodes, `arenai_agent_probe` + `analyze_probe.py`) on save_13/16/20/23, save_23 also side 1000.
- 2026-11-10: CSV at 636 k (10/11 07:34, run live). Curriculum: D 500 at 181 k, 700 at 281 k, **plateau 700 from 281 to 429 k**, then 750 → 1000 from 429 k to 595 k (494: 900 at 566 k, never 1000; 499: stuck 500 from 281 k). At equal D (fire / hit / kill):

| D | 494 | 499 | 500 |
|---|---|---|---|
| 450 | 0.67 / 0.093 / 5.7 | 1.27 / 0.089 / 10.9 | 0.73 / 0.134 / 11.6 |
| 550 | 0.49 / 0.088 / 3.9 | n/a | 0.56 / 0.140 / 11.2 |
| 700 | 0.63 / 0.103 / 3.8 | n/a | 0.60 / 0.092 / 4.4 (wave 1) |
| 900 | 0.54 / 0.101 / 5.3 | n/a | 0.40 / 0.140 / 9.6 |
| 1000 | n/a | n/a | 0.58 / 0.095 / 6.8 (wave 2) |

  fire ÷2 vs 499 from 100 k (0.5-0.7/s vs 1.2-1.4) at equal or higher kill ⟹ fire became selective, not rare.
- 2026-11-10: **Two entropy waves**: Hc +0.15 (260 k) → −1.0 (360 k, min −1.35) → +0.23 (540 k) → −0.63 (600-636 k). α pinned 0.001 except alpha_fire (0.01-0.03); canon α never lift (Hc ≥ −1 = target, clamp only plays below target). skip anti-correlated: 0.004 at Hc trough (340 k), 0.35 at Hc peak (500-540 k), 0.05 at 620 k; kl 0.017 → 0.031 → 0.024. D progressed during KL regime (skip 0.35) and stagnated during Hc waves. Probes side 550 (armed p_fire step ≥ 10, aligned <5° vs off-axis ≥ 20°):

| save | k | Hc CSV | σ canon med. (pre-tanh) | σ > 1 | saturated actions ±1 | <5° | p aligned / off-axis | ratio | empty med. | disarmed | hit/shell |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 13 | 351 | −0.93 | 2.05 / 1.71 | 83 % | 48 % | 11.0 % | 0.134 / 0.048 | 2.80 | 7.3 s | 32 % | 10.6 % |
| 16 | 432 | −0.05 | 0.46 / 0.45 | 19 % | 17 % | 15.1 % | 0.370 / 0.149 | 2.48 | 3.5 s | 48 % | 10.1 % |
| 20 | 540 | +0.23 | 0.41 / 0.49 | 5 % | 5 % | 10.7 % | 0.097 / 0.021 | **4.66** | **26.6 s** (74 % empty) | 10 % | **22.3 %** (aligned 35.6 %) |
| 23 | 621 | −0.63 | 1.16 / 0.90 | 54 % | 33 % | 11.1 % | 0.286 / 0.062 | 4.58 | 8.0 s | 37 % | 13.6 % |
| 23 side 1000 | 621 | −0.63 | 1.69 / 1.18 | 69 % | 42 % | 8.9 % | 0.131 / 0.028 | 4.70 | 13.2 s | 23 % | 7.5 % (338 m) |

  References: 499 save_16 ratio 1.94, σ 0.29/0.25, empty 3.0 s, hit/shell 6.8 %; 494 record ratio 3.1, hit/shell 18.5 %, empty 9-11 s. Aligned-shot enrichment 1.6-2.4 (494: 1.3-1.6).
- 2026-11-10: Reading: (a) received ×1 validated: with received flow ÷3, keeping a shell is rational again: reserve emptied at 7-27 s instead of 3 s, aligned/off-axis ratio 2.5 → 4.7, record hit/shell. Dump returns (3.5 s) at save_16 when level p rises to 0.24, then leaves. (b) **Hc waves are not σ collapse but the opposite**: pre-tanh σ explodes per state (p10 0.12 / p90 4.5: large σ when no enemy on axis), tanh makes the law bimodal at ±1, squashed entropy (`gaussian_tanh_entropy`, 1 sample, log-det included) falls. 48 % saturated canon actions at save_13 = bang-bang turret: <5° 15 → 11 %, median angle 13 → 17°, hit/shell 13.6 vs 22 %. KL (Gaussian on pre-tanh u) small when σ large ⟹ skip → 0: system oscillates between two attractors, small σ/KL regime (500-540 k, best perf) and large σ/bang-bang (waves). `LOG_SIGMA_MAX = 2` (σ ≤ 7.4) leaves all room. (c) Clamp/auto α is blind: Hc above target −1 during waves (except trough −1.35), so α stays at floor; same Hc value describes σ → 0 and σ → ∞. (d) Untested hypothesis: with shell restraint, sweeping the turret fully to find a target has positive value, hence large σ off-axis; 499 (σ 0.25-0.32) never showed it. Judge the run at D = 1000 on save_20-like (hit/shell, reserve, ratio), not CSV during a wave; if CSV Hc < −0.5, probe pre-tanh σ before saying collapse.
- 2026-11-10: Decision by Samuel 10/11: « on laisse tourner si c'est le meilleur run jamais fait »: no change, read rest of 500 (wave 2, D 1000). σ ceiling exists: `LOG_SIGMA_MAX = 2` in `arenai_agent/src/networks/constants.h`, shared by `actor.cpp` and `liquid_actor.cpp` (log-σ clamp before exp); lowering it = lever (led to 501).
- 2026-11-10: Reading 09:10, CSV at 749 k, probe save_27 (≈ 729 k), cruise metrics:

| | save_20 (540 k) | save_23 (621 k) | save_27 (729 k) |
|---|---|---|---|
| hit/shell side 550 | 22.3 % | 13.6 % | 14.1 % |
| reserve emptied med. / % empty / % disarmed | 26.6 s / 74 / 10 | 8.0 s / 97 / 37 | 8.6 s / 92 / 41 |
| aligned/off-axis ratio (p) | 4.66 (0.097/0.021) | 4.58 (0.286/0.062) | **2.53** (0.270/0.107) |
| side 1000: hit/shell, emptied, ratio | n/a | 7.5 %, 13.2 s, 4.70 | 7.6 %, 10.2 s, **1.88** (0.159/0.085) |
| σ canon med. / saturated | 0.41 / 5 % | 1.16 / 33 % | 1.25-1.39 / 36-39 % |

  CSV 600-749 k at D = 1000: hit 0.09-0.10 (600-700 k) → 0.115-0.12 (720-749 k), kill 6.4 → 8.2, fire 0.57-0.65 → 0.44-0.57/s, Hc oscillates without resolving (−0.83 at 640 k, −0.19 at 700 k, −0.61 at 740 k), skip 0.05 → 0.15 → 0.10, α canon still 0.001. ⟹ hit/shell and dump flat since save_23 (the trough was not a trough), **ratio ÷2** via off-axis rise (0.02 → 0.11 at 550, 0.03 → 0.085 at 1000); σ still bimodal at save_27, wave 2 lasts 150 k (1st: 120 k). Slight CSV hit rise at 720-749 k comes from rarer fire, not the probe. save_20 stays the best point of the run.

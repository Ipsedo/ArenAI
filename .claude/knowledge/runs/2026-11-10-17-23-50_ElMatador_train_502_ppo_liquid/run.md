# train_502_ppo_liquid

| | |
|---|---|
| machine | ElMatador |
| user | samuel |
| launch | 2026-11-10 17:23:50 |
| commit at launch | `39ae97f9` — update probe for actor (no liquid) |
| commits ≤ 90 min after | none |
| diff at launch | unknown (identity recorded after the run ended) |
| raw data | `outputs/train_502_ppo_liquid` on ElMatador |

## Established facts

- 2026-11-10: Run launched 10/11 17:23, only change vs 501: `continuous_target_entropy` −1 → 0 (diff not committed in `liquid_ppo_hyperparams.h`). Config otherwise identical to 500/501 (σ ceiling 0.7, received ×1 / inflicted ×3, 32 tanks, reserve 10/+2/20, γ 0.995). At 74 k (3 saves): D 350 at 68 k as 500 and 501, hit 10.7 % vs 9.6 %, kill 11.3 vs 9.2-9.9, kl 0.010 vs 0.012, skip 0, alpha_* = 0.001 everywhere; Hc 0.19-0.25 (501: 0.17, 500: 0.12). Why: α update (`entropy_alphas.cpp`) pushes α up only if Hc < target. With Hc ≈ +0.2 and target 0, error positive, α falls and pins at floor 1e-3 exactly as with target −1: change inert in this phase; acts only when Hc passes below 0 (501: 488 k, 500: 278 k; 500 saw α lift at 357 k with target −1 because Hc had dived to −1). Do not judge before ~300-500 k; verdict read at first negative Hc wave: if α rises above 0.001 and holds Hc near 0 without KL regime (skip, kl > 0.05) the change served.
- 2026-11-10: Point at 358 k (21:47, ~81 k/h, save every 20 min): target still inert, Hc never < 0 (plateau 0.3-0.5 since 130 k), alpha_dir/canon pinned 0.001. D 750 (501: 800; 700 reached at 316 k vs 288 k), hit 13.2 % ≈ 501, kill 7.8 vs 8.7, kl 0.019 equal, but skip 0.05-0.07 since 200 k = 2× 501 (0.015-0.03): same symptom as 501's second verdict, steeper slope, not attributable to the change (α identical). alpha_fire rises in both (0.008/0.012). Probe save_13 (351 k, side 550, 2 episodes) vs 501 save_13: σ_canon at ceiling 80 %/90 % (501: 88/70), |μ_canon| 0.40/0.45 (501: 0.59/0.60, better), p_fire aligned 0.24 / off-axis 0.029 (ratio 8.2; 501: 0.405/0.044, 9.2), hit/shell 15.3 % (501: 15.0), reserve empty median 89 s (501: 37.6 s, record of all probes), turret <5° 14.6 % (501: 13), 3.1 % of shots aligned. Less active trigger (global p_fire 0.048) but as selective ⟹ at 358 k 502 ≥ 501 at probe, duplicate at CSV. Raw: scratchpad 2bfa4a36/`probe_502_save13_side550.csv`.
- 2026-11-11: Verdict at 600 k (11/11 00:50, 23 saves). CSV vs 501 at equal index: mean Hc never < 0 (min 0.11 at 593 k); 501 negative from 500 k, −0.25 at 565/579 k; 500: −0.6. skip 502 rises to 0.13 (514-536 k) then **falls** to 0.06; 501: 0.22 monotone. kl 0.015 vs 0.023. D 1000 at 538 k (501: 566 k, 500: 595 k). α per action: alpha_dir_y 0.001 → 0.008 (360 → 515 k) then back to floor at 565 k; alpha_canon_x lifts at 574 k (0.0067 at 600 k). Target 0 does not show in mean Hc of CSV, it reads in alpha_*: one dim passes below 0, α brings it back, the others stay at floor. hit/kill drop after D=1000 in the 3 runs (25 k window: 500 14.7 → 9.2 %, 501 11.9 → 8.1 %, 502 13.5 → 6.8 %, kill 7.5 → 4.6, fire 0.39); deeper in 502.
- 2026-11-11: Probes. save_20 (540 k, D 1000 just reached, side 550) = record equalled: |μ_canon_x| 0.77 (10 % > 1.5), Hc est. canon_x +0.06, p_fire 0.46 / 0.027 (ratio 16.9), hit/shell **22.0 %**, reserve 65 s, aligned shots 5.8 %, turret <5° 14.1 %. save_22 (594 k, 2 episodes):

| | 502 s22 side 550 | 502 s22 side 1000 | 501 s21 (567 k) | 500 s20 (record) | 500 s23 side 1000 |
|---|---|---|---|---|---|
| \|μ_canon_x\| / frac > 1.5 | 1.17 / 33 % | 1.27 / 39 % | 28 % (all dims) | 0.42 / 0 | 0.65 / 2 % |
| Hc est. canon_x / y | −0.51 / 0.11 | −0.56 / 0.18 | n/a | 0.03 / 0.01 | −1.0 / −0.2 |
| p_fire aligned / off-axis (ratio) | 0.64 / 0.063 (10.2) | 0.69 / 0.052 (13.4) | n/a | 0.29 / 0.021 (13.8) | 0.73 / 0.028 (26) |
| hit/shell | 8.8 % | 4.4 % | 12.9 % | 22.3 % | 7.5 % |
| reserve empty median | 12.2 s | 21 s | 16.6 s | 80 s | 40 s |
| turret <5° / aligned shots | 12.8 % / 1.3 % | 12.3 % / 0.6 % | n/a | 10.7 % / 5.1 % | 8.9 % / 1.5 % |

  canon_x drift played out between 540 and 594 k (|μ| 0.77 → 1.17, Hc canon_x +0.06 → −0.51), undoing the record in 55 k. (502 save_13 at 351 k: hit/shell 15.3 %, reserve 89 s, |μ_canon| 0.40.) Why: 501's symptom (μ at bounds, σ at ceiling) reappears but on one dim, canon_x: turret yaws almost full speed in a third of states, aligned shots fall to 1.3 %, hit/shell 15 → 8.8 %, dump 89 → 12 s. alpha_canon_x reacts (from 574 k) as alpha_dir_y had (150 k to return) but correction is slow: proportional α at lr 3e-4, gap must settle before α weighs. CSV hit/kill drop after D=1000 is thus not just the arena transient. 502 > 501 on regulation (no KL regime, D ahead, Hc held), not yet on competence. Raw: scratchpad 111c2dd7/`probe_502_save{20,22}_side*.csv`.
- 2026-11-11: Reading at 688 k (02:00, 77 k/h). CSV 590-688 k: alpha_canon_x 0.0036 → 0.0084 (607 k) → 0.001 (650 k); hit 5.2 % (618 k) → 11.7-12.4 % (650-672 k), kill 2.6 → 9.2. Correction in 75 k (dir_y: 150 k). Relay: alpha_dir_y rises from 623 k (0.005 at 688 k), skip 0.06 → 0.14, kl 0.02. At equal D (+125 k after D=1000): 502 hit 11.5 % / kill 8.4 / skip 0.13 / kl 0.021; 500: 12.0 % / 8.2 / 0.13 / 0.022; only Hc differs (0.19 vs −0.45), mechanical. Probe save_25 (675 k, side 550, 2 ep.) vs 500 save_27 (729 k, same script):

| | 502 s25 | 500 s27 | 502 s20 (540 k) |
|---|---|---|---|
| \|μ_canon_x\| / > 1.5 | 0.64 / 4 % | 0.80 / 9 % | 0.77 / 10 % |
| \|μ_dir_y\| / > 1.5 | 0.76 / 12 % | 0.22 / 0 % | 0.64 / 5 % |
| Hc est. canon_x | +0.24 | −1.03 (free σ) | +0.06 |
| p_fire aligned / off-axis (ratio) | 0.61 / 0.095 (6.4) | 0.68 / 0.107 (6.4) | 0.46 / 0.027 (16.9) |
| hit/shell | 9.7 % | 14.1 % | 22.0 % |
| reserve empty median | 13.2 s | 25.6 s | 65 s |
| turret <5° / aligned shots | 10.1 % / 1.7 % | 11.6 % / 3.3 % | 14.1 % / 5.8 % |

  Why: regulation does what it promises (turret back, Hc canon_x > 0) but competence did not return: off-axis p_fire doubled (0.027 → 0.095), dump 13 s, hit/shell ÷2 vs save_20. What was lost between 540 and 594 k (selective trigger + aim) is not restored by μ returning within bounds; drift migrates to dir_y. At equal D, 502 ≡ 500 on all CSV metrics (seed noise, compare at equal D). Target 0 per action is neither a demonstrated competence gain nor loss; it is a guard against KL regime (skip stays < 0.15) leaving 50-75 k of drift before acting. The 540 k peak (22 %, 65 s) is the attainable level; open question: why it does not hold (one dim drifts to the bounds each time). Probes: scratchpad 111c2dd7.

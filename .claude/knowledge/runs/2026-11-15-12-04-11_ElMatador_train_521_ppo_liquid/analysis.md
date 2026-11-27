# Run 521 (ppo_liquid)

Launched 2026-11-15 12:04, last line 20:07, **stopped** at 752 k (8.06 h, 93 k steps/h, 2.99 M tank-steps/h, 27 saves). Commit 9d843717 + uncommitted `shell_economy.h` (`shells_per_hit` 3 → 2), confirmed by `config.json` and the 12:03 binary (high confidence). Run 522 took over at 20:08.

## What the run tests

520 config + **2 shells refunded per touch instead of 3**, sole change (no code diff beyond comments). Hypothesis: a binding reserve pushes the trigger toward precision rather than volume.
Rule at empty reserve: truncated in-place respawn, **no penalty**. `r − (r_hit + r_rcv + r_aim)` = −6·10⁻⁵ mean over the run (|max| 1.2·10⁻⁴): zero reference to read 522's penalty.

## CSV

D milestones: one step (≈ 30 k) behind 520 over the whole climb, D 1000 at 461 k (520: 429 k). Within seed noise.

| phase | index | D | fire/s | hit | kill | r_hit | Hd |
|---|---|---|---|---|---|---|---|
| climb | 114-207 k | 400-500 | 1.47-1.55 | 0.13 | 23.7 | 9.6-11·10⁻³ | 0.60 |
| trigger tightens | 208-460 k | 550-950 | 0.82-1.19 | 0.15-0.21 | 20.5-24 | 6.1-9.3·10⁻³ | 0.53 → 0.31 |
| volume | 460-540 k | 1000 | 0.61-1.23 | 0.17-0.21 | 20.5-22.3 | 6.1-7.1·10⁻³ | 0.31-0.38 |
| fallback | 540-620 k | 1000 | 0.17-0.48 | 0.22-0.25 | 15.1-19 | 3.6-5.8·10⁻³ | 0.16-0.25 |
| back to volume | 620-700 k | 1000 | 0.81-1.09 | 0.17-0.23 | 19.5-23.5 | 5.0-8.4·10⁻³ | 0.32-0.41 |
| end | 700-752 k | 1000 | 0.82-0.86 | 0.18-0.21 | 21.1-22.1 | 7.8-8.4·10⁻³ | 0.36 |

At equal D vs 520:

| D | fire 520 / 521 | hit 520 / 521 | kill 520 / 521 | r_hit 520 / 521 (·10⁻³) |
|---|---|---|---|---|
| 600 | 1.57 / 1.04 | 0.129 / 0.160 | 24.5 / 21.6 | 9.0 / 7.3 |
| 800 | 0.56 / 0.94 | 0.158 / 0.173 | 20.1 / 22.3 | 6.6 / 7.4 |
| 900 | 0.48 / 0.82 | 0.226 / 0.173 | 18.1 / 22.0 | 5.2 / 7.1 |
| 950 | 0.30 / 0.88 | 0.265 / 0.209 | 15.1 / 20.5 | 3.4 / 6.1 |
| 1000, 300 k after milestone | 0.45-0.64 / 0.82-0.86 | 0.22-0.26 / 0.18-0.21 | 18-21 / 21-22 | 4.6-5.7 / 7.8-8.4 |

Stability: kl 0.03-0.07, skip ≤ 0.08, ev 0.5-0.64 at D 1000, α_fire and α_zoom at floor since 60 k, α_canon_x 0.004-0.013. Hc between −0.13 and +0.15, no wave. Nothing abnormal in optimisation.

## Diagnosis

Established:
- Reward and precision move in opposite directions. At D 1000, corr(hit, r_hit) = −0.79, corr(kill, r_hit) = +0.88 (325 rows). Run's precision max (hit 0.25 at 600-620 k) is its reward min (r_hit 3.6·10⁻³, kill 15); reward max at D 1000 (8.4·10⁻³ at 680-720 k) at hit 0.17-0.18. Drift toward volume = gradient ascent, not instability.
- N = 2 removed 520's trough, not the drift. 520 spent 430-560 k at fire 0.2-0.4/s and r_hit 2-3.6·10⁻³; 521 has only a short fallback (540-620 k) and reaches r_hit 8·10⁻³ at 680 k vs 820 k for 520.
- At equal reward 521 is less precise. r_hit ≈ 8·10⁻³: 520 at hit 0.25-0.26 (840-900 k), 521 at 0.18-0.21 (680-752 k). Probes confirm (save_26 hit/shell 19-21 %, 520 save_28 24.5-25 %).
- Empty reserve costs nothing. Probes save_23/26: 73-85 % of tanks empty, median 1st empty 5-11 s. With the training truncated respawn an emptying refunds 10 shells immediately = free reload.
In the noise: 30 k delay on D milestones; fallback 540-620 k (520 has a longer, earlier one; one seed each side cannot attribute the duration difference to N = 2).
Not settled: end of run — hit rises 0.17 → 0.21 between 700 and 752 k at constant fire (0.82) and held r_hit (8·10⁻³); 50 k, no probe after save_26: cannot say whether precision gain onset or one more oscillation.

## Markers for 522 (empty-reserve penalty −D/1000)

- Order of magnitude of the penalty if 521's behaviour unchanged: at hit/shell 20 % and N = 2 a shot costs 0.6 net shell ≈ 17 shots per 10-shell reserve; at 0.8-1.25 shot/s one emptying per 13-20 armed seconds, 5 to 7.5·10⁻³ per tank-decision. At P = 1 (D 1000) penalty 5-7.5·10⁻³ vs `r` 5.4·10⁻³: 521's end regime would become null or negative. Estimate, not measurement (probe does not respawn).
- Direct CSV reading in 522: residual `r − r_hit − r_rcv − r_aim` vs −6·10⁻⁵; residual/P = emptyings per tank-decision.
- 521 values to beat or watch at equal D:

| D | fire | hit | kill | r_hit (·10⁻³) |
|---|---|---|---|---|
| 250 | 1.61 | 0.077 | 14.4 | 5.4 |
| 400 | 1.56 | 0.136 | 23.9 | 11 |
| 600 | 1.04 | 0.160 | 21.6 | 7.3 |
| 800 | 0.94 | 0.173 | 22.3 | 7.4 |
| 1000 | 0.75 | 0.204 | 20.6 | 6.6 |

- Risk: trigger extinction (479). CSV signs: fire < 0.2/s and kill dropping before D 400, Hd < 0.2 early. P = 0.25-0.4 at these levels is the run's bet.
- Probes: save_1-3 (armed p_fire > 0.1), save_9 (ratio ≥ 2.8, 521's value), then D 1000 (hit/shell > 27 %, 1st empty > 20 s, share of tanks emptying < 80 %).

## Best save

save_16 (432 k, D 950): hit/shell 22 % stochastic, 27-28 % deterministic, 1st empty 9-20 s. Later saves fire more and hit less per shell. Probe details: probes/save_9_16.md.

> Correction of 2026-11-25: the conclusions "better aim", "low canon gone", "dual active, not a pathology" come from the training environment, which respawns tanks at empty reserve. In game Samuel judges s36 bad (reserve empty very fast, canon pointing down). Deterministic, s36 empties its reserve in 5.9/6.2 s. The link between H_offset and α is not established (inconsistency on s36).

# Probe 553 save_36 — D 1000 cruise, vs 510 s28 and 552 s41

## Header

Run `train_553_ppo_liquid`, save_36 = 972 k, D 1000 since 509 k. CSV read to 979 k. `build/` binary of 11-24 23:13, commit `95f1812c` (the run's). Only addition: offset-entropy columns, uncommitted. No inference diff. Sides 550 and 1000, 4 episodes each, stochastic, 32 tanks. GPU shared with training. References: 510 s28 (756 k, D 1000, binary `1e171bbf`, probed 11-13) and 552 s41 (1.11 M). **Comparison with 510 confounded**: reserve max 20 vs 30, different credit and reward, modified environment (`train_environment.cpp` and `arenai_model` change between `1e171bbf` and `95f1812c`). Behaviours can be compared, a gap cannot be attributed to spatial aim. 552 → 553 is clean: the aim action is the only change.

## CSV at D 1000 (whole plateau)

| | 510 | 552 | 553 |
|---|---|---|---|
| n rows | 489 | 1075 | 523 |
| kill | 21.4 | 20.0 | **22.7** |
| fire | 1.01 | 0.40 | 0.77 |
| hit | 0.185 | **0.361** | 0.279 |
| miss (m) | 64.5 | **49.1** | 51.8 |
| r_hit | 0.0077 | 0.0065 | **0.0096** |
| r_rcv | −0.0026 | **−0.0022** | −0.0032 |
| s (steps/episode) | 48.1 | 39.4 | 41.7 |
| ev | 0.47 | **0.65** | 0.57 |
| kl | 0.055 | 0.016 | **0.014** |
| clip | 0.29 | 0.15 | **0.11** |
| skip | 0.050 | **0.044** | 0.062 |
| Hd | 0.35 | 0.14 | 0.24 |

For the record, 549 gives at D 1000: kill 24.6, fire 0.73, hit 0.238. 553's kill is flat between 509 and 979 k.

## Probes

| side 550 / 1000 | 510 s28 | 552 s41 | 553 s36 |
|---|---|---|---|
| tour<5° | 0.288 / 0.227 | 0.174 / 0.185 | **0.346 / 0.313** |
| median turret angle (°) | 12.5 / 14.3 | 13.7 / **10.4** | **10.2** / 11.8 |
| median canon elev. (°) | +0.9 / +1.8 | −4.9 / −4.7 | −0.2 / +0.3 |
| ratio | 5.1 / 10.5 | **16.7 / 62.4** | 4.7 / 9.8 |
| p_fire aligned & < 40 m | 0.66 / 0.66 | 0.50 / 0.52 | **0.90 / 0.83** |
| p_fire off-axis | 0.13 / 0.06 | **0.03 / 0.01** | 0.19 / 0.09 |
| shots/tank-s | 1.59 / 0.86 | 0.73 / 0.33 | 1.84 / 1.20 |
| hit/shell | 0.21 / 0.20 | **0.37 / 0.34** | 0.22 / 0.23 |
| enrichment | 2.16 / 2.68 | **3.04 / 3.21** | 1.84 / 2.20 |
| 1st empty reserve (s) | 6.4 / 8.6 | **17.9 / 31.0** | 5.4 / 5.8 |
| % disarmed | 15.7 / 13.7 | **1.8 / 0.7** | 25.5 / 23.3 |
| p_zoom | 0.46 / 0.37 | 0.93 / 0.90 | 0.93 / 0.92 |

Hit per shell by target distance at the shot (side 550 / 1000):

| distance | 510 s28 | 552 s41 | 553 s36 |
|---|---|---|---|
| < 60 m | 0.56 / **0.65** | **0.70** / 0.63 | 0.49 / 0.54 |
| 60-100 m | 0.38 / 0.39 | **0.48 / 0.50** | 0.39 / 0.45 |
| 100-200 m | 0.17 / 0.24 | **0.24 / 0.32** | 0.19 / 0.27 |
| 200-400 m | 0.07 / 0.04 | **0.11 / 0.13** | 0.11 / 0.08 |
| > 400 m | 0.07 / 0.05 | **0.17 / 0.14** | 0.07 / 0.08 |

## Reading

- Aim: 553 is the best of the three. tour<5° reaches 0.35 / 0.31, and the canon is horizontal. The cells drawn concentrate on rows 3-4, i.e. the horizon (57 % side 550).
- Trigger: 553 fires like 510, not like 552. It presses hard when aligned (0.90) but also off-axis (0.19). Its selectivity (4.7 / 9.8) equals 510's and stays 3 to 6 times under 552's. It empties its reserve in 5-6 s and spends 23-25 % of the time disarmed. 552 does the opposite: fires little and almost always rightly.
- Hit per shell: 553 is under 552 at all distances. The gap is largest at under 60 m (0.49 vs 0.70 side 550). As 553 fires 2.5 times more, it ends with more kills (22.7 vs 20.0) and more damage dealt (r_hit), but also takes the most (r_rcv). The CSV hit drop vs 552 therefore comes from the trigger (off-axis shots, emptied reserve), not from aim. The short-range deficit remains to explain.
- Optimisation: the calmest of the three (kl 0.014, clip 0.11). skip a bit higher (0.06).

## Offset entropy: open inconsistency

On s36, $\bar H_x$ / $\bar H_y$ are −1.89 / −1.63 at side 550 and −1.17 / −0.86 at side 1000, so under the −1 target on the x axis at both sides. Yet α_dx and α_dy **fall** (2.8e-4 and 1.3e-4 at 944 k) while training runs at mean sides of 804. On s8 the side dependence sufficed to reconcile probe and dual; here it no longer suffices for x. Distributions are very asymmetric (mean −0.86, median −0.25 at side 1000). Possible causes, not settled: different states between the probe (step ≥ 10, continuous inference) and training (30-step unrolled chunks), or tail of the one-draw estimator. Quick measure to settle: replay the probe without the step ≥ 10 filter, with recurrent state reset every 30 steps, then compare to a CSV-only export of $\bar H$ trainer-side (the latter requires a G2 validation).

## Conclusions

- Established (552 → 553, isolated change): spatial aim fixes the canon and improves aim (tour<5° ×2, canon at 0°), and the curriculum goes faster (D 1000 at 509 k vs 699 k).
- Established: the trigger lost 552's selectivity and returns to a 510 profile (dump, ratio ~5-10). The CSV shows it too (fire 0.77 vs 0.40, hit 0.28 vs 0.36). One seed: cannot say if aim-action effect or seed.
- Open: hit at under 60 m (0.49-0.54 vs 0.63-0.70 for 552), and the inconsistency between measured entropy and α.

# train_520_ppo_liquid — run analysis (CSV + probes)

Host: ElMatador. Launched 2026-11-14 23:15 (config.json mtime), reviewed 2026-11-15 ~11:00 at ~1 M (save_37), ~89 k steps/h (2.84 M tank-steps/h, ≈ 510/514/515).
Effective commit **0e5093a1** (committed 23:59, 44 min after launch). config.json = defaults of `liquid_ppo_hyperparams.h` at 0e5093a1 (gamma 0.998, discrete_target_proba 0.01, CNN 4 blocks 16→128, GN 2/4/8/16, micro-batch 512). Process ran on a deleted executable (compiled ≤ 23:15): good confidence, not certain.

## What the run tests

520 = 515 + `discrete_target_proba` 0.05 → 0.01 (entropy target of trigger and zoom: H(0.05) = 0.199 → H(0.01) = 0.056). Code between 515 and 520 changes only the `ev` computation (per optimizer batch instead of per GPU micro-batch): metric only.
515 differs from 514 by gamma 0.998 only ⟹ 514 = longest reference of the same lineage (economy 10/3/30, same CNN, same rule). 510 has economy 10/2/20: indicative only.

Hypothesis (memory note regulateur-entropie-gachette-plafonne-selectivite): in 510 and 514 α_fire rises ×20 at 400-600 k because a selective trigger has mean entropy under H(0.05), and the trigger loosens (p_off ×4-6 in 510 at save_20).

## CSV

D milestones identical per index in 510/514/515/520 (D 1000 at 429 k): non-discriminating.

| slice (end) | run | hit | kill | fire/s | Hd | α_fire |
|---|---|---|---|---|---|---|
| ~430 k | 510 | 0.21 | 19.8 | 0.88 | 0.38 | 7.9e-4 |
| | 514 | 0.17-0.19 | 21.6 | 1.05 | 0.44 | 1.5e-4 |
| | **520** | **0.24** | 16.3 | 0.41 | 0.29 | 1e-4 |
| ~520-530 k | 510 | 0.20 | 18.8 | 0.88 | 0.35 | 5.7e-3 |
| | 514 | 0.19 | 21.6 | 0.97 | 0.38 | 2.1e-3 |
| | **520** | **0.23** | 14.8 | 0.34 | 0.16 | 1e-4 |
| ~600-630 k | 510 | 0.18 | 21.2 | 0.94 | 0.34 | 9.5e-3 |
| | 514 | 0.17 | 21.8 | 0.94 | 0.35 | 9.0e-3 |
| | **520** | **0.24** | 18.0 | 0.49 | 0.29 | 1e-4 |
| ~780-870 k | 510 | 0.18-0.19 | 21.6-22.6 | 1.03 | 0.31-0.35 | 4.8-8.9e-3 |
| | 514 | 0.18 | 22.2 | 1.05 | 0.30 | 8.1e-3 |
| | **520** | **0.23-0.25** | 20.5-22.0 | 0.70-0.75 | 0.23 | 1e-4 |

- **α_fire stays at floor (1e-4) over the whole run**: the targeted mechanism is removed; the 400-600 k rise of 510/514 does not occur.
- **CSV hit 23-25 % at D = 1000 over 460 k steps** vs 17-21 % for 510 and 514: gap of 4-6 points held over the whole cruise, beyond the seed noise seen between 510/514.
- **Cadence halved** (0.3-0.75 shot/s vs 0.9-1.2) and **kill dip** at 430-530 k (11-16 vs 19-22), back to 22-23 at 860-880 k (≈ 510/514).
- Slice 426-470 k (arrival at D = 1000): kl 0.1-0.43, clip 0.4-0.7, skip ≈ 0.002 (rest) and skip_canon 0.30-0.43; fire 0.06-0.4/s, kill 11-12, s 55-74 s. Withdrawal phase: few shots, long survival.
- Critic: ev 0.7-0.9 at 430-460 k, then isolated negative spikes (−0.4 at 610 k, −2.4 at 612 k) without lasting effect; `v` rises 0.005 → 0.05-0.08 from 600 k, like the cadence recovery (more touches = more variable targets).
- 760-880 k: Hc −0.2 to −0.58 (min 790-800 k), rest-group skip 0.04-0.07 (max 0.067 at 812 k); α_dir_y leaves the floor (2e-4 → 8.5e-3) and α_canon_x 0.011-0.014. Continuous regulator active, nothing abnormal yet, to follow.

## Diagnosis

Established (structural, imposed by code): with target H(0.01) the trigger regulator no longer raises α_fire; the entropy push on the trigger disappears during the whole cruise.
Established (CSV, one seed but wide lasting gap): hit per shot better than 510/514 at equal D over the whole cruise, with half the shots.
Not settled by CSV: whether the higher CSV hit comes from a more selective trigger (low p_off) or a trigger that fires less everywhere (CSV blind to selectivity and dump ⟹ probe); whether low cadence costs kills (dip 430-530 k, caught up at 860 k); the negative Hc drift and rising skip at 780-880 k (continuous σ collapsing?) ⟹ check σx, σy at probe.

## The reward measures touches per second, not precision

Over the whole run and in 510/514, r_rcv = −r_hit / 3 (±1 %): each inflicted ×3 touch is a received ×1 touch by another agent ⟹ r = ⅔ · r_hit ∝ touches per tank-step. The reward sees neither hit/shell nor the reserve, only the touches they allow. Selectivity has value for the reward only if the reserve constrains cadence.

r per 50 k slice (×1e-3):

| idx | 300 | 400 | 450 | 500 | 600 | 700 | 800 | 850 | 900 | 950 | 1000 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 514 | 5.0 | 4.9 | 4.6 | 5.0 | 5.5 | 5.8 | 5.9 | 6.2 | 6.2 | 5.5 | 5.1 |
| 520 | 3.9 | **1.9** | **1.7** | 2.2 | 4.1 | 3.4 | 5.0 | 5.3 | 5.5 | 5.1 | 4.7 |

520 lost up to 60 % of 514's reward during its extinct-trigger phase (400-550 k) and rejoined only from 850 k. By reward, 520 was never better than 514.

## Probe chronology (probes/timeline.md)

- save_9 (243 k): total dump (p_off 0.6, 42-64 % disarmed steps).
- save_16 (432 k): trigger nearly extinct (0.13-0.29 shot/s, touches/tank-s ÷3 vs 514).
- save_20 → save_37: cadence ×4.5, hit/shell 30 → 24 %, disarmed steps 1 % → 5-11 %, touches/tank-s rejoin 514 (0.32/0.14 vs 0.27-0.33/0.12-0.17).
- Flat aim since save_16 (tour<5° 27-33 %). The h<5° drop (44 → 35 %) comes from the trigger threshold lowering, not the turret.
- save_36: isolated site relapse (−2.7°), as 514 save_21.

## Overall diagnosis

Established: target H(0.01) removes the trigger regulator (α_fire at floor all run); without regulator the trigger oscillates strongly (total dump → extinction → recovery) whereas 514 converges smoothly; the trigger climbs back by itself to 514's cadence following the reward gradient (touches/s); aim (precision/angle frontier) is the same as 510/514 since ~430 k: only the chosen point on the cadence ↔ precision curve changes.
In the noise / not settled: will 520 stop before 514's point (for now: same touches/s with 15-25 % fewer shots and 2-3× fewer disarmed steps); site relapse of save_36 (PPO on canon group, skip_canon 0.08 → 0.17 at 920-980 k, or noise).

## Scenarios for 1 → 1.15 M (save_42)

| scenario | expected signature (probe 550/1000, CSV) | probability | reading |
|---|---|---|---|
| A. Convergence to 514 | shots/s ≥ 1.4/0.8, hit/shell 19-22 %, disarmed 10-13 %, p_off ≥ 0.10; CSV fire ≥ 1.1, hit ≤ 0.20 | most probable | target 0.01 changes path not arrival: equilibrium set by reward + economy; entropy regulation no longer the lever |
| B. Stop at intermediate point | touches/s ≈ 514, shots/s ~1.3/0.6, disarmed ≤ 6 %, p_off ≤ 0.07 | plausible: reward flat between the two points | 520 gets same reward with fewer shells: better for the game; keep target 0.01 |
| C. Dump overshoot | disarmed > 15 %, p_off > 0.15 on 2 saves | possible (undamped trigger) | unregulated trigger instability; keep save_32/37 |
| D. Canon instability | site bias > 1.5° on 2 saves, skip_canon > 0.2 | low-moderate | canon group drifts in jerks; tie to aim credit without critic (λ 0 on Φ) |
| E. Aim progresses | tour<5° > 34 %, h<5° rises at equal cadence, touches/s > 0.35/0.17 | low: no run showed it after ~430 k | aim ceiling not fixed |

Tie-break per save: touches/tank-s, %disarmed, p_off, site bias (session script `verdict.py`). A and B distinguishable only over 2-3 saves: 514 already oscillated save to save.

## Implications

- For the game: save_32 or save_37 (same touches/s as 514, half the disarmed steps, better hit/shell); save_32 probed deterministic.
- For training: if A confirmed, trigger entropy regulation is not a lever on the equilibrium; only reward (touches/s) and shell economy decide where the trigger stops on the cadence ↔ precision curve. The aim ceiling (tour<5° ~30 %), common to all runs since 510, is the fundamental limit.

## Probes (probes/save_20_32.md)

save_20 / save_32: dump gone (disarmed 1-3 % vs 10-37 %, empty reserve median 22-63 s), hit/shell 24.5-27.6 % (514: 18-22) at equal aim (tour<5° 25-29 %), p_off 0.017-0.061, p_al 0.26-0.46. Deterministic (save_32): 0.81/0.36 shot/s and hit/shell 28-29.5 % (514 s34 det: 1.25/0.41, 22-23 %): trigger not mute, fires less and better. Cadence ×2 from save_20 to save_32 although α_fire at floor. σ canon healthy; movement freezes (μ dir_y ≈ −0.95, saturated 10-13 %) ⟹ negative Hc.

## Options and markers

- save_32 usable in game (deterministic probed); in game still check the frozen movement (μ dir_y ≈ −0.9).
- Follow Hc and rest-group skip beyond 900 k: Hc < −0.5 lasting with skip > 0.1 = known σ collapse / KL regime.

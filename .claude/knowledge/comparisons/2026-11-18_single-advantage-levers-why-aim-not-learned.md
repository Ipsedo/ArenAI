# Single advantage + event reward + flattened CNN: why aim is not learned

Runs: 528 (s1, s3, s5, s9, s14), 529 (s9, s27), pilots 530, 531, 532.
Date: 2026-11-18. Checkpoints probed at side 550 except the first saves (side = the run's D). Collection: `arenai_agent_sigma_gradients` (run's binary, no code modified), 12 episodes per configuration at λ 0.95 and λ 0, 6 episodes at γ 0 / λ 0 (in that case the advantage equals −V up to an affine transformation: the only way to read V without touching the tool). Scripts in `outputs/comparisons/2026-11-18_leviers_avantage_unique/` (`vsens.py`, `total_signal.py`, `credit.py`, `decode.py`, `vkernel.py`, `gamma_share.py`, `measure_pilot.sh`, `vhead.py`, `vdistill.py`, `vcnn.py`, `vshort.py`, `vlag.py`, `vprofile.py`, `early.sh`, `collect.sh`, `cfg/`), outputs in `outputs/comparisons/2026-11-18_leviers_avantage_unique/data/`.

## 1. The critic does not see the aiming gesture

Variation of V when the turret approaches the target (armed tank, no shot, no event), in standard deviations of V per rollout, per 10° of approach (`vsens.py`):

| checkpoint | band | run's V | value fitted on geometry |
|---|---|---|---|
| 529 s9 | < 50 m | 0.000 to +0.007 | +0.09 to +0.12 |
| 529 s9 | 50-100 m | −0.002 to +0.004 | +0.06 to +0.10 |
| 528 s9 | < 50 m | +0.02 to +0.04 | +0.28 to +0.30 |
| 528 s9 | 50-100 m | +0.005 to +0.009 | +0.15 to +0.16 |
| 529 s27 | < 50 m | +0.004 to +0.012 | +0.16 to +0.18 |

The "geometric value" is an MLP fitted offline on the Monte-Carlo return from the nearest enemy's geometry (out-of-fold per episode). It is a measuring standard, not a proposal.

Consequence: ρ(run advantage, corrective gesture) = −0.002 to +0.005 at λ 0, 0.005 to 0.010 at λ 0.95. Share of the advantage exploitable by a gradient, per head (`total_signal.py`, standard error 0.002):

| checkpoint | dir_x | dir_y | canon_x | canon_y |
|---|---|---|---|---|
| 528 s1 | −0.006 | 0.000 | +0.012 | +0.010 |
| 528 s5 | 0.000 | +0.012 | +0.008 | +0.007 |
| 528 s9 | −0.002 | +0.002 | +0.013 | +0.004 |
| 529 s9 | −0.005 | 0.000 | +0.004 | +0.002 |
| 529 s27 | +0.003 | −0.001 | +0.004 | 0.000 |

## 2. The signal exists in the events

Same measurement with the geometric standard as critic (`credit.py`), ρ with the corrective gesture:

| checkpoint | yaw < 100 m, λ 0.3 | elevation, λ 0.3-0.6 | at λ 0.95 | run advantage |
|---|---|---|---|---|
| 528 s1 | 0.030 | 0.067 | 0.027 / 0.044 | 0.021 / 0.024 |
| 528 s5 | 0.055 | 0.054 | 0.019 / 0.030 | 0.010 / 0.033 |
| 528 s9 | 0.076 | 0.116 | 0.020 / 0.032 | 0.008 / 0.011 |
| 529 s9 | 0.025 | 0.037 | 0.017 / 0.007 | 0.010 / 0.007 |
| 529 s27 | 0.020 | 0.021 | 0.005 / 0.009 | 0.005 / 0.012 |

528 s9 (trigger open at 50-100 m) reaches the level of the former separate credit (one-step Φ: 0.08-0.10). The signal drops when the trigger closes (529 s27).

## 3. Where information is lost in the critic

**Not in the CNN.** (Corrected in §7: this showed only that the critic encodes no less than the actor, on an easier contrast.) Linear probe on CNN features (`decode.py`, 32 k images of 529 s9, fitted on one episode, evaluated on the other), AUC "aligned < 5° versus 10-39°", enemy < 100 m:

| CNN | block 4 | block 5 | block 6 | output 1×2 |
|---|---|---|---|---|
| critic | 0.784 | 0.787 | 0.783 | 0.751 |
| actor | 0.773 | 0.753 | 0.759 | 0.737 |
| untrained | 0.672 | 0.649 | 0.623 | 0.578 |

The critic encodes alignment as much as the actor, whose trigger is selective. Modest level in both (0.75); linear-probe regression of |yaw| is unstable (negative R²), inconclusive.

**In the temporal response.** Distributed-lag regression of V on alignment (target < 10° and < 100 m) at lags 0 to 4 s (`vkernel.py`):

| checkpoint | run's V: weight at lag 0 | share at lags 0-1 | standard: weight at lag 0 | share 0-1 |
|---|---|---|---|---|
| 529 s9 | +0.12 | 18 % | +0.91 | 63 % |
| 528 s9 | +0.13 | 31 % | +1.12 | 59 % |
| 529 s27 | +0.11 | 14 % | +0.38 | 35 % |

The critic responds to present alignment 3 to 9 times too little and spreads the rest uniformly over 4 s (weight at 3-4 s as large as at 0 s in 529). It knows "an enemy has been in front for a while", not "the turret just aligned".

**Clues on the cause** (correlational, not settled):

- Critic target: at γ 0.998 (horizon 50 s) angles explain 6 to 12 % of the return variance; 12-17 % at γ 0.99; 25 % at γ 0.95 (`gamma_share.py`). Most of the target is long-horizon noise.
- Liquid cell weights: critic input norms 5 times the actor's (1.4-2.5 versus 0.3-0.46), rising from s9 to s27; critic τ slowing (p50 0.47 → 0.65 s, p90 1.88 → 2.66 s) while the actor's stays at init (0.42 / 1.4 s). In the actor vision enters mostly through fast neurons (correlation −0.72 with log τ); in the critic this structure has disappeared (−0.1).
- The critic does 8 epochs at lr 3e-4 (actor: 4 epochs at 1e-4) on these targets. CSV `ev` is measured on rows being fitted, against a target that contains V: 0.75-0.84 whereas out-of-sample corr(V, Monte-Carlo return) = 0.26 (R² 0.07; standard: 0.33).

## 4. Other measurements

- Actor latency: corr(yaw, setpoint) maximal at lag +1 decision in 528/529, 0 in 520/522; same order of gain. The turret tracks, but the mean setpoint vanishes with the target at +5-8° off axis (529 s14/s27) and commands in the cone are 5-7° per decision (520/522: 0.8-1.8°).
- Return variance: hits 70-80 %, kills 10 %, damage received 6-8 %, death 4-6 %, empty reserve 0.4-6 %.

## 5. Limits

- Everything is measured on frozen policies; nothing tested in training.
- The geometric standard knows only the nearest enemy, neither reserve nor life.
- The critic of 520/522 was not measured (old binaries): unknown whether it was as slow.
- Proprioception not being exported, the critic was not replayed offline: the exact cause of the slow response (liquid cell, target horizon, optimisation regime) remained to settle.

## 6. Training pilots (2026-11-18)

Two runs starting from the 529 config, one single difference each (verified by diff of `config.json`), critic measured at saves 3 and 5 (`measure_pilot.sh`, outputs `data/pilote_53*.txt`, `data/reference_529_s3_s5.txt`). Slope of V per 10° of approach, enemy < 50 m:

| run | difference | s3: run's V | s3: standard | s5: run's V | s5: standard | share lags 0-0.1 s (s3 / s5) |
|---|---|---|---|---|---|---|
| 529 | — | +0.008 to +0.024 | 0.15 | −0.001 to +0.026 | 0.13-0.17 | 31 % / 25 % |
| 530 | critic epochs 2, lr 1e-4 | +0.005 to +0.036 | 0.16-0.18 | −0.001 to +0.007 | 0.11-0.13 | 16 % / 12 % |
| 531 | γ 0.99 | −0.005 to +0.004 | 0.11-0.14 | −0.003 to +0.004 | 0.07-0.14 | null response / 27 % |

Weight of V on present alignment: 529 0.18 / 0.12, 530 0.12 / 0.11, 531 0.01 / 0.07 (standard 0.2-0.7). Neither pilot straightens the slope.

- 530: critic input weights stayed small (0.19 versus 1.12), τ at init, CSV ≡ 529 up to 190 k ⟹ the critic's optimisation regime is not the cause.
- 531: the short horizon does not help the critic and degrades the actor: kill 9.5 → 1.8 at 100 k (529: 18), D stayed at 250 at 140 k (529: 300, 530: 350), Hc −0.37 from 100 k ⟹ the target horizon is not the cause.

Remaining, among the three causes of section 5: the image → V path itself: V reads the image only through the liquid cell. Untested at that point.

## 7. Offline measurement: the liquid cell is not the lock (2026-11-18)

`arenai_agent_sigma_gradients --critic_inputs <file>` (option added, uncommitted) exports at each step the proprioception and the critic's CNN output. 24 episodes of 529 s9 at side 550 (821 k living rows), 18 for fitting, 6 never seen for measurement. Value heads retrained on these frozen inputs against the Monte-Carlo return (`vhead.py`, 40 passes, 2 seeds):

| value head | corr(V, return) | slope of V per 10° (< 50 m) | weight at lag 0 |
|---|---|---|---|
| geometric standard | 0.30 | +0.14 to +0.15 | 0.74 |
| run's critic, replayed in Python | 0.27 | +0.004 to +0.010 | 0.13 |
| memoryless (MLP on the step's input) | 0.27 | −0.008 to +0.006 | 0.16 |
| liquid cell reinitialised | 0.13-0.19 | +0.002 to +0.015 | 0.13-0.15 |
| cell + direct input → V path | 0.15-0.19 | +0.003 to +0.013 | 0.13-0.15 |
| run's critic fine-tuned on the return | 0.17 | 0.000 to +0.004 | 0.13-0.14 |

A head with no memory at all is as blind as the run's critic: the cell is not at fault. Recurrent heads overfit (576 sequences: fit loss 0.25-0.38, test worse than the run's critic), so their row is little informative.

What the critic's CNN output contains (`vdistill.py`, MLP 2×512 without memory, test):

| input | AUC aligned < 5° vs 5-40° (< 100 m) | R² of standard distillation | slope of distilled V |
|---|---|---|---|
| CNN features + proprio | 0.66 | 0.12 | +0.02 |
| CNN features alone | 0.67 | −0.03 | +0.02 to +0.04 |
| proprio alone | 0.57 | −0.02 | +0.01 to +0.03 |

Even given the standard as target, the critic's CNN 1×2×128 output allows reconstructing only 12 % of it, and the slope stays 4 to 7 times below the standard. The lock is thus upstream of the cell: alignment is almost absent from what the critic's CNN transmits. Section 3 ("not in the CNN") is to be corrected: it showed only that the critic encodes no less than the actor, on an easier contrast (< 5° versus 10-39°, AUC 0.75).

Limit: frozen CNN. This measurement does not say whether the CNN can learn alignment when trained on a target that depends on it, nor why the run's value loss does not lead it there.

## 8. CNN trained end to end offline: it is the target, not the architecture (2026-11-18)

`--frames` added to `arenai_agent_sigma_gradients` (uncommitted). 24 episodes of 529 s9, one image in three: 272 k images (206 k fit, 66 k test on 6 never-seen episodes). The run's CNN (7 blocks, output 1×2×128) + proprio + memoryless head, 8 passes (`vcnn.py`, outputs `data/vcnn_529_s9_*.txt`). Test measurements, slope at 0.3 and 0.6 s:

| target | starting CNN | slope of V per 10° | weight at lag 0 | AUC aligned < 5° | fit / test loss |
|---|---|---|---|---|---|
| geometric standard (reference) | — | +0.11 to +0.13 | 0.84 | — | — |
| run's critic, replayed | — | +0.01 to +0.03 | 0.10 | 0.66 (section 7) | — |
| standard | reinitialised | +0.05 | 0.38 | 0.74 | 0.21 / 0.85 |
| standard | run's CNN | +0.06 to +0.07 | 0.43 | 0.75 | 0.24 / 0.81 |
| Monte-Carlo return | reinitialised | +0.01 | 0.13 | 0.64 | 0.13 / 0.95 |
| Monte-Carlo return | run's CNN | −0.03 to 0.00 | 0.13 | 0.70 | 0.14 / 0.88 |

- With a target that depends on alignment, this CNN learns to read it from the first pass (slope +0.05 to +0.06) and reaches half the reference, limited by overfitting (206 k images). The architecture is thus not the lock.
- With the return as target, same images, same network: null slope, the network memorises return noise (fit loss 0.13, test 0.95) and does not extract alignment. That is what the run's critic does.

Conclusion: the critic does not learn aim because the part of the return explained by alignment (6-12 % of variance at γ 0.998) is too weak against noise to shape the CNN, not because the CNN or the cell are incapable.

Limits: offline on a fixed set of 206 k images (the run sees millions of fresh samples, but its critic stays at +0.01-0.03 after 7.8 M tank-steps); one checkpoint; one seed per configuration; the standard is a noiseless target, the most favourable case.

## 9. Short-horizon value head, offline (2026-11-18)

Same protocol as section 8 on a new collection (24 episodes of 529 s9, 272 k images), CNN reinitialised, 6 passes, one head per γ on the same CNN (`vcnn.py`, config `<γ>+<γ>:scratch:7`, outputs `data/vcnn_529_s9_g*.txt`). On this set: standard fitted on γ 0.998 slope +0.09 to +0.11, run's critic +0.011. Values at the last pass, slope at 0.3 and 0.6 s:

| heads trained | slope of short head | slope of γ 0.998 head | AUC aligned < 5° read in CNN | fit / test loss |
|---|---|---|---|---|
| γ 0.998 alone | — | +0.01 / −0.00 | 0.67 | 0.19 / 1.07 |
| γ 0.99 alone | +0.05 / +0.04 | — | 0.72 | 0.25 / 1.00 |
| γ 0.95 alone | +0.05 / +0.05 | — | 0.77 | 0.35 / 0.83 |
| γ 0.9 alone | +0.06 / +0.05 | — | 0.78 | 0.38 / 0.82 |
| γ 0.998 + 0.95 | +0.05 / +0.04 | +0.03 / +0.02 | 0.75 | — |
| γ 0.998 + 0.9 | +0.05 / +0.05 | +0.02 / +0.03 | 0.74 | 0.31 / 0.93 |

- Alignment read in the CNN grows monotonically as the horizon shortens (0.67 → 0.78); at γ 0.9-0.95 the CNN encodes it as much as with the geometric target (0.75), without geometry.
- The short head reaches +0.05 per 10°, a quarter of its own standard (+0.18 to +0.20): the signal is learned but stays attenuated. Weight at lag 0: 0.20-0.24 (standard 1.04).
- With two heads, the γ 0.998 head goes from ~0 to +0.02-0.03 (2 to 3 times the run's critic), far from its standard (+0.09 to +0.11): the CNN is better shaped, the long head benefits only in part.

Limits: one seed, slope noisy by ±0.02 from one pass to the next, overfitting on 206 k images, memoryless head, Monte-Carlo target (the run uses TD(λ) with the liquid cell). Offline γ 0.99 gives a slope, whereas pilot 531 (γ 0.99 in training) gave none at saves 3 and 5: an offline result does not guarantee the effect in training.

## 10. Pilot 532: short head γ 0.9 in the run — negative at save_5 (2026-11-18)

532 = 529 + a second value head at γ 0.9 (own TD(λ) target and normaliser, losses summed, advantage unchanged on the γ 0.998 head). Measured at save_3 and save_5 (D = 250).

| slope of V per 10° (< 50 m, k = 3 and 5) | save_3 | save_5 |
|---|---|---|
| 529, single head | +0.008 to +0.024 | −0.001 to +0.026 |
| 532, long head | +0.013 to +0.037 | +0.013 to +0.015 |
| geometric standard (532) | +0.14 to +0.24 | +0.20 |

Weight at lag 0 of the long head: 0.03 then 0.01 (standard 0.55 then 0.92).

The two heads of 532 s5 replayed on 24 episodes (`vshort.py`, `data/vshort_53{2,9}_s5.txt`):

| at save_5 | 529 (D 300) | 532 (D 250) |
|---|---|---|
| long head, slope /10° | +0.002 to +0.029 | 0.000 to +0.012 |
| short head, slope /10° | — | +0.003 to +0.016 |
| short head, weight at lag 0 | — | 0.00 |
| standard fitted on the γ 0.9 return, slope | — | +0.14 to +0.16 |
| critic CNN output → aligned < 5° (AUC) | 0.60-0.63 | 0.55 |
| standard distillation from CNN + proprio (R²) | 0.17 | 0.02 |

The short head itself does not see alignment, and 532's critic CNN encodes it rather less than 529's at the same age. The expected mechanism (section 9: +0.05/10° on the short head offline) did not take in the run. The short head follows its γ 0.9 target less well than the long head (corr 0.45 versus 0.50): it learned nothing of its own.

CSV: trough from 40 k to 70 k (kill 6 versus 9-13), recovery at 138 k (kill 20.8, hit 12.7 %), but D still at 250 at 142 k (529: 300 at 114 k; 530: 300 at 67 k).

Reservations: one seed, different D (250 versus 300), only 135 k steps; the offline test was on a D 550 policy (529 s9), reinitialised CNN, Monte-Carlo target rather than normalised TD(λ). The `ValueTargets` rework (permanent head in ppo / ppo_liquid / ppg) is coded, uncommitted; no unit test covers `ValueTargets::compute`.

# Diagnosis — why the trigger closes (runs 508 to 543)

Runs: 508-543 (focus 510, 514, 520, 522, 524, 525, 528, 529, 539, 541, 542, 543).
Written 2026-11-20 after the probe of 543 save_13. Sources: all stochastic probes at side 550 of runs 508-543 (`<run>/probes/save_*_side550.json`, 4 episodes, 32 tanks) and the `metrics.csv`. Complements `2026-11-20_summary-since-v1.3.2.md`.

## 1. The three run families

| family | runs | signal driving the cannon |
|---|---|---|
| separate credit on Φ | 510, 512-515, 520-522 | geometric aim potential, one step, no critic; exists whether or not the agent fires |
| pure event | 524, 525, 528, 529, 539 | GAE on hits / received / empty-reserve penalty; exists only if a shell leaves |
| credit on H | 541, 542, 543 | one-step variation of the shot value learned from shots fired |

## 2. Level reached (CSV)

| run | family | final index | D max | kill at D max | last 100 rows: D / kill / fire |
|---|---|---|---|---|---|
| 510 | Φ | 867 k | 1000 | 21.4 | 1000 / 21.6 / 1.02 |
| 514 | Φ | 1 052 k | 1000 | 22.4 | 1000 / 22.2 / 0.96 |
| 520 | Φ | 1 136 k | 1000 | 20.1 | 1000 / 22.4 / 0.52 |
| 522 | Φ | 1 902 k | 1000 | 19.2 | 1000 / 20.4 / 0.30 |
| 524 | events | 699 k | 1000 | 8.3 | 1000 / 8.5 / 0.21 |
| 528 | events | 441 k | 750 | 14.9 | 750 / 14.1 / 0.35 |
| 529 | events | 794 k | 700 | 15.8 | 627 / 9.9 / 0.23 |
| 539 | events | 525 k | 800 | 16.3 | 786 / 18.1 / 0.39 |
| 541 | H | 860 k | 900 | 10.8 | 623 / 3.0 / 0.12 |
| 542 | H | 587 k | 950 | 14.2 | 909 / 6.7 / 0.07 |
| 543 | H, constant penalty | 384 k | 750 | 12.5 | 734 / 13.0 / 0.39 |

All eight Φ-credit runs reach D 1000 (512, 513, 521 included, kill 17-21); none of the other eight holds kill ≥ 17 at D 1000.

## 3. What the probes establish

Probes of save_9 and later, side 550.

| | Φ (45 probes) | events (17) | H (17) |
|---|---|---|---|
| turret < 5° from a target, median | 28 % | 11 % | 13 % |
| hits per tank-second, median | 0.20 | 0.10 | 0.12 |
| hit/shell, median | 0.21 | 0.20 | 0.18 |
| correlation fire rate / turret < 5° across saves | −0.49 | +0.57 | +0.88 |
| turret < 5° when trigger closed (armed p_fire ≤ 0.035) | 31 – 45 % (5 probes) | 6 – 12 % (5) | 2 – 13 % (6) |

### 3.1 The trigger sorts as well without Φ

Logit gap between p_fire aligned-near and off-axis, at maturity:

- Φ: 2.2 to 3.2 (510, 514, 520, 522).
- events: 3.0 to 4.3 (524, 525, 528, 529, 539).
- H: 3.0 to 4.2 (541, 542, 543).

The "now is the time to fire" detector is at least as good in the failing runs. The trigger is not the failing piece.

### 3.2 Every family goes through a closure

Between 240 and 430 k the empty-reserve penalty lowers the common level of the trigger, in all three families:

- 522: armed p_fire 0.055 (s9) → 0.025 (s10) → 0.017 (s16), then 0.03-0.06 from s24 to s68.
- 520: 0.030 at s16, then 0.06-0.19.
- 524 / 525: 0.022 / 0.026 at s9, stay there.
- 541: 0.016 → 0.003 (s14-s16), 0.18 at s20, 0.015 at s24.
- 543: 0.009 at s13.

The release (522) is itself a run with a very closed trigger: 0.3-0.5 shot per tank-second, the rate of the "extinguished" phases of 524, 528 and 529.

### 3.3 What differs during the closure: the turret

With the trigger closed, the turret of the Φ runs keeps aiming (31-45 % of the time at < 5°), because its signal does not depend on firing. The rare shells then leave well placed: hit/shell 0.36-0.46 in 522. Reward stays positive and the level stabilises.

In the runs without Φ the turret signal exists only through shells fired. Trigger closed, the turret is at < 5° only 2 to 13 % of the time. The rare shells leave badly (543 s13: 51 % at more than 20°), nothing tells the trigger any more that firing aligned pays, and the level keeps dropping. The across-save correlation between fire rate and aim is positive in these runs (+0.57 and +0.88) and negative in the Φ runs.

541 already showed it in both its drops: the turret lets go one save before the aligned trigger.

### 3.4 H worsens the drop

With trigger closed, the turret of the H runs falls lower (2-5 % in 541 s14-s16 and 543 s13) than that of the pure event runs (6-12 %), and so does the trigger level (0.003-0.016 versus 0.012-0.028). H is labelled only on shots fired and its advantage is normalised: without shots the cannon receives a unit-variance signal with no content. Mechanism inferred, not tested.

### 3.5 The empty-reserve penalty is not the cause

543 (constant penalty 0.5) closes at D 700-750 and 351 k, earlier than 541 (−D/1000, D 850, 378 k). The indexing on D, suspected in the summary, is ruled out.

## 4. Diagnosis

Two skills that condition each other share a single signal:

1. aiming pays only if a shell leaves;
2. firing pays only if the turret is aligned.

The empty-reserve penalty pushes the trigger level down, as intended. With an aim signal independent of firing, the descent stops on a stable point: turret aligned, few shells, well placed (510-522). Without it, the descent cuts the aim signal, the turret drifts, the remaining shells lose their value, and the loop closes on extinction (524, 525, 529, 543) or on a dump / extinction oscillation (541).

Established: sections 2 and 3.1 to 3.3, 3.5.
Inferred: causal direction turret → trigger (temporal order in 541, correlations); mechanism of 3.4.
Not tested by intervention: that an aim signal independent of firing suffices on the current tree (flattened CNN + up-to-date image). The other PC's witness (flattened CNN, Φ credit, without up-to-date image) gives kill 17.8 at D 1000 and turret < 5° 40 %.

## 5. Possibilities (none run as of 2026-11-20)

1. **Restore the separate cannon credit on Φ** on the current tree. Only setup that held D 1000, eight runs out of eight. Never tried with the up-to-date image. Known objection: Φ picks the enemy closest to the trajectory, hence biases target choice.
2. **Same structure, target-independent potential**: cannon credit on a geometric quantity that does not designate an enemy (e.g. "the theoretical shell would hit an enemy body", already computed for runs 481-485). To define and measure offline (one-step correlation with the gesture, threshold 0.08) before any code.
3. **Φ as primer, removed afterwards**: Φ credit until aim holds, then weight brought to 0. Measured risk: without an independent signal, aim is not maintained (3.3).
4. **Stay without an independent aim signal**: no measured lever left; 16 runs (523-543) varied reward, penalty, economy, entropy, batch, noise, critic, H there.

Ruled out by measurements: tuning the empty-reserve penalty (543), raising the trigger's entropy target (542), H as is (541-543).

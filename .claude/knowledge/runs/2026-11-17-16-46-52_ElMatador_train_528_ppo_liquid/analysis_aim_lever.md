# Lever search — why the turret does not hold the target (pure events, flat CNN)

2026-11-17. Measures on existing probes (528 s5/s9/s14, 525, 526, 520/522 s16, side 550) and CSVs. No code modified. Scripts: session scratchpad (`hold.py`, `jitter.py`, `sim_hold.py`, `aim.py`).

## 1. Holding the target pays a lot, and the game already pays it

Pass = yaw toward nearest enemy < 5°, enemy < 100 m, tank armed on entry (528 s9):

| pass duration | n | shots | touches | touches/s of presence |
|---|---|---|---|---|
| 0.1 s | 882 | 0.18 | 0.07 | 0.7 |
| 0.2 s | 353 | 0.37 | 0.18 | 0.9 |
| 0.3-0.4 s | 281 | 0.72 | 0.34 | 1.0 |
| 0.5-0.9 s | 160 | 1.86 | 1.04 | 1.7 |
| 1-1.9 s | 41 | 4.95 | 3.05 | 2.4 |
| ≥ 2 s | 11 | 11.9 | 7.5 | 2.9 |

Outside a pass: 0.076 touch/s. Same shape at s14 (0.07 → 0.55 → 2.45). One second on target = 3 touches (+1.8 reward and 15 shells refunded) vs 0.07 for a one-decision pass. The event reward thus contains the reward of holding; it is not reached: 3 % of passes last ≥ 1 s at s9, 0.6 % at s14. Mean duration 0.25 s then 0.19 s (522/520: 1.1 s).

## 2. What separates runs that hold from those that do not

Plant measured: one gesture acts over two decisions, 9.2° + 9.0° per unit of a·|a|.

| | σ canon_x | exploration noise | mean command \|μ\| | dwell < 5° |
|---|---|---|---|---|
| 525 s1 (untrained) | 0.37 | 2.3°/decision | 0.8° | 0.37 s |
| 526 s3 | 0.23 | 1.5° | 1.2° | 0.42 s |
| 526 s5 | 0.40 | 2.4° | 1.4° | 0.34 s |
| 528 s5 | 0.52 | 3.1° | 3.3° | 0.22 s |
| 528 s9 | 0.50 | 3.0° | 3.0° | 0.26 s |
| 528 s14 | 0.47 | 3.0° | 4.0° | 0.20 s |
| 525 s9 | 0.57 | 3.5° | 3.6° | 0.19 s |
| 522 s16 (separate credit) | 0.20 | 1.7° | 1.1° | 1.12 s |
| 520 s16 (separate credit) | 0.23 | 1.6° | 1.0° | 1.02 s |

Plant simulation, fixed target:

| σ | no correction at all (μ = 0) | ideal proportional controller |
|---|---|---|
| 0.5 | 0.38 s | 0.96 s (89 % of time < 5°) |
| 0.4 | 0.50 s | 1.8 s |
| 0.3 | 0.78 s | 5.3 s |
| 0.2 | 1.43 s | 16.5 s |

- 520/522's holding (1.0-1.1 s) is that of an **immobile turret at σ 0.2** (1.43 s in simulation), not fine servoing: their correction gain is 525's (audit 2026-11-17: +0.09 to +0.14 in both).
- Event runs do the opposite on both terms: σ rises 0.30 → 0.5-0.6 within the first 135 k, and the command becomes a sweep of 3-4° per decision (40°/s). Dwell falls during training, 0.37 s (untrained) → 0.20 s.
- At σ 0.5 stopping the sweep gives only 0.38 s on target (+0.15 s, ~+0.15 touch per pass per §1). At σ 0.2 the same stop gives 1.4 s (+1.2 s, ~+2 touches). Reward gain for the same change of μ is an order of magnitude larger at low σ. PPO optimises the return of the noisy policy: at σ 0.5 "sweep and fire in passing" and "stop on target" are almost equal.

## 3. Why canon σ stays at 0.5

- Initial rise (0.30 → 0.58 between 0 and 135 k): α at floor (1e-4), the policy gradient widens σ. Reading: nearest enemy out of field 67-72 % of time, a big random gesture finds it, events pay search through noise.
- Then the entropy regulator holds it: canon_x entropy measured at probe −0.01 to +0.16 for σ 0.53-0.60, exactly the 0 target of `continuous_target_entropy`. `alpha_canon_x` leaves floor at ~200 k and rises to 1.3-2.6e-3 between 300 and 380 k in 524, 525 and 528: PPO pushes σ_canon_x below target and the regulator opposes, when the trigger closes.
- 526 (batch 2400): σ canon 0.23-0.28 at s3 with α_canon 2e-3, no rotation bias, tour<5° 15-16 %; at s5 σ back to 0.39-0.45, dwell 0.42 → 0.34 s. Only case where σ was low in a single-advantage run, and it did not stay.
- 520/522/control: σ 0.2 with α_canon_x 0.010-0.017 from 81 k. Separate aim credit tightened σ despite the regulator. Perhaps how it acted.

## 4. Established / not

Established by measure: holding pays (§1 table) and is not reached; runs separate on two quantities only, canon σ and sweep; correction gain is the same; at σ 0.5 stopping the sweep earns ~10× less than at σ 0.2; entropy target 0 corresponds to σ ≈ 0.5 for canon, and the regulator has been active on canon_x since ~300 k.
Not established: causality. All correlational (run comparison + simulation). The ideal controller holds 0.96 s even at σ 0.5: noise does not forbid holding, it divides its gain. The sweep may stay the optimum even at low σ (it serves to find out-of-field targets). Old low-σ runs without aim (498: σ 0.15; 433/439: collapsed σ): all with the spatial softmax, laterally blind, hence not transposable but not reassuring either.

## 5. Options (Samuel)

1. Canon-specific σ ceiling for the two canon actions (≈ 0.25, level of 520/522/control), canon entropy target lowered accordingly so α does not push against the ceiling. Reward, CNN, single advantage unchanged. Internal code (`networks/constants.h`, `SoftClamp`, default of `continuous_target_entropy`). Direct causal test of the hypothesis.
2. Minimal variant: only canon entropy target at −1 (`continuous_target_entropy` = [0, 0, −1, −1]). Lifts the regulator's opposition after 300 k but leaves the initial rise; weaker test.
3. Time-correlated exploration noise on canon: would let exploration produce "stop sweeping for 0.5 s", which 10 Hz white noise almost never does. Heavier change (PPO log-probabilities).
Refutation criterion for 1, at save_5 (135 k) side 550: dwell < 5° > 0.6 s and tour<5° > 20 % = hypothesis held; dwell 0.2-0.3 s with a sweep of 3-4°/decision = refuted, the bottleneck is the sweep itself.

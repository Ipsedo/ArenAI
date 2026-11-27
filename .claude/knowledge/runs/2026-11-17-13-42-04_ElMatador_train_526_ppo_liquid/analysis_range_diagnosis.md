# Diagnosis — where the touches are lost (524, 525, 526 vs 522)

2026-11-17. Existing probes, side 550 unless stated. Distance = that of the best-aligned target at the moment of the shot. 522 = reference that aims (separate aim credit). Scripts: session scratchpad (`range.py`, `pfire.py`).

## 1. Touches per tank-second, by target distance

| | total | < 50 m | 50-100 m | 100-200 m | > 200 m |
|---|---|---|---|---|---|
| 526 s3 | 0.124 | 0.070 | 0.017 | 0.016 | 0.021 |
| 526 s5 | 0.093 | 0.054 | 0.011 | 0.010 | 0.017 |
| 525 s5 | 0.089 | 0.059 | 0.007 | 0.008 | 0.016 |
| 525 s9 | 0.076 | 0.055 | 0.005 | 0.005 | 0.011 |
| 525 s12 | 0.039 | 0.029 | 0.002 | 0.003 | 0.005 |
| 524 s16 | 0.073 | 0.052 | 0.007 | 0.004 | 0.010 |
| 522 s5 | 0.122 | 0.073 | 0.023 | 0.011 | 0.014 |
| 522 s9 | 0.098 | 0.041 | 0.027 | 0.012 | 0.016 |
| 522 s16 | 0.075 | 0.029 | 0.028 | 0.010 | 0.008 |
| 522 s24 | 0.160 | 0.063 | 0.055 | 0.021 | 0.021 |

Single advantage: 60-75 % of touches come from targets within 50 m and all the rest decays from save to save. 522: the 50-100 m band rises until it equals the near band.

## 2. Precision of a shell by distance (hit/shell all shots — aligned shots < 5°)

| | < 50 m | 50-100 m | 100-200 m | > 200 m |
|---|---|---|---|---|
| 526 s3 | 41 — 82 % | 7 — 34 % | 3 — 11 % | 2 — 7 % |
| 526 s5 | 38 — 70 % | 7 — 25 % | 3 — 11 % | 3 — 6 % |
| 525 s9 | 61 — 81 % | 19 — 35 % | 13 — 26 % | 13 — 28 % |
| 524 s16 | 61 — 76 % | 17 — 31 % | 8 — 22 % | 12 — 24 % |
| 522 s5 | 54 — 77 % | 15 — 38 % | 4 — 16 % | 4 — 16 % |
| 522 s16 | 74 — 84 % | 55 — 62 % | 30 — 44 % | 21 — 27 % |

Within 50 m a shell hits 4 times out of 10 without aiming. Beyond, being < 5° is not enough: 25-35 % in single-advantage runs, no progress between saves; 522 goes from 38 to 62 %.

## 3. Trigger on a target aligned < 5°, by distance (p_fire, count)

| | off-axis | < 50 m | 50-100 m | 100-200 m | > 200 m |
|---|---|---|---|---|---|
| 526 s3 | 0.59 | 0.70 (836) | 0.65 (815) | 0.62 | 0.61 |
| 526 s5 | 0.38 | 0.62 (713) | 0.48 (833) | 0.40 | 0.41 |
| 525 s5 | 0.19 | 0.64 (736) | 0.38 (829) | 0.28 | 0.26 |
| 525 s9 | 0.010 | 0.38 (966) | 0.14 (944) | 0.07 | 0.04 |
| 525 s12 | 0.006 | 0.10 (1 424) | 0.017 (1 103) | 0.013 | 0.009 |
| 524 s9 | 0.012 | 0.28 (1 503) | 0.07 (1 895) | 0.03 | 0.02 |
| 522 s9 | 0.036 | 0.11 (3 699) | 0.14 (5 912) | 0.09 | 0.06 |
| 522 s24 | 0.011 | 0.16 (3 581) | 0.13 (9 201) | 0.07 | 0.05 |

Single advantage: the trigger closes beyond 50 m (÷3 to ÷6 vs near) and the number of aligned states at 50-100 m stays at chance level (~900). 522: same trigger level at 50-100 m as near, and 6 to 10× more aligned states.

## 4. Encounter density

Share of time with an enemy within 50 m: 39-41 % side 550 at s3-s5, 9-18 % side 1000.

## Reading

1. The event reward pays only point-blank shots: that is where the touches are, and they do not require aiming.
2. The trigger learns fast (the clearest signal) to stop firing beyond 50 m, where a shell hits 3-7 % and costs a scarce reserve plus the empty penalty.
3. Once the trigger is closed at range, almost no shell leaves toward 50-200 m (525 s9: one shot per ~40 s per tank in the 50-100 m band). The turret has no event left to learn aim at range. The lock closes between 135 k and 243 k.
4. The curriculum criterion (touches / shots > 0.08) is satisfied by this closing alone: D rises, encounters within 50 m become rare, kills drop (525: 18 → 8 from D 500 to D 800).
5. In 522 and the control, separate credit learned aim before the trigger closed: at 50-100 m an aligned shot already hit 38 % at s5 and 62 % at s16, so the trigger stayed open there.
526 (batch 2400) touches none of this: it slows the trigger, but precision at 50-100 m falls meanwhile (34 → 25 %).
Limit: reading by run comparison, not intervention. 522 differs from 524-526 by aim credit, CNN and KL stop; the control (identical CNN) is no longer available here to redo it.

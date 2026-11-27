# 556 — gun elevation timeline, saves 1 → 24 (2026-11-27)

Probe side 1000, 4 episodes, 32 tanks, stochastic, binary at `5367a45e`, train rules. Question: when did the
resting return of the gun to the horizon disappear (`save_21_gun_elevation.md`)? One probe per save.

| save | index | D | armed elev. median | armed < −5° | σ_y median | rest μ_y, rel. elev. −25…−3° | setpoint < 40 m | hit/shell | hit/shell < 100 m | close yaw-aligned, pitch > +2° | first empty (s) | shots/tank-s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 27 k | 250 | −5.8° | 0.53 | 0.22 | 0.00 / −0.02 / −0.01 | −1.4° | 0.043 | 0.29 | 0.37 | 2.7 | 1.22 |
| 5 | 135 k | 400 | −9.9° | 0.71 | 0.19 | 0.00 / +0.01 / +0.02 | +1.7° | 0.051 | 0.32 | 0.40 | 3.1 | 1.10 |
| 9 | 243 k | 600 | −7.7° | 0.63 | 0.36 | −0.02 / −0.03 / 0.00 | +5.3° | 0.079 | 0.40 | 0.35 | 7.6 | 0.86 |
| 13 | 351 k | 800 | −11.7° | 0.76 | 0.33 | −0.01 / 0.00 / +0.02 | +5.0° | 0.129 | 0.35 | 0.64 | 26.4 | 0.38 |
| 17 | 459 k | 950 | −12.4° | 0.72 | 0.18 | 0.00 / −0.01 / 0.00 | +6.0° | 0.148 | 0.38 | 0.49 | 19.6 | 0.58 |
| 21 | 567 k | 1000 | −11.4° | 0.77 | 0.19 | −0.01 / −0.01 / 0.00 | +6.5° | 0.185 | 0.42 | 0.57 | 17.9 | 0.46 |
| 24 | 648 k | 1000 | −6.0° | 0.56 | 0.16 | −0.05 / −0.03 / −0.01 | +5.6° | 0.204 | 0.51 | 0.40 | 19.7 | 0.45 |

Rest μ_y: mean canon_y action mean with no enemy in front (|yaw| > 60°), relative elevation buckets
[−25,−15), [−15,−8), [−8,−3); negative raises the gun. References at 459 k: 554 s17 −0.07 / −0.05 / −0.02,
555 s17 −0.02 / −0.04 / −0.03 (armed < −5°: 0.25-0.33 in game matches).

## Reading

- The resting return was never lost: it is absent from save_1 on (rest μ_y within ±0.03 of zero, gun below −5°
  for 53-77 % of armed time at every save up to 21). The low gun is the unlearned state, not a regression.
- First sign of a return at save_24: rest μ_y −0.05 / −0.03 / −0.01, elevation median −6.0°, close-range
  hit/shell 0.51 (0.42 at s21), pitch > +2° on 40 % of close yaw-aligned shells (57 %). One save: not a trend.
- The parallax firing setpoint (target centred in the image) appears between saves 5 and 9 (< 40 m:
  +1.7° → +5.3°) and stays at +5…+6.5°.
- The share of close shells fired with the gun too low peaks at save_13 (0.64), when the trigger tightens
  (shots/tank-s 0.86 → 0.38).
- corr(μ_y, pitch) with a frontal target: −0.40 at save_1, −0.27 at saves 17-24: the pitch servo does not
  strengthen over the run.

## Not established

Whether 554 / 555 acquired the return earlier than 556 or started with it (no early probe of them with this
measure); whether save_24 is the start of the same acquisition. Next: game match on save_24 (third save of the
trajectory), probe of 554 / 555 saves 5-9 with their own binaries.

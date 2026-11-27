# Probes 506 — save_1, 5, 13, 17, 22

`build/` binary of 11/11 22:04 (code c02137c6, `--force`: flagged diff comes from commit dating). 4 episodes per side, stochastic, economy 10 / +2 / 20.

## Side 550 (side 1000 in parentheses when it differs)

| save | index | D | p_arm | ratio | hit/sh | empty_s | %disarmed | shots/s | turret<5° | σ cos/sin/pitch | \|μ\| cos/sin (signed mean) | μsat | σyaw° |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 27 k | 300 | 0.71 | 1.0 | 3.4 % (1.1) | 2.4 | 60 % | 1.70 | 17.9 % | 0.23/0.26/0.21 | 0.07/0.10 | 0 | 81 |
| 5 | 135 k | 350 | 0.48 | 1.5 (1.3) | 5.3 % (1.8) | 3.8 | 44 % | 1.81 | 16.4 % | 0.22/0.28/0.20 | 0.37/0.14 (−0.36/+0.11) | 0 | 39 |
| 13 | 351 k | 600 | 0.18 | 4.2 | 8.7 % (4.3) | 3.8 | 29 % | 1.10 | 16.9 % | 0.51/0.49/0.41 | 0.68/0.45 (−0.52/−0.15) | 0.2 % | 37 |
| 17 | 459 k | 650 | 0.08 | 4.9 (5.4) | 8.5 % (4.7) | 3.5 | 20 % | 0.64 | 17.8 % | 0.58/0.60/0.49 | 0.98/0.68 (−0.90/+0.49) | 9 % | 30 |
| 22 | 594 k | 800 | 0.12 | 4.3 | 11.5 % (5.6) | 3.9 | 22 % | 0.86 | 18.3 % | 0.61/0.61/0.48 | 0.94/1.49 (−0.70/+1.27) | 49 % | 21 |

## Yaw setpoint: a fixed sector, not aiming

Setpoint direction `atan2(tanh μ_sin, tanh μ_cos)` on step ≥ 10 decisions (side 550):

| save | resultant length R | mean direction | corr(setpoint gap, nearest-enemy yaw), all / \|yaw\| < 45° | mean μ_dir_y |
|---|---|---|---|---|
| 1 | 0.57 | 83° | +0.01 / +0.01 | −0.02 |
| 5 | 0.90 | 165° | +0.03 / +0.29 | −0.29 |
| 13 | 0.62 | −161° | +0.03 / +0.22 | −0.27 |
| 17 | 0.79 | 157° | +0.03 / +0.28 | −0.54 |
| 22 | 0.77 | 127° | +0.06 / +0.34 | −0.37 |

- From save_5, setpoint concentrates in a rear sector of the chassis (127-165°), and μ_dir_y negative: tanks reverse-drive, turret turned toward travel direction. The σyaw° decline (81 → 21°) measures this concentration around a fixed bias, not precision on the target.
- Target tracking is weak and limited to the ±45° cone (corr +0.22 to +0.34). Beyond, nothing.
- turret<5° flat from save_1 to save_22 (16-18 %): aim did not progress in 594 k.
- At save_22, μ_yaw_sin saturated in 49 % of decisions (|μ| > 1.5). H_yaw_sin −1.0 / −0.7 thus reaches target −1 through tanh saturation while σ rose to 0.61.

## Trigger

- Selectivity appears between save_5 and save_13 (ratio 1.5 → 4.2) and holds to save_22 (4.3), through the fall of p_off (0.44 → 0.08) more than p_al (0.67 → 0.35).
- Episode-start dump never disappears: reserve empties in 3.5-3.9 s in 95-98 % of tanks at every save.
- save_17 (CSV trough 410-460 k, kills 11.7 → 2.2): p_arm 0.08, shots/s 0.64 = trigger at its lowest, p_al 0.25. Then alpha_fire leaves floor (1e-4 → 6e-3 at 553 k) and the trigger rises (p_arm 0.12 at save_22).

## Landmarks (equal D, side 550)

- 504 save_9 (243 k, D 595): hit/shell 16.1 %, reserve empty 22.4 s, ratio 6.1, turret<5° 12.5 % ⟹ 506 save_13 (D 600): 8.7 %, 3.8 s, 4.2, 16.9 %.
- 504 save_16 (432 k, D 950): 11.2 %, 9.1 s, ratio 4.4, turret<5° 14.5 %. 506 save_22 only at D 800.
- 506 turns the turret slightly better (turret<5° +2-4 points), but hits less per shell and still empties its reserve.

## save_22 re-probed with turret angle (column turret_yaw, 12/11)

- Convention verified: setpoint = +turret angle + 3.8° (R 0.84). Turret stays parked near 125° (R 0.80) and sits at 17° median from its setpoint.
- In field (|yaw| < 40°, n 49.6 k): setpoint = 0.43·turret + 0.27·yaw + 0.41·setpoint(t−1). Network uses the current position (0.84 total): the « fixed sector + δ, turret ignored » hypothesis is false. It corrects only 27-45 % of the gap per decision, with ~16 % pull toward the parked sector.
- Median setpoint/target error in field: 16° near the sector, 26° at 30-60° from the sector.
- Learned corrector is a weak, noisy P, as in velocity (504: ≈ 3°/decision). Absolute did not change its nature.

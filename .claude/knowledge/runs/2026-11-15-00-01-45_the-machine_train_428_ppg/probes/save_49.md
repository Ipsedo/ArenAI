# Probe train_428_ppg / save_49

- Checkpoint: index ≈ 1323k, D = 1000. This is the last save. The run is in progress, CSV at 1343k.
- Binary: `cmake-build-release/arenai_agent/arenai_agent_probe`, compiled at 9d843717, **no inference diff** with the run code (0e5093a1).
- Protocol: sides 550 and 1000, 4 episodes each, stochastic mode, economy 10 / +3 / 30.

## Summary, compared to save_27 (729k) and 423 save_21 (567k)

| side | save | ratio | p_al (n) | p_off | p_arm | hit/shell | h<5° | shots<5° | enrich | empty_s | %empty | %disarmed | fire/s | turret<5° | σx | σy | p_zoom |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 550 | 423 s21 | 4.86 | 0.359 (836) | 0.074 | 0.139 | 0.229 | 0.369 | 0.500 | 2.04 | 9.7 | 0.70 | 0.095 | 1.08 | 0.277 | 0.37 | 0.37 | 0.47 |
| 550 | s27 | 8.60 | 0.146 (1441) | 0.017 | 0.054 | 0.268 | 0.345 | 0.649 | 2.00 | 31.9 | 0.49 | 0.026 | 0.50 | 0.340 | 0.35 | 0.34 | 0.40 |
| 550 | **s49** | 6.97 | 0.194 (1070) | 0.028 | 0.056 | **0.292** | **0.426** | 0.633 | 1.76 | 47.0 | 0.40 | 0.004 | 0.53 | **0.375** | 0.29 | 0.29 | **0.03** |
| 1000 | 423 s21 | 7.53 | 0.293 (592) | 0.039 | 0.061 | 0.191 | 0.327 | 0.467 | 2.15 | 20.0 | 0.73 | 0.060 | 0.53 | 0.235 | 0.58 | 0.56 | 0.39 |
| 1000 | s27 | 21.9 | 0.149 (888) | 0.007 | 0.025 | 0.270 | 0.352 | 0.656 | 2.60 | 46.1 | 0.39 | 0.011 | 0.23 | 0.264 | 0.44 | 0.43 | 0.34 |
| 1000 | **s49** | 11.1 | 0.196 (775) | 0.018 | 0.026 | **0.295** | **0.466** | 0.573 | 1.75 | 69.9 | 0.32 | 0.008 | 0.25 | **0.332** | 0.43 | 0.40 | **0.02** |

Armed p_fire by angle to the best target ([0-2[, [2-5[, [5-10[, [10-20[, [20-45[, [45-180[ °):
- side 550: 0.138 / 0.080 / 0.038 / 0.034 / 0.038 / 0.021
- side 1000: 0.071 / 0.034 / 0.017 / 0.014 / 0.017 / 0.018

## Reading

- **Zoom abandoned.**
  - p_zoom falls from 0.34–0.40 to 0.02–0.03, and its correlation with the trigger disappears (0.43–0.48 → ≈ 0).
  - This is the Hd drop in the CSV (0.35 → 0.07–0.09 from 1.18M): `alpha_zoom` barely leaves the floor (≤ 1.8e-4). The 0.01 discrete target lets the zoom become deterministic.
- **Precision per aligned shot: new fact.** h<5° (hits per shell fired under 5°) goes from 0.35 to **0.43 / 0.47**. This is the first ballistic gain at equal angle of the lineage: 423 and save_27 capped at 0.33–0.37. This gain arrives **without zoom**. Hit/shell rises to 0.29.
- **Aim.** The turret is within 5° of a target in 0.375 / 0.332 of decisions, vs 0.340 / 0.264 at save_27, up on both sides. Canon σ stays at 0.29–0.43, no ceiling nor saturation.
- **Trigger less sharp in angle.**
  - The fire probability when the barrel is aligned and the target close rises (p_al 0.15 → 0.19).
  - But the off-axis floor also rises (p_off 0.007–0.017 → 0.018–0.028) and becomes **flat beyond 10°** (0.014–0.038).
  - The ratio therefore drops (8.6 → 7.0 and 21.9 → 11.1). On angle alone, it goes from 6.4 / 9.4 to 3.5 / 2.6.
  - 15–20 % of shots leave at more than 20° (8–9 % at save_27), with hit/shell 0.007.
- **In deterministic mode**, the agent still fires only in 2.8 % (side 1000) to 6.6 % (side 550) of aligned states: it stays mute in game.
- **Reserve.** Emptied even less: 32–40 % of tank-episodes empty it, the first time at 47 / 70 s median, and disarmed steps are under 1 %.
- **Chassis movement: change of strategy.**
  - save_27: dir_y (wheel speed, pre-tanh) at −1.04 on average, i.e. full speed one way; steering dir_x nearly null (−0.19).
  - save_49: steering **dir_x at +1.05** (tanh ≈ 0.78, 17 % saturated) and dir_y +0.35, i.e. reduced speed the other way.
  - The tank therefore turns in circles, steered to one side only. Consistent with the shorter median life on side 550 (47 s vs 96 s) and with kills rising back to 18–21 in the CSV.

## Conclusions

**Established** (same binary, same protocol as save_27)
- Precision per aligned shot progresses (h<5° 0.43–0.47), as does turret alignment.
- Zoom is abandoned.
- The chassis turns in circles steered to one side.

**In the noise / undecided**
- The ratio drop comes from a higher but flat off-axis floor. Aligned counts are 775 to 1070, so the ratio remains noisy.
- Unknown whether abandoning zoom helps precision (zoom = FOV ×4, perhaps harmful to target detection) or is unrelated.
- Unknown whether turning in circles is an aiming strategy (yaw sweep by the chassis) or an evasion one.

## Useful next steps
- Measure corr(μ canon, angular error) and corr(μ dir_x, yaw) to know who aligns: canon or chassis.
- Check in game the chassis rotation behaviour.

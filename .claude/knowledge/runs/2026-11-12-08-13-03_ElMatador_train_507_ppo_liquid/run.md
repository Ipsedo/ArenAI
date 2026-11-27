# train_507_ppo_liquid

| | |
|---|---|
| machine | ElMatador |
| user | samuel |
| launch | 2026-11-12 08:13:03 |
| commit at launch | `c02137c6` — absolute angle for turret |
| commits ≤ 90 min after | `25a75a34` |
| diff at launch | unknown (identity recorded after the run ended) |
| raw data | `outputs/train_507_ppo_liquid` on ElMatador |

## Established facts

- 2026-11-12: Finding (missed in 506's analysis): the aim potential shaping was ALREADY active since `b5309399` (06/11, scale 0.3, Cauchy 2.5 m + 3°): `TrainTankEnvironment::step` adds 0.3·(γΦ′ − Φ) per decision. Φ = `JoltEnemyTank::get_aim_potential` (theoretical shell, ray_test, max over living enemies). r_aim ≈ −1.9e-4 = tax −(1−γ)·0.3·Φ ⟹ mean Φ ≈ 0.13. `aim_tolerance_*` members serve only Φ (hit paid as damage fraction). Why: at 3°, Φ slope almost null beyond ~15° (at 190 m: Φ 0.36/0.12/0.035 at 5/10/20°) while turret sits at 12° median; same defect as 472. 506's aim gain 0.27 was measured WITH this potential.
- 2026-11-12: Choice by Samuel: Φ own tolerance 5 m + 10° (in `jolt_enemy_tank.cpp`), all enemies, scale 0.3 kept; velocity command back (`ENEMY_NB_CONTINUOUS_ACTION` 4, torch_types right_joystick, names canon_x/canon_y), default entropy target {0,0,0,0} in the 3 hyperparams, `MIN_ALPHA` 1e-4 kept. Controller/model `aim_target` support stays (unused, no `include/` change). Tests: new `AimPotentialKeepsASlopeAcrossTheFieldOfView` (20° ⟹ Φ ∈ ]0.3; 0.9[), « missed » enemy moved to 72°. model 91, agent 2434 ok in `build/`. `cmake-build-release` not rebuilt (506 running). Compare to 504 at equal D; at probe: slope μ_canon_x vs bearing, turret <5° > 18 %, r_aim (bigger tax since mean Φ higher, expected ~−5e-4).
- 2026-11-12: Run launched 12/11 08:13 (commit `25a75a34`). CSV at 40-60 k (D 300-350): hit 10.4-11.2 % vs 8.0-8.7 % for 504, kills 12-14.5 vs 7.5-8.4, fire equal. Probe save_2 vs 504 save_2 (side 550/1000): hit/shell 5.0/2.2 % vs 3.8/1.5 %, p_arm 0.55 vs 0.79 (less dump, empty 3.0 s vs 2.4), but turret <5° 9/6 % vs 10 % and corr(yaw, μ_cx) in field −0.37 vs −0.31 ⟹ turret not yet better. Report `probes/save_2.md`.
- 2026-11-12: Probe save_5 (135 k, D 450; 504 save_5 at D 400): turret not better (<5° 13.2/9.6 % vs 13.8/12.7 %, corr yaw–μ_cx −0.38 ≡ −0.39); hit/shell 10.0/5.1 % vs 8.4/3.9 % (hits of shots <5° 18/12 % vs 16/8 %); trigger less selective (ratio 2.9/3.7 vs 3.5/5.0, p_al 0.30 vs 0.53) and more economical (empty 6-8 s vs 6 s). CSV lead does not come from alignment. Turret gate at save_9 (504: <5° 12.5 %).
- 2026-11-12: Probe save_9 (243 k, D 650; 504 save_9 at D 600): turret slightly better (<5° 14.9/13.4 % vs 12.5/10.1 %, gain μ_cx/10° 0.076 vs 0.062) but under 18 % threshold, within noise; trigger stayed in dump (empty 4 s, p_arm 0.17, 0.87 shot/s) while 504 learned economy (22-42 s, p_arm 0.05) ⟹ hit/shell 9.9/5.1 % vs 16.1/7.6 %. CSV kills ×1.4 = shot volume. Report `probes/save_9.md`.
- 2026-11-12: Probe save_12 (324 k) vs 504 save_12 — **verdict: lever inconclusive.** turret <5° 13.9/11.6 % vs 14.5/13.1 % (equal over 4 saves), corrector slope +35 % without effect; 507 trigger out of dump (reserve 21-28 s, ratio 8.7/13.6, hit/shell 18.7/11.8 %) but 504 had done the same at save_9 then fell back ⟹ seed noise. Underlying diagnosis: see comparisons note on aim diagnosis (range + out-of-field).

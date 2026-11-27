# train_556_ppo_liquid

| | |
|---|---|
| machine | ElMatador |
| user | samuel |
| launch | 2026-11-27 02:00:15 |
| commit at launch | `5367a45e` — docs(harness): skip commit hooks once final-check passed |
| commits ≤ 90 min after | none |
| diff at launch | none (clean tree, run in progress) |
| raw data | `outputs/train_556_ppo_liquid` on ElMatador |

## Established facts

- 2026-11-27, 568 k: D 1000 at 461 k (554: 950 max, 555: stuck 850), kill 21 vs 18 over 430-546 k, but hit 0.17 vs 0.27-0.29 and fire ×2.
- 2026-11-27, matches s17 and s21 (`matches/save_21.md`): below 554 s17 / 555 s17 in game — dry at 10 s 0.27 / 0.48 (0.03-0.05), hit/shell 0.18-0.26 (0.32-0.39), armed gun < −5° 70 % of the time; 553 s36 profile. One seed, confounded.
- `mask_loss` 0.28 → 0.18-0.21, no clear plateau.
- 2026-11-27, probe s21 (`probes/save_21.md`) and shots by range vs 554 s17 / 555 s17, side 1000 (`comparisons/2026-11-27_shots-by-range-554-555-556.md`): share of shells with best-aligned target > 200 m ≡ in the three runs (0.53 / 0.51 / 0.55) ⟹ not the cause. Specific to 556: under 100 m, 57 % of yaw-aligned shells leave with pitch error > +2° (554 38 %, 555 24 %) and hit 0.36 there (0.69 / 0.63); close-range hit/shell 0.42 vs 0.58 / 0.56; volume 0.46 shots/tank-s (0.37 / 0.21); first empty reserve median 18 s (43 / 35). Armed gun median −11.7° (−1.9 / −3.1).
- 2026-11-27, gun elevation diagnosis s21, side 1000 (`probes/save_21_gun_elevation.md`): pitch servo has the right sign (corr −0.31); firing setpoint follows the camera-parallax curve (gun +6.5 / +2.0 / +0.7° below the target at < 40 / 40-70 / 70-110 m, as 554; 555 holds the gun above); resting posture has no return to the horizon (μ_canon_y ≥ 0 above −3°, 554 / 555 raise the gun between −25° and −3°). Oracle pitch only: hit/shell 0.185 → 0.281, close-range 0.42 → 0.68, first empty reserve 18 → 35 s ⟹ elevation alone accounts for the gap to 554 / 555 on train rules.
- 2026-11-27, gun elevation timeline saves 1-24, side 1000 (`probes/timeline.md`): no resting return to the horizon at any save from 27 k to 567 k (rest μ_y within ±0.03, armed gun < −5° 53-77 %) ⟹ unlearned state, not a regression; first sign at save_24 (648 k): rest μ_y −0.05 / −0.03, elevation median −6.0°, close-range hit/shell 0.51 vs 0.42 at s21 (one save). Parallax firing setpoint appears between saves 5 and 9.
- 2026-11-27, match s24 (648 k; `matches/save_24.md`), trajectory s17 → s21 → s24, side 1000 / 500: armed gun < −5° 0.71 → 0.70 → 0.49 / 0.72 → 0.52 → 0.43, hit/shell 0.18 → 0.19 → 0.22 / 0.22 → 0.26 → 0.28, kills/tank at 500 0.29 → 0.33 → 0.43; dry at 10 s unchanged 0.27 → 0.26 → 0.24 / 0.48 → 0.39 → 0.30 (554 / 555 s17: 0.02-0.05) ⟹ the early dump in game is not carried by gun elevation; still below 554 s17 / 555 s17.
- 2026-11-27, first 10 s of the game matches (`matches/early_dump.md`): 556 s21 / s24 fire 5.1-5.7 shells/tank at side 1000 (554 / 555 s17: 2.3-2.9), 34-35 % of tanks fire ≥ 8 shells (11-13 %), and these do not spawn nearer to an enemy (107-141 m vs 55 m for 554 / 555). Off-target trigger bimodal: 2-3 % of decisions ≥ 20° off any target have p_fire > 0.5 (p99 0.85-0.93; 554 0.65, 555 0.22), in bursts, turret sweep stopped and zoom on; aligned targets ≥ 100 m above threshold 7-13 % (1-2 %). ⟹ early dump = trigger, not gun elevation.
- 2026-11-27, frames of off-target firing decisions, s24 deterministic, side 1000 (`probes/save_24_offtarget_frames.md`): on 24 frames drawn at random the agent fires at an object centred in the zoomed image — 14 tanks (no living enemy within 20° of the axis ⟹ dead tanks, inferred) and 10 map items; not-firing frames are bare terrain (19 / 24). 554 s17 fires at the same content (14 / 10) half as often (1.1 % vs 2.2 % of off-target decisions).
- 2026-11-27, trigger on wrecks, s24 deterministic, side 1000 (`probes/save_24_wrecks.md`; Samuel saw the wreck shooting in game): wreck on the gun axis fired at 0.37 / 0.47 / 0.17 of decisions at < 60 / 60-100 / 100-200 m vs 0.74 / 0.62 / 0.21 for a living enemy, same posture (turret stopped, zoom on); 14 % of all shells go to a wreck on the axis; nothing within 20°: 0.012. Code: killed tanks stay as wrecks until the episode ends, pay nothing when hit, and are drawn like living tanks.
- 2026-11-27, wreck vs living tank decoding, s24 deterministic, side 1000, 8 episodes (`probes/save_24_wreck_decoding.md`): balanced accuracy 0.58-0.69 for a CNN trained on the pixels and 0.56-0.66 for a linear probe on the agent's vision features, at every distance band (chance 0.50, shuffled labels 0.48-0.52), while « tank on the axis vs nothing » is decoded at 0.80-0.98 by the same CNN ⟹ the image barely carries the wreck / living distinction; the detached parts are not enough.
- 2026-11-27, wreck decoding with hatched wrecks, same protocol (`probes/save_24_wreck_decoding_hatched.md`, binary `c6177f2a`): pixels CNN 0.93-0.99 under 200 m (before 0.58-0.69), 0.61 at 200-400 m (control 0.77); untrained vision features 0.52-0.74 ⟹ the distinction is in the input under 200 m; P4 training rule recorded there.

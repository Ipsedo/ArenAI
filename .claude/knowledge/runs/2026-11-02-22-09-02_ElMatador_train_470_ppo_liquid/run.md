# train_470_ppo_liquid

| | |
|---|---|
| machine | ElMatador |
| user | samuel |
| launch | 2026-11-02 22:09:02 |
| commit at launch | `173a0865` — hit: +0.3, damage: -0.2 => attacking is more attractive |
| commits ≤ 90 min after | none |
| diff at launch | unknown (identity recorded after the run ended) |
| raw data | `outputs/train_470_ppo_liquid` on ElMatador |

## Established facts


- 2026-11-03: = 469's config (identical up to output_folder) + commit `173a0865` (r_hit 0.3, received damage −0.2). Read 2026-11-03 at 735 k steps (still running). Ammo (jolt_enemy_tank.cpp): 10 initial, +10 per hit, max 30, cooldown 1/6 s (6 shots/s max), empty reserve with no shell in flight = death at −0.5. Break-even +10/hit = 10 % hit; measured 7-9 % ⟹ the reserve always empties.
- 2026-11-03: `s` (episode duration s, window 32) 16.4 → 13-15 s over the run, no rise; 467: 19 → 13; 468/469: 17-23 s (rarer shots). Duration ≡ time to empty ~10-18 shells at ~1/s. `fire` 0.85-1.18/s over the eight eighths. **Not p_init × 10 Hz** (error made the morning of 2026-11-03, corrected by the probe): can_fire true only 18 % of time and p_fire 0.68 when can_fire ⟹ cadence bounded by the game. D: 250 → 700 (500 k), plateau 700 after; kill 0.8 → 2.0 (550 k) → 1.1-1.3; hit 7.5-8.8 % → 6.5 %; r_hit max 0.0057 at 550 k then 0.0035; Hc 0.18 → −0.17 (σ tightening). No end-of-episode cause in CSV.
- 2026-11-03: probe save_6 (`probes/save_6.md`): trigger flat vs angle (0.70-0.74 from 0° to 90°), 70 % > 0.5 ⟹ deterministic dump in game; only non-fire state = turret at full tilt first 2 s; turret weaker than 467. The CSV cadence 1/s and the 6/s dump in game are the same trigger (p ≈ 0.9 outside spawn) read sampled vs deterministic (threshold 0.5). Episode duration will not move while hit < 10 %; the per-shot reward (Cauchy, hit 0.3) still does not condition the trigger on angle — D gain comes from elsewhere (spawn/curriculum), not aim, which regressed.
- 2026-11-03: shot-level measurement (`probes/save_6.md`, second part): P(hit) depends strongly on alignment and distance: 50 % when aligned < 5° and target < 40 m (n = 49), 20 % < 2°, 7 % at 5-20°, 1.6 % > 20°, 6.4 % overall; only 1.2 % of shots are aligned+near ⟹ the gradient exists, the policy does not exploit it. Physics allows hitting one time in two when aligned and near; a 5-20° shot at 150 m pays 7 % and stays positive in expectation, so the agent never retains a shell. A fire cost can only work between 0.3 × 0.07 ≈ 0.02 (current shot negative) and 0.3 × 0.2 ≈ 0.06 (aligned shot positive), paid at impact with the hit; the signal must survive the phase where every shot is losing (465 = trigger extinguished): alignment observation, cost ramp, or scripted trigger during aim learning. Out-of-arena shells: 11 % of shots fly 20 s in the void, to bound or penalise.
- 2026-11-03 (Samuel: « fais le 3 »): `shot_cost` 0.03 in jolt_enemy_tank.cpp, subtracted in `detail.aim` for each landed shell, i.e. paid at impact with the hit and the Cauchy (r_aim in CSV becomes net per-shot reward). Expected: current shot (7 % hit, Cauchy ≈ 0.055) ≈ 0.021 + 0.0055 − 0.03 ≈ −0.003; shot aligned < 2° (20 %) ≈ +0.04; aligned+near (50 %) ≈ +0.17. Model tests 78/78 unchanged. Empty reserve/death untouched. Known weakness: an out-of-arena shell pays its cost 20 s later (γ^200 ≈ 0.13). Run 471 = this code.

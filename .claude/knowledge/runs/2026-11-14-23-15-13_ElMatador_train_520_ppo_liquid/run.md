# train_520_ppo_liquid

| | |
|---|---|
| machine | ElMatador |
| user | samuel |
| launch | 2026-11-14 23:15:13 |
| commit at launch | `e554371e` — update readme |
| commits ≤ 90 min after | `c33e68de`, `4c099cb6`, `0e5093a1` |
| diff at launch | unknown (identity recorded after the run ended) |
| raw data | `outputs/train_520_ppo_liquid` on ElMatador |

## Established facts

- 2026-11-15: 520 = 515 + `discrete_target_proba` 0.05 → 0.01 (commit 0e5093a1); tests the entropy regulator of the trigger (510/514: α_fire ×20 at 400-600 k ⟹ trigger loosens). Same-lineage reference 514 (economy 10/3/30); 515 = 514 + gamma 0.998.
- 2026-11-15 (CSV at 888 k): α_fire 1e-4 from start to end (510/514: ×20 at 400-600 k); CSV hit 23-25 % over the whole D 1000 cruise (510/514: 17-21); fire 0.3-0.75/s (÷2); kill dip at 430-530 k (11-16), reaches 510/514 (22) at 860 k. D milestones identical to 510/514/515.
- 2026-11-15: probes save_20 (540 k) / save_32 (864 k), sides 550/1000: disarmed 1.2-3.1 % (514: 10-15), empty reserve p50 22-63 s, hit/shell 24.5-27.6 % (514: 18-22) at equal tour<5° (25-29 %); h<5° 40-44 % (514: 30-34) ⟹ trigger picks its moment better inside the cone. p_off 0.017-0.061, p_al 0.26-0.46; p > 0.5 on 3.5-23 % of aligned states only, but save_32 deterministic still fires 0.81/0.36 shots/s (68-76 % of stochastic) with hit/shell 28-29.5 %, disarmed 1-2.7 % (514 s34 det: 1.25/0.41, 22-23 %, 3-6 %) ⟹ never conclude "mute" from the p > 0.5 share alone, probe with `--deterministic`. From s20 to s32 cadence ×2, p_off ×2.6 at 550 with α_fire at floor. μ dir_y frozen ≈ −0.95 (as 514) ⟹ Hc < 0 at 780-880 k.
- 2026-11-15: save_36 (972 k): hit/shell 17.5/17.9 %, h<5° 24/25 % (s32: 40), disarmed 12.5/7.4 % while p_off drops (0.037/0.017) ⟹ precision drop, not trigger release; μ dir_y −0.95 → −0.5. Settled with save_35/37: site bias −2.7° at s36 only (s35/s37 −0.7°) ⟹ oscillation. Touches per tank-second s32 0.276/0.142, s35 0.292/0.143, s37 0.320/0.140: flat; cadence ↑, precision ↓ (h<5° 40 → 31-35 %), disarmed 3 → 5-11 % ⟹ slow drift toward 514's equilibrium without regulator. Tie-break criteria: median site bias (shots < 5°, nearest target, 50-200 m) and touches/tank-s.
- 2026-11-15: full chronology (probes/timeline.md): save_9 total dump (p_off 0.6, disarmed 42-64 %), save_16 trigger nearly extinct (0.13-0.29 shot/s, touches/s ÷3 vs 514), then cadence ×4.5 from s20 to s37; tour<5° flat 27-33 % since s16; h<5° 44 → 35 % = trigger threshold dropping. r_rcv = −r_hit/3 (510/514/520) ⟹ r = ⅔ r_hit ∝ touches per tank-step: r of 520 ≤ 514 over the whole run (÷2.5 at 400-550 k), rejoins at 850 k. Most likely scenario: convergence to 514's point (the target changes the path, not the arrival); alternative: same touches/s with fewer shells (s37: 0.32/0.14 vs 514 0.27-0.33/0.12-0.17, disarmed 5/3 vs 11-13 %).
- 2026-11-15: oracle on save_37 (probes/oracle/save_37.md, fov 80): touches/tank-s policy 0.320/0.140; yaw + trigger oracle with agent's site 0.282/0.190; full oracle 0.586/0.361 (×1.8/×2.6) ⟹ bottleneck in the agent, and it is the site: perfect yaw earns nothing without it. Agent's trigger compensates by firing close (72-83 m vs 138-174 m for oracle). Clean 2×2×2 plan = 2 probe options to add (Samuel validation); later added (uncommitted probe change) and run, see probes/oracle/save_37.md.
- 2026-11-15: first-use of the memory note "run 520": save_32 usable in game (jeu-ne-tire-pas-mauvais-checkpoint); follow p_off and dir saturation on later saves. First run where the dump disappears at D 1000; single seed but the gap is wide and consistent with the mechanism.

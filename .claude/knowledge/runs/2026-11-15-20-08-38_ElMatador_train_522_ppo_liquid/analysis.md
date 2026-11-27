# Run 522 (ppo_liquid)

Launched 2026-11-15 20:08; analysed while running on 2026-11-16 (06:52 at 894 k, 10.7 h, 83 k steps/h, 2.66 M tank-steps/h, 33 saves; final updates up to 18:30). Code = 9d843717 + empty penalty uncommitted at launch, committed as-is during the run (ee1ca624, +161 min). High confidence: training binary of 2026-11-15 20:07, `r` residual matches the penalty from 5 k.

## What the run tests

521 + **penalty −D/1000 at the step the reserve empties** (−0.25 at D 250, −1 at D 1000), truncated respawn kept. Config identical to 521 (2 shells per touch, reserve 10, max 30). One change.
Reading the penalty: residual `r − r_hit − r_rcv − r_aim` (≈ 0 in 521); residual / P = emptyings per tank-decision.

## CSV (to 894 k)

D milestones identical to 520 to the thousand: D 1000 at 429 k (521: 461 k).

| phase | index | D | fire/s | hit | kill | r_hit (·10⁻³) | r (·10⁻³) | one emptying every |
|---|---|---|---|---|---|---|---|---|
| taxed dump | 0-120 k | 250-416 | 1.25-1.74 | 0.07-0.15 | 11-23 | 4.9-9.1 | 0.3-1.9 | 9-10 s |
| trigger descent | 120-280 k | 450-700 | 1.2 → 0.19 | 0.15 → 0.28 | 22.9 → 18.0 | 8.9 → 4.6 | 1.5-2.4 | 12 → 55 s |
| trough | 280-460 k | 750-1000 | 0.15-0.37 | 0.30-0.37 | 11.8-18.1 | 2.2-3.8 | 1.1-1.6 | 100-290 s |
| recovery | 460-800 k | 1000 | 0.13-0.29 | 0.33-0.42 | 15.7 → 18.8 | 3.0 → 4.8 | 1.3 → 2.5 | 135-250 s |
| current end | 800-894 k | 1000 | 0.24-0.35 | 0.40-0.44 | 17.6-19.0 | 4.2-4.7 | 2.2-2.6 | 180-245 s |

At D 1000 vs same lineage:

| | 520 (840-900 k) | 521 (680-752 k) | 522 (800-894 k) |
|---|---|---|---|
| hit | 0.25-0.26 | 0.18-0.21 | **0.40-0.44** |
| fire/s | 0.74-0.95 | 0.81-1.09 | 0.24-0.35 |
| kill | 20.5-23 | 21-23.5 | 17.6-19 |
| r_hit (·10⁻³) | 7.2-8.5 | 7.8-8.4 | 4.2-4.7 |
| miss (m) | 52 | 56-69 | 39-44 |

Optimisation: skip < 1 %, kl 0.04-0.07 (peaks 0.10 at 460 k and 680 k), ev 0.66-0.85, α_fire and α_zoom at floor. Hc ≈ 0 cruising, −0.28 on last slice (880-894 k). Hd 0.43 → 0.13-0.18.

## Probes (details: probes/timeline.md), save_32 (864 k), sides 550/1000

| | shots/s | hit/shell | touches at shots < 5° | ratio | 1st empty p50 | %empty | disarmed | touches/tank-s |
|---|---|---|---|---|---|---|---|---|
| 520 save_32 | 1.06 / 0.53 | 26.0 / 27.0 % | 40 / 40 % | 7.5 / 17 | 22 / 28 s | 42 / 48 % | 3.1 / 2.3 % | 0.276 / 0.142 |
| 521 save_26 | 1.25 / 0.82 | 20.6 / 19.1 % | 33 / 31 % | 4.4 / 7.9 | 5 / 9 s | 80 / 85 % | 19 / 15 % | 0.258 / 0.157 |
| **522 save_32** | 0.43 / 0.13 | **35.2 / 36.6 %** | **52 / 54 %** | 12 / 57 | 32 / 51 s | 38 / 26 % | 0.6 / 0.4 % | 0.152 / 0.046 |
| 522 save_32 det | 0.19 / 0.08 | 48.0 / 42.2 % | 62 / 57 % | 5.6 / 16 | 31 / 61 s | 14 / 15 % | 0.1 % | 0.090 / 0.035 |

## Diagnosis (894 k)

Established:
- **Record precision per shell**, confirmed by two independent measures: CSV hit 0.40-0.44 and probed hit/shell 35-37 % (46 % at save_16), vs 26-27 % for 520's best point and 30 % for its trough. The gap is in the trigger: the turret is not better aligned than 520's (tour<5° 28/21 % vs 29/25 %).
- Shot sorting: 59-62 % of shots at < 5° (520: 56-59 %, 521: 51-52 %) and hit at 52-54 % (520: 40 %). Median miss 9-11 m (520: 14-18 m, 521: 19-37 m). Off-axis p_fire 0.005-0.020.
- Dump and disarming gone: 0.4-0.6 % of time disarmed, one emptying per tank every 3-4 min in training.
- Touch throughput sacrificed: touches/tank-s 0.152/0.046 = 55 %/32 % of 520 save_32. r_hit 4.2-4.7·10⁻³ vs 8·10⁻³, kill 18-19 vs 21-23.
- Dynamics = trough then recovery on axis, as 520 but 160 k earlier and without going back through volume: cadence trough at save_10 side 550 (0.24 shot/s) then save_16 (0.16/0.08), recovery to 0.42-0.43 side 550 at save_24-32. r_hit ×2 from 440 k to 820 k at rising hit: in 522 precision and reward rise together, whereas 521 opposed them (corr −0.79).
Measured weak points:
- Side 1000 trigger stays very low: 0.13 shot/s at save_32 (0.19 at save_24), p_fire 0.075 at 0-2°. Most of the throughput is lost there (a third of 520).
- Deterministic: mute at save_16, then 0.19/0.08 shot/s at save_32 (45-60 % of stochastic) with hit/shell 42-48 %. p > 0.5 on 2-7 % of aligned states only.
- Turret regresses since save_16: tour<5° 45/37 % → 39/23 % → 28/21 % while σ canon rises (0.25-0.35 → 0.45-0.60). Final level = 520's.
- μ dir_x saturates: mean −1.4, |μ| > 1.5 on 45-54 % of states at save_32 (0 % to save_24, where dir_y saturated at 25-38 %). Consistent with Hc −0.28 at CSV end.
- Zoom abandoned: p_zoom 0.77 (save_9) → 0.55-0.60 → 0.09-0.16 (save_32). Origin of Hd drop.
- No restraint at low reserve until save_11 (p_fire identical at 1-2 and 6-10 shells); not re-measured later.
In the noise: hit/shell 46 % (save_16) vs 35-37 % (save_24, save_32): very different cadences, 4 episodes per point; the 30 k lead over 521 on D milestones.
Not settled: whether touch throughput rejoins 520/521 while keeping precision (r_hit gains ≈ 0.5·10⁻³ per 100 k since 460 k, 3.5·10⁻³ missing); whether dir_x saturation and zoom abandonment are policy choice or slide.
Markers: failure = CSV fire < 0.15 lasting with kill < 15 (back to trough), or dump (emptyings > 1 per 60 s). Success = touches/tank-s ≥ 0.20/0.10 at hit/shell ≥ 33 %. Best probed save: save_32 (hit/shell 35-37 %, ratio 12/57, fires deterministic).

---

# Update 2026-11-16 13:15: 1 482 k, 17.1 h, 55 saves, still running

Throughput 85.6 k steps/h (2.74 M tank-steps/h). Code unchanged (commits f80e0521 and 9cabe7b1 during the run do not touch the 2026-11-15 20:07 training binary).

CSV 894 k → 1 482 k (D 1000):

| slice | fire/s | hit | kill | r_hit (·10⁻³) | r (·10⁻³) | one emptying every | ev | skip | α_fire |
|---|---|---|---|---|---|---|---|---|---|
| 800-880 k | 0.26-0.28 | 0.38-0.42 | 17.6-18.8 | 4.2-4.7 | 2.2-2.6 | 150-215 s | 0.69-0.73 | < 0.3 % | 1.1·10⁻⁴ |
| 880-1 040 k | 0.32-0.41 | 0.37-0.41 | 19.9-20.5 | 4.8-5.3 | 2.5-2.7 | 120-150 s | 0.59-0.70 | 0.1-0.6 % | 1.0-1.4·10⁻⁴ |
| 1 040-1 160 k | 0.16-0.34 | 0.36-0.39 | 18.2-21.2 | 4.2-5.5 | 2.2-2.6 | 97-172 s | 0.47-0.67 | 1.2-2.5 % | 1.4-2.4·10⁻⁴ |
| 1 160-1 320 k | 0.24-0.36 | 0.35-0.44 | 19.2-21.2 | 4.9-5.9 | 2.7-2.9 | 90-197 s | 0.55-0.66 | 0.6-2.4 % | 3·10⁻⁴ → 7·10⁻⁴ |
| 1 320-1 482 k | 0.29-0.47 | 0.37-0.41 | 18.8-19.8 | 4.9-5.4 | 2.5-2.7 | 126-172 s | 0.58-0.64 | 1.1-3.5 % | 1·10⁻³ → 5.2·10⁻³ |

- Plateau since ≈ 960 k: r 2.5-2.9·10⁻³, r_hit 4.8-5.9·10⁻³, kill 19-21, hit 0.35-0.44. r record at 1 274 k (3.1·10⁻³) without follow-up. The 460-960 k r_hit slope (≈ 0.5·10⁻³ per 100 k) stopped: marker "r_hit > 5.5·10⁻³ and kill > 20 with hit ≥ 0.38" hit only in isolated slices (1 240-1 320 k at hit 0.35-0.38).
- Emptyings rise slightly: one per 90-135 s per tank vs 150-215 s at 800-880 k. Still far from dump (9-10 s at run start).
- **α_fire leaves the floor**: 1.0·10⁻⁴ to 1 040 k, then steady exponential growth, ×50 in 420 k (5.2·10⁻³ at 1 454 k, run max, still rising). Same window CSV fire 0.16-0.17 (1 100-1 180 k) → 0.41-0.50 (1 420-1 482 k). Signature of the 510/514 regulator (α_fire ×20 then trigger released), never shown by 520.
- Optimisation: skip slowly rises (0 → 3.5 %, max 1 440 k), kl 0.03-0.09 around target, ev 0.55-0.65 (vs 0.70 at 800-880 k), an ev hole at 940 k (−4.0 on one row). Hc oscillates −0.24 to +0.11 no drift. α_dir_x back to floor: dir_x saturation at save_32 was transient.

Probes save_40/48/54 (details probes/timeline.md):

| 550 / 1000 | shots/s | hit/shell | hit < 5° | ratio | p_off | disarmed | tour<5° | touches/tank-s |
|---|---|---|---|---|---|---|---|---|
| save_32 (864 k) | 0.43 / 0.13 | 35.2 / 36.6 % | 52 / 54 % | 12 / 57 | 0.020 / 0.005 | 0.6 / 0.4 % | 28 / 21 % | 0.152 / 0.046 |
| save_40 (1 080 k) | 0.49 / 0.17 | 36.3 / 40.0 % | 47 / 55 % | 16 / 46 | 0.015 / 0.006 | 1.6 / 0.6 % | 27 / 19 % | 0.178 / 0.068 |
| save_48 (1 296 k) | 0.57 / 0.28 | 34.9 / 35.9 % | 45 / 47 % | 15 / 36 | 0.020 / 0.009 | 0.8 / 1.0 % | 37 / 25 % | 0.199 / 0.101 |
| save_54 (1 458 k) | 0.57 / 0.30 | 37.0 / 34.2 % | 49 / 47 % | 12 / 32 | 0.025 / 0.013 | 1.2 / 0.4 % | 34 / 26 % | **0.211 / 0.101** |
| save_54 det | 0.43 / 0.24 | 40.3 / 39.8 % | 54 / 54 % | 11 / 21 | 0.027 / 0.011 | 0.5 / 0.1 % | 29 / 27 % | 0.173 / 0.095 |
| 520 save_32 | 1.06 / 0.53 | 26.0 / 27.0 % | 40 / 40 % | 7.5 / 17 | — | 3.1 / 2.3 % | 29 / 25 % | 0.276 / 0.142 |

Established (13:15):
- Success criterion set at 894 k reached at save_54: touches/tank-s 0.211/0.101 (threshold 0.20/0.10) at hit/shell 37/34 % (threshold 33 %). Touch throughput +39 % side 550 and ×2.2 side 1000 since save_32, monotone over 4 saves, at constant per-shell precision (35-40 % on the 8 points).
- Side-1000 weakness resolves via cadence: 0.13 → 0.17 → 0.28 → 0.30 shot/s, p_fire at 0-2° 0.075 → 0.15. 522 at 76 %/71 % of 520 save_32's touch throughput (55 %/32 % at save_32), with 2-3× fewer shells fired and 0.4-1.2 % of time disarmed.
- Deterministic no longer a handicap: save_54 det 0.43/0.24 /s (75-80 % of stochastic vs 45-60 % at save_32), hit/shell 40 %, touches/tank-s 0.173/0.095 = ×1.9/×2.7 vs save_32 det.
- dir_x saturation gone (mean μ −1.4 → −0.1, saturated share 45-54 % → 8-10 %) and zoom back (p_zoom 0.09-0.16 → 0.34-0.51, 0.69 when target at 0-2°): both save_32 alerts were oscillations.
Signal to watch: off-axis rises side 1000 since save_32: p_off 0.005 → 0.006 → 0.009 → 0.013, p_fire at 45-180° 0.004 → 0.011, ratio 57 → 46 → 36 → 32. Cadence rise from save_40 to save_54 comes partly from a raised floor, not only the axis (0-2° flat at 0.15 between save_48 and save_54). Coincides with α_fire rising. Per-shell precision not clearly hurt yet (hit < 5° 54-55 % → 47 % side 1000; 15 % of shells leave at ≥ 20° and never hit, stable share).
Noise: hit/shell 34-40 % save to save, tour<5° 27-37 % side 550 (4 episodes per point); absolute side-1000 ratio (310-370 aligned-and-close decisions per probe): only its slope over 4 saves counts.
Not settled: whether α_fire keeps rising and releases the trigger as 510/514 (p_off ×4-6 in one save) or stabilises (target H(0.01) = 0.056 same as 520 which never left the floor: 522's trigger is more closed); whether the CSV plateau since 960 k is a true ceiling (probes still show progress s40 → s48, nothing clear s48 → s54).
Markers: release = side-1000 p_off > 0.03 or disarmed > 3 %, CSV hit < 0.33, emptyings more than one per 60 s. Progress = touches/tank-s > 0.23/0.12 at hit/shell ≥ 33 %.

---

# Update 2026-11-16 16:15: 1 725 k, 19.8 h, 64 saves

| slice | fire/s | hit | kill | r_hit (·10⁻³) | r (·10⁻³) | one emptying every | ev | skip | α_fire |
|---|---|---|---|---|---|---|---|---|---|
| 1 460-1 520 k | 0.35-0.63 | 0.33-0.37 | 20.3-21.3 | 5.3-5.8 | 2.5-2.6 | 83-97 s | 0.46-0.70 | 2.4-3.8 % | 5-7·10⁻³ |
| 1 520-1 600 k | 0.31-0.35 | 0.38-0.44 | 21.2-22.3 | 5.3-6.2 | 2.75-3.2 | 116-173 s | 0.55-0.66 | 2.8-5.7 % | 0.8-1.3·10⁻² |
| 1 600-1 725 k | 0.17-0.25 | 0.38-0.39 | 18.6-20.5 | 4.9-5.9 | 2.5-3.0 | 112-137 s | 0.53-0.66 | 2.3-6.8 % | 1.7-3.3·10⁻² |

- The 1 440-1 500 k cadence peak did not hold: fire 0.63 then back to 0.2/s, emptyings back from 83 s to 130 s. No dump.
- r record at 1 600 k (3.19·10⁻³, r_hit 6.2·10⁻³, kill 22.3), then back to plateau (2.5-2.7·10⁻³); 960 k plateau holds with ±15 % oscillations.
- α_fire keeps rising (3.3·10⁻² at 1 720 k, ×330 from floor) without releasing the trigger: fire falls, Hd falls (0.15 → 0.115), hit holds. Unlike 510/514 the regulator pushes without visible effect at this stage.
- skip 2-7 % (max 6.8 % at 1 620 k), kl 0.03-0.07, Hc −0.28 to +0.08.

Probes save_59 / 63:

| 550 / 1000 | shots/s | hit/shell | hit < 5° | ratio | p_off | disarmed | tour<5° | touches/tank-s |
|---|---|---|---|---|---|---|---|---|
| save_54 (1 458 k) | 0.57 / 0.30 | 37.0 / 34.2 % | 49 / 47 % | 12 / 32 | 0.025 / 0.013 | 1.2 / 0.4 % | 34 / 26 % | 0.211 / 0.101 |
| save_59 (1 593 k) | 0.48 / 0.29 | 38.3 / 38.6 % | 51 / 52 % | 16 / 31 | 0.018 / 0.008 | 0.8 / 0.4 % | 31 / 25 % | 0.182 / 0.111 |
| save_63 (1 701 k) | 0.32 / 0.22 | 42.1 / 41.1 % | 53 / 53 % | 16 / 35 | 0.011 / 0.007 | 0.4 / 0.7 % | 31 / 23 % | 0.133 / 0.092 |
| save_63 det | 0.26 / 0.16 | 46.5 / 46.8 % | 57 / 57 % | 15 / 31 | 0.013 / 0.008 | 0.3 / 0.1 % | 32 / 23 % | 0.121 / 0.076 |

Established: the feared release did not occur. Side-1000 off-axis fell back (p_off 0.013 → 0.008 → 0.007, 45-180° 0.011 → 0.005) while α_fire ×6 between save_54 and save_63; the save_32 → save_54 rise was an oscillation, not the onset of 510/514's mechanism. The trigger closes instead: p_fire at 0-2° 0.20 → 0.11 side 550 and 0.15 → 0.10 side 1000, whole curve ÷ ≈ 2 at constant shape (ratio 16/31-35). Run moves along a cadence/precision trade-off: save_54 = touch-throughput max side 550 (0.211), save_59 side 1000 (0.111), save_63 = precision max since save_16 (42/41 %, 46.5 % det) for 37 %/9 % less touch throughput than save_54. Aim does not move: tour<5° 31 %/23-25 %, hit of aligned shots 51-53 %, median miss 8-9 m. Since save_24 all movement is in the trigger.
Noise: r record at 1 600 k: one 20 k slice then back to plateau.
Not settled: why α_fire rises without effect (target H(0.01) = 0.056 stays above this trigger's entropy, empty penalty keeps it closed; at α_fire = 3·10⁻² the bonus has no hold yet; 510 released at 1.1·10⁻²); whether save_63's closing is a new transient trough (like save_16) or drift.
Options noted: stop (760 k since 960 k oscillating around same reward; useful saves save_54 throughput, save_59 balanced / best side 1000, save_63 precision incl. det) or continue to see if α_fire finally acts. Markers: release = p_off side 1000 > 0.03, disarmed > 3 %, hit CSV < 0.33; closing = CSV fire < 0.15 lasting with kill < 17.

---

# Update 2026-11-16 17:17: 1 837 k, 21.2 h, 69 saves (CSV only)

| slice | fire/s | hit | kill | r_hit (·10⁻³) | r (·10⁻³) | one emptying every | Hd | α_fire |
|---|---|---|---|---|---|---|---|---|
| 1 700-1 740 k | 0.21-0.23 | 0.36-0.39 | 19.9-20.2 | 5.1-5.3 | 2.5-2.7 | 112-127 s | 0.12-0.13 | 3.4·10⁻² (max) |
| 1 740-1 800 k | 0.27-0.35 | 0.34-0.35 | 20.6-21.6 | 5.3-6.1 | 2.45-2.7 | 78-94 s | 0.15-0.17 | 3.3 → 1.9·10⁻² |
| 1 800-1 837 k | 0.25 | 0.38-0.39 | 20.7 | 5.2-5.6 | 2.7 | 100-139 s | 0.15-0.17 | 1.65 → 1.5·10⁻² |

- α_fire finally acted, weakly, then reversed: max 3.4·10⁻² at 1 740 k, then Hd 0.115 → 0.165, fire 0.21 → 0.35, hit 0.39 → 0.34, emptyings 127 → 78 s over 1 740-1 800 k; α_fire then falls (1.5·10⁻² and falling), hit back to 0.39 and emptyings to 139 s. The regulator found an equilibrium with the empty penalty without dump: a release of a few percent, not 510/514's.
- r did not move: 2.45-2.70·10⁻³ over the window, as since 960 k. The release traded precision for volume at equal reward (r_hit 6.1·10⁻³ at hit 0.34, compensated by emptyings).
- skip 1.7-3.2 %, kl 0.04-0.07, ev 0.49-0.71.
Open question at 1 725 k ("will α_fire act higher up") settled: yes, at ≈ 3·10⁻², bounded reversible effect. No mechanism in progress can take the run off the plateau: aim flat since save_24 and trigger oscillating around the same point for 880 k of compute.

---

# Update 2026-11-16 18:30: save_68 and oracle plan (probes/oracle_save_68.md)

- save_68 (1 836 k): hit/shell 46/45 % (51 % deterministic), aligned shots hit 60-62 %, touches/tank-s 0.150/0.110, disarmed < 1 %. Highest precision of the run with save_16.
- Oracle plan: full oracle 0.533/0.448 touches/tank-s (×3.5/×4); oracle aim with agent trigger 0.147/0.059; oracle trigger with agent aim 0.128/0.087. No single link pays.
- The trigger does not read alignment: at < 2° on a target at 50-100 m (80 % hit), p_fire 0.20 with its own aim, 0.06 under oracle aim; aligned at < 100 m without recent shot 0.08 (0.016 under oracle), within the second after a shot 0.16-0.23.
- Plateau = the aim/trigger local optimum of coordination already measured on 520, shifted toward precision by the empty penalty.

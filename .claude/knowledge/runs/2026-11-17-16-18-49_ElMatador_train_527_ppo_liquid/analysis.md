# Run 527 — ppo_liquid

Launched 2026-11-17 16:18, read at 16:42, **running**. Index 28 k (31 rows), D 250, 73.7 k steps/h (2.36 M tank-steps/h), save_1 written 16:40.
Code: uncommitted tree on `68c41e20`, binary `cmake-build-release` compiled 16:18:48 (one second before launch) ⟹ contains the kills/tank curriculum (confirmed by `curriculum_ratio_low/high` = 0.3 / 0.5 in `config.json`). High confidence.

## What the run tests

Planned (memory note 2026-11-17): two confounded levers, kills/tank curriculum **and** `--shells_per_hit 5`. Actually launched:

| | 526 | 527 |
|---|---|---|
| `actor_batch_size` | 2400 | 512 (back to 525 default) |
| curriculum criterion | touches/shots 0.05 / 0.08 | kills/tank 0.3 / 0.5 |
| `shells_per_hit` | 2 | **2** |

The process command line has no `--shells_per_hit` and `config.json` gives 2: **the economy lever is not in this run**. 527 = 525 + new curriculum criterion alone. While D = 250 the curriculum decided nothing, so 527 is so far a **seed replica of 525** (same reward code, same config).

## CSV (D 250 throughout)

| index | run | kill | hit | fire/s | r_hit | Hc | kl | clip | ev |
|---|---|---|---|---|---|---|---|---|---|
| 10 k | 527 | 10.0 | 0.070 | 0.79 | 3.1e-3 | 0.19 | 0.021 | 0.26 | 0.56 |
| 28 k | **527** | **12.2** | **0.093** | 0.85 | 3.9e-3 | 0.36 | 0.028 | 0.26 | 0.64 |
| 28 k | 525 | 8.3 | 0.071 | 0.77 | 3.0e-3 | 0.22 | 0.036 | 0.33 | 0.38 |
| 28 k | 524 | 11.8 | 0.078 | 1.12 | 3.5e-3 | 0.31 | 0.029 | 0.29 | 0.59 |
| 28 k | 526 | 13.3 | 0.075 | 1.69 | 4.4e-3 | −0.10 | 0.008 | 0.13 | 0.46 |

- Steady climb: kill 3 → 12.3, hit 0.017 → 0.093, r 0 crossed at ~10 k, r_hit +50 %.
- fire stable 0.65-0.96/s, no sign of extinction; skip 0, kl 0.03 (target 0.10), clip 0.26.
- Hc rises 0.13 → 0.36 (σ widens), α in free decay toward floor (2e-4): regulator inert.
- ev 0.5-0.6 with an isolated trough at −0.51 (17 k), no consequence.

## Diagnosis

Established: `shells_per_hit` = 2: break-even stays 50 % (point blank), the run does not test the "open the trigger at 50-100 m" hypothesis. D stuck at 250 is not an anomaly: the criterion requires > 0.5 kill/tank = 16 kills per episode over 8 probe episodes, CSV is at 12.2 (0.38/tank, in the 0.3-0.5 dead band). At D = 250 all episodes are the same side, so CSV `kill` is what the criterion reads. Under the old criterion 524-526 all passed to D 300 at 47.6 k with kill 16 / 12.9 / 18.
Noise: 527 above 525 at equal code (kill 12.2 vs 8.3, hit 0.093 vs 0.071 at 28 k): seed-noise amplitude at this stage, not an effect; the early 524/525 gap is too.
Not settled: exit date from D 250 (kill ≥ 16 reached at 44 k (524), 39 k (526) but 141 k (525); at the current slope (+0.3 kill per 2.8 k) ~60-70 k, ≈ 17:10-17:20); all that motivates the run (trigger at range, kills when D rises) reads only at D ≥ 600 and by probe in distance bands.
Options posed to Samuel: (1) let it run: clean test of the kills/tank criterion alone, comparable to 525 at equal D; (2) stop and relaunch with `--shells_per_hit 5`: the two levers confounded as planned (stop cost 25 min); (3) let it run and launch the 5-shell run later to separate the two effects.
Markers: D 300 expected at 40-70 k; if D still 250 at 150 k with kill 12-15 the 0.5 threshold is too high for this regime. Probe save_5 (~135 k) by distance band: touches/tank-s and aligned hit at 50-100 m, aligned p_fire 50-100 m vs < 50 m (525 s9: 0.14 vs 0.38), empty reserve and disarmed. CSV at D 600-800: 525 collapsed (kill 16 → 8, fire 0.34 → 0.16).

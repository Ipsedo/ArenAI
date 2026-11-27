# 522 save_68 (1 836 k, D 1000): final probe and oracle plan

Binary `cmake-build-release` of 2026-11-15 20:07 (commit 9d843717, no inference diff). 4 episodes, 32 tanks, economy 10/2/30, stochastic unless stated. Oracle: `--oracle --oracle_fov 80`, tolerance 1°. Letters: aim (yaw + site), trigger; A = agent, O = oracle.

## Final probe (550 / 1000)

| save | shots/s | hit/sh | hit < 5° | ratio | p_off | disarmed | tour<5° | touches/tank-s |
|---|---|---|---|---|---|---|---|---|
| 54 | 0.57 / 0.30 | 37.0 / 34.2 % | 49 / 47 % | 12 / 32 | 0.025 / 0.013 | 1.2 / 0.4 % | 34 / 26 % | 0.211 / 0.101 |
| 63 | 0.32 / 0.22 | 42.1 / 41.1 % | 53 / 53 % | 16 / 35 | 0.011 / 0.007 | 0.4 / 0.7 % | 31 / 23 % | 0.133 / 0.092 |
| 68 | 0.32 / 0.25 | 46.0 / 44.6 % | 62 / 60 % | 12 / 19 | 0.014 / 0.009 | 0.1 / 0.6 % | 32 / 26 % | 0.150 / 0.110 |
| 68 det | 0.26 / 0.15 | 50.8 / 51.1 % | 64 / 66 % | 11 / 27 | 0.017 / 0.008 | 0.1 / 0.0 % | 37 / 29 % | 0.131 / 0.075 |

save_68 = highest precision of the run with save_16, at double save_16's cadence. Touch throughput between save_63 and save_54 side 550, equal to best side 1000.

## Oracle plan

| arm (550) | shots/s | hit/sh | touches/tank-s | disarmed | turret < 5° | p_fire aligned < 2° |
|---|---|---|---|---|---|---|
| agent alone | 0.33 | 46.5 % | 0.152 | 0.2 % | 32 % | 0.115 |
| oracle trigger, agent aim | 0.18 | 71.1 % | 0.128 | 0 % | 41 % | — |
| oracle aim, agent trigger | 0.25 | 59.6 % | 0.147 | 0.1 % | 93 % | 0.019 |
| all oracle | 1.28 | 41.7 % | 0.533 | 3.9 % | 60 % | — |

| arm (1000) | shots/s | hit/sh | touches/tank-s | disarmed | turret < 5° | p_fire aligned < 2° |
|---|---|---|---|---|---|---|
| agent alone | 0.25 | 44.9 % | 0.111 | 0.3 % | 26 % | 0.119 |
| oracle trigger, agent aim | 0.13 | 65.6 % | 0.087 | 0 % | 25 % | — |
| oracle aim, agent trigger | 0.10 | 58.7 % | 0.059 | 0 % | 97 % | 0.008 |
| all oracle | 1.73 | 25.9 % | 0.448 | 12.5 % | 67 % | — |

- Game ceiling: ×3.5/×4 the agent's touch throughput (520 save_37: ×1.8/×2.6, because 520 fired more).
- No single link pays, as in 520: perfect trigger on agent aim −16/−22 %, perfect aim with agent trigger −3/−47 %.
- With perfect aim the agent's trigger fires less: turret aligned 93-97 % of time, aligned p_fire divided by 6 to 15.

## What the trigger reads (armed decisions, aligned < 2°)

At equal angle and distance (target 50-100 m, where shots hit 80 %): p_fire 0.20/0.23 with agent aim, 0.057/0.065 with oracle aim. Share of decisions in this cell rises from 3.2 % to 10.7 % side 550: the opportunity is 3× more frequent and the trigger 3.5× lower.

p_fire aligned < 2° on a target < 100 m, side 550:

| state | agent alone | oracle aim |
|---|---|---|
| within 1 s after an impact that hit | 0.234 | 0.189 |
| within 1 s after a missed impact | 0.162 | 0.149 |
| no recent shot | 0.077 | 0.016 |

- Trigger depends first on having fired recently (×3 with its aim, ×12 under oracle), little on the shot's outcome (hit 0.23, miss 0.16).
- The first shot of an engagement is the bottleneck: aligned < 2° on a target < 100 m without recent shot, p_fire = 0.08 per decision, 0.016 under oracle aim.
- The agent's aim command does not explain the gap: at |μ canon| < 0.1, p_fire 0.20 (agent) vs 0.09 (oracle).

## Opportunity balance (save_63, repeated at save_68)

- Aligned shots < 2°: 90-95 % touches at < 50 m, 68-80 % at 50-100 m, 43-59 % at 100-200 m. Each shot at < 100 m returns more shells than it costs (2 × 0.7 > 1).
- 20-27 % of shells leave at ≥ 5° at a target > 100 m and hit at 0-7 %.
- Nearest enemy < 100 m in 23-38 % of decisions; in field ±40° only 38 % of time (chance 22 %), median bearing 76-82°, aligned < 5° 15-21 %. Same figures as the 2026-11-12 diagnosis on 504/507.

Reading: 522's plateau is the same local optimum as 520, shifted toward precision by the empty penalty: the trigger does not fire on visible alignment but on an engagement state sustained by its own shots, aim rarely aligns finely, and each link is tuned to the other's defect. Not settled: why the first shot is 5× rarer under oracle aim (alignment cue specific to the agent's aiming dynamics, or targets chosen differently by the oracle).

# train_428_ppg — CSV analysis

- Algorithm: PPG, 32 tanks. Launched 2026-11-15 at 00:01. Analysed at 10:44, run **running**, index 739k.
- Throughput: 69 k steps/h (2.21 M tank-steps/h). 427 ran at 74 k steps/h.
- Effective commit: **0e5093a1**, high confidence. The binary `cmake-build-release/.../arenai_agent_train` was compiled at 00:01:45, i.e. at launch. Commit 9d843717, which arrived during the run, touches only the desktop preset and the shipped models.

## What this run tests (vs 427, d1bb4020)

| change | 427 | 428 |
|---|---|---|
| `discrete_target_proba` (fire/zoom entropy target) | 0.05 (H ≈ 0.199 nat) | **0.01** (H ≈ 0.056 nat) |
| `gamma` | 0.995 (horizon ≈ 20 s) | **0.998** (horizon ≈ 50 s) |
| `ev` | computed per GPU block (256 rows) | computed per optimisation batch (512 rows), still in-sample |

Commits c33e68de/4c099cb6 (agent-side zoom removal, 512×256 vision, extra CNN block) were entirely reverted: the net diff excluding comments contains only the three lines above. **The γ change and the discrete-target change are confounded.**

Inherited from 427: aim potential as ratio `miss / AIM_TARGET_RADIUS`, canon action mapped by a·|a|, economy 10 / +3 / 30.

## CSV reading

**Curriculum.** D milestones are identical to the index for 424 to 428 (300 @ 21k … 1000 @ 429k). The curriculum thus climbs at maximum speed in all these runs, and these milestones cannot tell them apart. Comparison is at D = 1000, at equal index.

| D = 1000, index | run | hit | fire (/s) | kill | s (s) | miss (m) | Hd | ev |
|---|---|---|---|---|---|---|---|---|
| 429–530k | 423 | 0.194 | 0.84 | 21.3 | 48 | 64 | 0.41 | 0.74 |
| | 427 | 0.137 | 1.17 | 22.4 | 44 | 81 | 0.55 | 0.73 |
| | **428** | 0.174 | 0.79 | 22.0 | 45 | 76 | 0.46 | 0.76 |
| 530–640k | 423 | 0.204 | 0.70 | 21.2 | 50 | 62 | 0.41 | 0.75 |
| | 427 | 0.165 | 0.70 | 21.9 | 50 | 77 | 0.45 | 0.78 |
| | **428** | 0.192 | **0.39** | 18.7 | 53 | 68 | 0.35 | 0.85 |
| 640–740k | 423 | 0.203 | 0.83 | 23.1 | 49 | 67 | 0.36 | 0.73 |
| | 427 | 0.147 | 1.09 | 23.4 | 44 | 89 | 0.52 | 0.71 |
| | **428** | **0.232** | 0.44 | **16.9** | 51 | **54** | 0.35 | 0.82 |

**Trajectory.**
- **Up to 530k**, 428 closely follows 427. Only hit is a bit above (0.17 vs 0.14).
- **Switch around 560–620k.** Fire cadence drops from 0.9–1.0 to 0.18–0.25 /s (minimum 0.084 @ 709k). At the same time hit rises: 0.19 → 0.26 @ 724k, record of the lineage. Miss falls from 76 to 49–55 m.
- In exchange, kill goes from 22 to 14–17 per episode and r from 4.1e-3 to 2.3–3.3e-3.
- This « few shots, more precise, fewer kills » regime resembles 425's (economy 2/20) at D = 950: hit 0.23, fire 0.43, kill 14.5.
- **Discrete α.** `alpha_fire` and `alpha_zoom` stay at the floor (1e-4) from 64k and all run (max 1.02e-4 after 100k). In 427, `alpha_fire` rose to 0.017 and `alpha_zoom` was 3e-3 at end of run: discrete entropy fell under the H(0.05) target. With the H(0.01) target, the constraint is **never active** in 428, and nothing holds the trigger back from becoming nearly deterministic.
- **Continuous α**: active between 1e-3 and 0.03, like 427. Hc stays around 0 (trough −0.5 @ 648k), without collapse.
- **Critic**: ev from 0.82 to 0.85 beyond 560k, above 427 (0.71–0.78). Not strictly comparable (ev computed on a larger batch, and different γ). There is a spike to −1.86 @ 512k, a one-off artefact already seen in 427 at 1.17M.
- **Optimiser**: clip ≈ 0.2, kl ≈ 0.03, skip ≤ 0.07 and skip_canon ≤ 0.02. Nothing abnormal.

## Diagnosis

**Established**
- The discrete entropy constraint is inert in 428 (α at floor), while it was intermittent in 427.
- The drop in fire cadence and the rise in hit/shot arrive together, around 560–620k, at constant D (1000). It is therefore not a curriculum effect.
- At 640–740k, hit (0.23) and miss (54 m) are the best of the lineage at equal index. kill is the lowest (16.9 vs 23 for 423 and 427).

**Undecided**
- **What causes the switch.** Two explanations are possible:
  - the trigger freed from its entropy target becomes selective;
  - with γ = 0.998, fire cost and reserve (horizon 50 s) are valued more, and the policy fires less everywhere.

  The two changes are confounded, and the CSV does not measure selectivity.
- **The hit/shot gain.** It may come from better aim or merely from selecting easy shots (fewer shots, so the remaining shots are the most favourable). The falling miss goes in the direction of selection, without settling.
- **What kill becomes.** Fewer kills with a higher hit indicates globally more passive play. Not necessarily a competence regression (the game is self-play), but the « in-game kills » objective recedes.

## Options (no decision)

1. Probe `save_27` (729k): look at trigger, reserve and alignment. See `probes/save_27.md`.
2. Probe a save from before the switch (≈ `save_19`, 510k) to compare selectivity before and after at the same D.
3. To separate the two causes: a run 427 + γ 0.998 alone, or 427 + target 0.01 alone.
4. Let it run to see whether kill rises back at low cadence, or whether fire keeps heading to 0 (« extinguished trigger » regime of 392).

**Markers**: fire < 0.15 /s sustained = extinguished-trigger alert. kill < 12 = regression. Hit/shot stays to be read next to fire.

## Probe save_27 (729k) — see `probes/save_27.md`

The trigger is selective: ratio 8.6 (side 550) and 21.9 (side 1000), vs 4.9 and 7.5 in 423 save_21. This gain comes from the extinction of off-axis shots (p_off 0.017 / 0.007), but the aligned-fire probability is halved (0.15).

Hit/shell rises to 0.27, while the hit/shell of aligned shots stays at 0.35: the gain comes from the composition of the shots, not their precision. The reserve is emptied much less (39–49 % of tank-episodes vs 70 %). In deterministic mode, the agent fires in 4–7 % of aligned states: nearly mute in game.

## Probe save_49 (1323k) — see `probes/save_49.md`

Precision per aligned shot rises to 0.43–0.47, above the 0.33–0.37 of the whole lineage, and turret alignment progresses. Zoom is abandoned (p_zoom 0.02–0.03), matching Hd ≈ 0.08 in the CSV from 1.18M. The chassis turns in circles, steered to one side.

The trigger is less sharp in angle (ratio 7.0 / 11.1, flat off-axis floor beyond 10°) and still mute in deterministic mode (3–7 % of aligned states).

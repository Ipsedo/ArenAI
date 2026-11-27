# 556 save_24 — wreck vs living tank with hatched wrecks (E1 of plan wreck-rendering, 2026-11-27)

Evaluator E1 of `.claude/plans/2026-11-27_wreck-rendering.md`. Same protocol as `save_24_wreck_decoding.md`
(probe side 1000, 8 episodes, 32 tanks, deterministic, features + vision frames at every decision, classes,
train episodes 0-5 / test 6-7, balanced accuracy, chance 0.50), binary = `c6177f2a` (feat(view): draw dead tanks as hatched wrecks) with the hatched wreck rendering (band light 0.2, period = render height / 32 = 4 px on the vision).
The checkpoint's network only produces the trajectories; it was trained without hatching.
Data: `outputs/train_556_ppo_liquid/features/save_24_side1000_det_hatched.*` (frames deleted, 22.7 GB). Probe 4 min 21 s, decoding 1 min 35 s.

| target distance | trunk h | vision features | pixels CNN (2 seeds) | before (pixels CNN) | control: tank on axis vs nothing (vision / CNN) | n test living / wreck |
|---|---|---|---|---|---|---|
| < 60 m | 0.66 | 0.74 | **0.93 / 0.97** | 0.59 / 0.60 | 0.92 / 0.96 | 547 / 635 |
| 60-100 m | 0.70 | 0.63 | **0.98 / 0.99** | 0.60 / 0.60 | 0.90 / 0.96 | 749 / 888 |
| 100-200 m | 0.62 | 0.58 | **0.94 / 0.94** | 0.69 / 0.66 | 0.79 / 0.92 | 605 / 781 |
| 200-400 m | 0.58 | 0.52 | 0.61 / 0.61 | 0.58 / 0.58 | 0.67 / 0.77 | 689 / 515 |

Shuffled labels: 0.47-0.51 everywhere. Trigger on the test samples (armed): living 0.53 / 0.49 / 0.16 / 0.04,
wreck 0.53 / 0.66 / 0.28 / 0.06.

## Verdict against the decision rule

- ≥ 0.90 under 200 m: **met** (0.93-0.99, from 0.59-0.69). Refutation threshold (< 0.80 under 100 m): not hit.
- ≥ control at 200-400 m: **not met** (0.61 vs 0.77). At that range a tank covers a few pixels of the 128 px
  vision; 2 px bands do not survive there.
- The untrained network's vision features barely move (0.52-0.74): expected, the encoder never saw hatching;
  only a training (piece P4, G4) can say whether the agent learns to use it.

The information is now in the input under 200 m, where 556 s24 fires at wrecks most (0.37 / 0.47 / 0.17 of
decisions before). The trigger of this untrained checkpoint still fires at wrecks as often as at living tanks.

## P4 — training with hatched wrecks (G4, validated 2026-11-27, Samuel launches it)

Hypothesis: with wrecks visible, the agent stops firing at them. Reference 556, same config, binary ≥ `c6177f2a`.
Decision rule at the index of 556 s24 (648 k): `save_K_wrecks` probe, side 1000, deterministic — firing with a wreck
on the axis at < 60 / 60-100 / 100-200 m drops from 0.37 / 0.47 / 0.17 to ≤ half, firing on a living enemy not
below 0.74 / 0.62 / 0.21; game match side 1000 — tanks dry at 10 s from 24-27 % to ≤ 10 %. Stop at 648 k.

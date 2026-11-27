# Supervised CNN bench: 522 save_68 images

Question: is the spatial softmax suitable? Which CNN localises an enemy tank in the game image?

## Method

- Export: `arenai_agent_probe --frames <bin> --frames_every N` (option added 2026-11-16), save_68 policy, side 550, zoom off (`--zoom off`, fixed field). 40 episodes, one image every 3 s: 36 k images 128 × 256, 40 colour sets of tanks and terrain.
- Label: yaw and site, in cannon frame, of the enemy most on the axis (`best_yaw`, `best_pitch`, added to `AimGeometry`), kept if in field (|yaw| < 39.5°, |site| < 22.5°) within 200 m: 13.6 k images. Number of enemies in exported field (`view_count`): median 3, a single one in 21 % of images. Label without occlusion test.
- Supervised training outside the agent (`bench_curve.py`, torch GPU): CNN + head (linear 128, LayerNorm, SiLU, linear 2), Huber loss, AdamW, 20 passes at 32 episodes (constant step count at smaller sizes). Train on 8, 16 or 32 episodes, test on 8 unseen episodes (2 768 images).
- References on test: yaw std 10.8°, median error without looking at the image (predict 0) 3.7°, left/right chance 0.50.

## Results (test, yaw)

| CNN | episodes | median error | left/right AUC | idem, target < 100 m | idem, single enemy | corr pred/true | median error, target < 100 m |
|---|---|---|---|---|---|---|---|
| A current: 4 convs + spatial softmax | 8 / 16 / 32 | 4.9 / 5.7 / 5.9° | 0.51 / 0.51 / 0.52 | 0.48 / 0.50 / 0.49 | 0.54 / 0.51 / 0.52 | +0.02 / 0.00 / +0.02 | 4.0 / 4.8 / 5.0° |
| B 4 convs + flatten 8×16 | 8 / 16 / 32 | 4.4 / 4.5 / 4.7° | 0.56 / 0.57 / 0.62 | 0.64 / 0.65 / 0.66 | 0.56 / 0.58 / 0.62 | +0.07 / +0.07 / +0.12 | 3.2 / 3.3 / 3.5° |
| C 6 convs + flatten 2×4 | 8 / 16 / 32 | 4.2 / 4.0 / 3.8° | 0.58 / 0.59 / 0.66 | 0.63 / 0.66 / 0.70 | 0.56 / 0.61 / 0.63 | +0.07 / +0.09 / +0.18 | 3.2 / 2.8 / 2.6° |
| D 7 convs + flatten 1×2 | 8 / 16 / 32 | 3.9 / 3.9 / 3.6° | 0.57 / 0.61 / 0.68 | 0.65 / 0.72 / 0.73 | 0.57 / 0.62 / 0.68 | +0.05 / +0.12 / +0.19 | 2.7 / 2.7 / 2.4° |

All memorise their training images (median error 0.0-0.7°).
First pass (label "nearest enemy", 32 episodes, `bench_cnn.txt` for 6 episodes): same ranking, A 0.53, C 0.59, D 0.59 left/right; spatial-softmax variants (16×32 map, coordinates as input) at A's level; coordinates as input + flatten at C's level.

## Established

1. Spatial softmax does not learn to localise, even supervised directly: left/right 0.51-0.52 whatever the data, zero correlation, worse than not looking at the image. It memorises training and generalises nothing. Both variants (finer map, coordinates as input) do not save it.
2. Flattened outputs learn and improve with data: D goes 0.57 → 0.68 (0.73 for a target < 100 m) from 8 to 32 episodes, no sign of plateau.
3. More stride-2 convolutions does better: D (1×2 map) > C (2×4) > B (8×16). D is the form closest to the pre-2026-10-20 CNN.
4. Ambiguity between several enemies is not the cause: single-enemy images are not easier (0.68 for D).
5. The task is hard in itself: with 10.8 k images the best CNN stays far from 1-2° (median error 2.4° under 100 m vs 3.7° without looking). Random colours, small tanks, agent's tank at image bottom-center.

## Limits

10.8 k training images, 20 passes, single seed, hyperparameters untuned per architecture; D's curve still rising (level with much more data not measured); supervised ≠ RL: ranking may change with a weaker signal, but a CNN that does not generalise with the exact label has no reason to without it.

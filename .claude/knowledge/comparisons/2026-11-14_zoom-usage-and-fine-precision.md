# Zoom: usage, abandonment, and what it says about fine precision

Runs: 510, 514, 515.
Date: 2026-11-14. Side-550 probes of 510, 514 and 515 (armed decisions, step ≥ 10). The real FOV is not logged by the probe: it is simulated from p_zoom, drawing the zoom according to p_zoom and applying the game's transition step.

## Mechanics (code)

- `CanonItem::fov()` advances the FOV by a step `(45° − 23.3°) · (1/30) / 0.25 s` = 2.9° **at each call**. The call comes from the renderer.
- In training and in the probe, vision is rendered only **once per decision** (`render_frame` after the `action_repeat` frames). The transition thus takes 7.5 decisions = **0.75 s**, versus 0.25 s in the game, which renders every frame.
- The zoom is drawn at each decision (Bernoulli p_zoom). At p ≈ 0.5, the FOV wanders between 23 and 45° (simulated median FOV 26-42° depending on the save).
- **The current FOV is not in the observation**: neither in proprioception nor as network input. The agent does not know the pixel → degree ratio of the image it sees, which varies by a factor 2.

## Zoom usage across saves

| run | save | p_zoom | aligned < 5° | off-axis ≥ 20° | corr(p_fire, p_zoom) | zoomed share (FOV < 30°) |
|---|---|---|---|---|---|---|
| 510 | 5 | 0.68 | 0.72 | 0.62 | 0.62 | 88 % |
| 510 | 9 | 0.30 | 0.36 | 0.28 | 0.68 | 16 % |
| 510 | 16 | 0.47 | 0.63 | 0.39 | 0.55 | 43 % |
| 510 | 28 | 0.40 | 0.55 | 0.34 | 0.53 | 35 % |
| 514 | 9 | 0.43 | 0.54 | 0.39 | 0.86 | 25 % |
| 514 | 20 | 0.32 | 0.52 | 0.23 | 0.87 | 32 % |
| 514 | 23 | 0.33 | 0.59 | 0.20 | 0.86 | 34 % |
| 514 | 34 | **0.09** | 0.12 | 0.07 | 0.33 | **2 %** |
| 515 | 9 | 0.49 | 0.62 | 0.44 | 0.90 | 44 % |

- **The zoom is selective.** The agent zooms more when aligned (0.52-0.67) than off-axis (0.20-0.44), and more when the enemy is near. Not noise.
- **The zoom follows the trigger.** In 514/515, corr(p_fire, p_zoom) is 0.86-0.93: the two discrete heads move together. In 510 the correlation is weaker (0.53-0.68).
- **Abandonment is late and specific to 514.** In 514's CSV, Hd drops from 0.33 to 0.15 between 720 k and 1 M steps. alpha_zoom reacts only from 960 k, because the discrete target (0.05) acts only once the probability is very low. 510 keeps p_zoom ≈ 0.40 until save_28. 514's CSV hit does not drop during the abandonment (18-21 %).

## Zoom and precision of aligned shots (< 5°, 50-200 m, target = nearest)

| probe | p_zoom < 0.3 | 0.3-0.7 | > 0.7 |
|---|---|---|---|
| 514 s9 (550 / 1000) | — | 44 / 47 % | 55 / 61 % |
| 515 s9 | — | 28 / 27 % | 49 / 60 % |
| 514 s20 | 22 / 23 % (n 18-30) | 44 / 34 % | 49 / 52 % |
| 514 s34 | 39 / 39 % (n ≈ 1200) | 46 / 48 % | 57 / 40 % (n < 60) |

These figures are correlations. p_zoom follows the states where the target is easier (near, aligned), and the real FOV is unknown.

## Reading

1. **The zoom seems to help fine precision** (+10 to +30 points on aligned shots). The clue pointing toward 512×256.
2. **Shots at intermediate p_zoom are often the worst** (515 s9: 27-28 % versus 49-60 %). At p ≈ 0.5 the FOV wanders and the image scale changes from one decision to the next without the agent knowing it. Hypothesis: the **unknown scale**, more than resolution, costs precision. An error of x pixels at the centre is worth 1× or 2× in angle depending on the FOV.
3. **The 0.75 s transition in training** (versus 0.25 s in game) lengthens the period of uncertain scale. It also makes the effect of a zoom action small (2.9° of FOV) and diluted over 7-8 decisions, hence hard to credit.

## What would settle it

- **Forced-zoom probe** (`--zoom on|off`, to add) on one checkpoint (514 save_23, 515 save_9): aligned hit/shell at fixed FOV 45° versus fixed 23°. That is the causal effect of ×2 resolution at equal policy, hence what to expect from 512×256.
- Design decisions for Samuel, nothing coded:
  - zoom transition per frame rather than per render (training/game alignment);
  - current FOV in the observation;
  - or zoom removed in favour of a higher fixed resolution.

## Result of the forced-zoom probe (2026-11-14, `--zoom on|off`)

Same checkpoint, zoom left to the policy / forced on (FOV 23°) / forced off (FOV 45°). Hit is measured on aligned shots (< 5°) at the nearest target, by distance.

| probe | zoom | hit/shell | 0-50 m | 50-100 m | 100-200 m | median miss |
|---|---|---|---|---|---|---|
| 514 s23, 550 | policy | 22.0 % | 82 % | 55 % | 22 % (n 351) | 6.0 m |
| | on | 20.4 % | 82 % | 50 % | 22 % (n 259) | 6.0 m |
| | off | 20.3 % | 78 % | 48 % | 16 % (n 112) | 6.0 m |
| 514 s23, 1000 | policy | 18.4 % | 82 % | 55 % | 29 % (n 419) | 6.0 m |
| | on | 16.5 % | 78 % | 46 % | 26 % (n 810) | 7.0 m |
| | off | 18.2 % | 85 % | 45 % | 17 % (n 265) | 6.3 m |
| 515 s9, 550 | policy | 17.3 % | 82 % | 51 % | 20 % (n 89) | 5.6 m |
| | on | 16.4 % | 81 % | 51 % | 18 % (n 147) | 5.6 m |
| | off | 12.2 % | 86 % | 41 % | 0 % (n 23) | 5.9 m |
| 515 s9, 1000 | policy | 12.5 % | 79 % | 62 % | 27 % (n 117) | 5.9 m |
| | on | 12.2 % | 78 % | 51 % | 26 % (n 230) | 6.5 m |
| | off | 6.7 % | 84 % | 48 % | 0 % (n 20) | 5.6 m |

### Reading

- **Forced zoom on does not make aligned shots more precise.** Hit by distance is equal or lower than with the policy, and the median miss does not move (6 m). Doubling px/° does not reduce the error of aligned shots at < 100 m, where most hits happen.
- **Forced zoom off costs only beyond 100 m**: 16-17 % instead of 22-29 % for 514. 515 save_9 no longer even fires far (n 20-23, 0 %), and its hit/shell falls 17 → 12 % and 12.5 → 6.7 %.
- **The p_zoom ↔ hit correlation of the first part was confounded by state.** The agent zooms in easy situations (near target, already aligned).
- Conclusion: angular resolution does not limit fine precision below 100 m. It matters only at 100-200 m, which the 2026-11-12 oracle already said (> 200 m). 512×256 would not target the main lock. The error of aligned shots (miss ~6 m, ~50 % at 50-100 m) lies elsewhere: dispersion of μ, command delay, elevation (hausse).
- Limits: a single 4-episode probe per arm. Both forced zooms leave the policy's distribution (514 save_23 zoomed only 34 % of the time), which slightly penalises the "on" arm.

# Probe train_514_ppo_liquid / save_3

- Index 81 k, D = 400 (CSV at 82-84 k: D 400, hit 12.9-13.0 %, kill 24.4-24.5, skip 0, skip_canon 0.20, kl 0.06-0.09).
- Run code: d1bb4020 (centred aim potential `miss / AIM_TARGET_RADIUS`, turret/canon action passed through a·|a|, economy 10/3/30).
- Binary: `build/arenai_agent/arenai_agent_probe`, compiled 14/11 01:13. Script dates it 3cd9e210 by its mtime, but it was compiled from the working tree just before the commit: `aim_action_to_rate` and `aim_rate_to_action` are indeed present in `libarenai_agent.so` and in the probe (nm). Rest of the diff has no inference effect (economy read from config.json, potential = reward only). Launched in `--force`.
- Sides 550 and 1000, 4 episodes, stochastic, economy 10/3/30.

## Summary

| run | side | p_arm | ratio | hit/sh | h<5° | shots<5° | empty_s | %disarmed | shots/s | turret<5° | σ cx/cy | \|μ\|cx |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **514 s3** | 550 | 0.81 | 1.03 | 7.9 % | 19.0 % | 24 % | 2.3 | 58 % | 1.65 | 20.2 % | 0.20/0.21 | 0.18 |
| 513 s3 | 550 | 0.81 | 1.05 | 10.2 % | 22.1 % | 30 % | 2.3 | 51 % | 1.79 | 20.5 % | 0.17/0.19 | 0.09 |
| 510 s3 | 550 | 0.77 | 1.05 | 10.4 % | 24.2 % | 25 % | 2.5 | 54 % | 1.78 | 20.8 % | 0.19/0.24 | 0.11 |
| **514 s3** | 1000 | 0.81 | 1.03 | 4.5 % | 12.7 % | 22 % | 2.2 | 66 % | 1.49 | 16.7 % | 0.20/0.21 | 0.15 |
| 513 s3 | 1000 | 0.80 | 1.04 | 5.5 % | 13.4 % | 26 % | 2.2 | 67 % | 1.45 | 19.0 % | 0.17/0.20 | 0.08 |
| 510 s3 | 1000 | 0.75 | 1.05 | 5.2 % | 14.0 % | 22 % | 2.3 | 66 % | 1.41 | 16.6 % | 0.19/0.24 | 0.11 |

Aligned and <40 m counts: 494 (550) and 299 (1000), so ratio at 1000 is noisy. No μ saturation, σ never at ceiling.

## Reading

- **Trigger**: flat and in dump, like 510 and 513 at same index. p_fire is 0.80-0.82 over all angle slices (gap 0.02) and exceeds 0.5 everywhere in deterministic. Reserve empty at 2.2-2.3 s, and 96-99 % of tank-episodes empty it. corr(p_fire, p_zoom) 0.92-0.93 vs 0.79 in 513: both discrete heads move together (p_zoom 0.58 vs 0.71).
- **Turret**: 20.2/16.7 % vs 20.5/19.0 (513) and 20.8/16.6 (510). Nothing distinguishes it yet.
- **|μ| canon**: 0.15-0.18 vs 0.08-0.11. Expected with a·|a|, since the same angular rate requires an action of √ norm (0.18 → rate 0.03). Not a sign of agitation.
- **σ canon**: 0.20/0.21, at 510's level. With a·|a|, this σ corresponds to a lower rate noise near 0 (fine control), but the probe does not yet measure the resulting aim precision beyond h<5°.
- **Aligned fire**: h<5° 19/12.7 % and hit/shell 7.9/4.5 %, slightly under 510/513 (−20 % at 550). CSV gives equal hit at equal index (13.0 % vs 13.5 % for 510). Gap stays within known probe noise (±15-35 %).

## Established / noise

- Established: at 81 k, 514 behaves like 510 and 513 at the same point (flat trigger, dump at 2.3 s, turret ~20 %). Neither centred potential nor a·|a| has a visible effect yet on trigger or turret.
- Noise: slight lag in hit/shell and h<5°, and stronger fire/zoom coupling.

## Useful next

- save_5 (~135 k) vs 513 save_5 and 510 save_5: this is where 510 takes off (ratio 1.5).
- save_9 (~243 k): landmark 510 ratio 6.4/10.8, reserve 9.1 s; 513 ratio 3.1/3.7.
- Own effect of a·|a|: look at fine aim error (σyaw, median miss of aligned shots) at saves where turret exceeds 25 %.

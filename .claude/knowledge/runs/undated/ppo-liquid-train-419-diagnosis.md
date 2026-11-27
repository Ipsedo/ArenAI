# PPO liquid train 419 — critic lr 1e-4 (2026-10-04)

Host: ElMatador (memory notes); raw data no longer on disk.
Source note date: 2026-10-04. `train_419_ppo-liquid_critic-lr-1e-4`: only diff vs 418 = `critic_learning_rate` 3e-4 → 1e-4 (commit `e1b51243` « refactor » does not change behaviour: SiLU from lambda to module). 10.4 M steps, save_96.

- Metrics: early lead held longer: D=400 hit 0.089 / kill 9.68 (418: 0.076 / 6.74; 414: 0.076 / 7.15). Then same ceiling ~0.075-0.082 from D=600. Curriculum slower than 414 after D=600 (700 at 4.9 M vs 3.2; 800 at 8.2 vs 4.8). `ev` unchanged (~0.2-0.3), kl/clip as 418.
- Weights (per save, net/path ratio over 15 saves, random = 0.26): critic `to_value` saves 1→16 **0.55** (directed; 418: 0.22) and 11.5 %/save (418: 27 %); but saves 35→50 and 80→95: **0.09-0.11**, 23 %/save — still oscillates later, amplitude ÷ ~2. Actor: **identical to 418** — CNN 0.26 in all phases, `to_output` directed early (0.65) then 0.25. Actor CNN still not specialised at save_95 (γ ≈ 1, uniform features p10/p90 1.55/1.93, no favoured spatial cell).
- Prediction was « if the critic causes diffusion, the actor ratio goes > 0.26 »; it does not move ⟹ actor gradient noise does not come (mainly) from critic oscillation. Hypothesis « critic makes the actor diffuse » refuted.
- Next test proposed 2026-10-04 (not run here): `--minibatch_size 4096` (136 chunks instead of 34 ⟹ per-step noise ÷ 2). The « ~4.5 %/save » of actor 418 was underestimated by √5; true value ~10-11 %/save.
- Samuel's question (2026-10-04): « il faut changer le CNN ? IMPALA ? » Answer given: not first — `mu.0`/`sigma.0`, wired directly to the loss, are also in random walk (0.21-0.24): the noise is in the loss signal, upstream of any CNN. IMPALA never really tested: `train_405_ppo_liquid_impala` = 0.18 M steps only. Class `ImpalaConvolutionNetwork` (vision.cpp) is not standard IMPALA (1 residual block per stage instead of 2, no normalisation, 7 stages) and is no longer wired into LiquidActor/LiquidCritic (removed in `1e31f562`).
- Related: run 418.

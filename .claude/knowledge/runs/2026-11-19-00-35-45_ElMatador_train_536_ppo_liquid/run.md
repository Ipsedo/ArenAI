# train_536_ppo_liquid

| | |
|---|---|
| machine | ElMatador |
| user | samuel |
| launch | 2026-11-19 00:35:45 |
| commit at launch | `cc4ee96b` — add new graphic parameters to GUI |
| commits ≤ 90 min after | none |
| diff at launch | unknown (identity recorded after the run ended) |
| raw data | `outputs/train_536_ppo_liquid` on ElMatador |

## Established facts

- 2026-11-19 ~01:00: Samuel's decision after run 534 (skip KL = μ head growing, ×2 faster since 523): overnight run with three confounded changes vs 534. (1) **Spatial softmax restored as default**, no backward compatibility: `SpatialSoftmax` restored in `networks/misc`, `ConvolutionNetwork` ends on it (output 3 × channels), defaults `vision_channels` 4 blocks / `group_norm_nums` [2,4,8,16] in ppo, ppg, ppo_liquid, sac; constructor parameters `vision_height/width` kept (unused) for the duration of the bet. (2) actor lr back to 1e-4 (default of `liquid_ppo_hyperparams.h`). (3) **a·|a| removed** (Samuel's choice on question: « La retirer »): `aim_action_to_rate` / `aim_rate_to_action` deleted from `torch_types`, probe oracle linear; old `aim.py`: command in ° now 9.2 × tanh(μ), without the |tanh(μ)|. Coloured noise ρ 0.9 unchanged. Samuel's criteria: look at kl and aim; if neither improves, put the flat CNN back and test actor batch 2400.
- 2026-11-19 morning (520 k): **verdict: bet lost on both criteria.** kl twice lower than 534 up to 9 k, caught up at 37 k (skip 0.21), then worse: skip 0.52 at 74 k, 0.84 at 110 k, 0.92-0.97 from 150 k to 520 k (≈ 2-3 actor steps of 76 per rollout). D never left 250, kill 2-8, hit flat 0.07 (534: D 350, kill 20, hit 0.10 at 80 k). μ head (Frobenius) 0.32 at s1 and 0.48 at s3: as large as 523-535, softmax does not hold it; dir_y alone 0.35 → 0.54 at s19. Hc −0.05 → −0.8, α at floor. Not probed. Samuel: « lol ca a pas du tout arreté le skip ». Healthy markers at 20 k / 27 k: kl 0.010 and ‖mu.weight‖ 0.155-0.16 (507/508) vs 0.025-0.028 and 0.245 (523-528); with coloured noise 534 had kl 0.061 and skip 0.29 at 20 k.
- 2026-11-19 (Samuel: « remets cnn plat et le nouveau batch actor, les deux en defaut »): softmax removed (`networks/` files and `vision_channels` defaults restored from HEAD, 7 flat blocks), `actor_batch_size` 512 → 2400 as default in `liquid_ppo_hyperparams.h` only. a·|a| stays removed (not re-requested): the next run differs from 534 by actor batch + a·|a|. The three changes of 536 being confounded, a healthy kl would not say which one acted.

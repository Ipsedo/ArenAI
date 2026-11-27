# Probes 510 save_9 → save_32, run binary (14/11)

- Binary: `arenai_agent_probe` compiled in a worktree at 7422d098 (run commit), same rule and economy as training (respawn at empty, 10/2/20). Script wrongly dates it to ddbaf3da (main repo mtime) ⟹ `--force`.
- Sides 550 and 1000, 4 episodes, stochastic. Outputs: `replay_7422d098/probes/`.

## Summary (550 / 1000)

| save | idx | D | ratio | p_off | p_arm | hit/sh | h<5° | shots/s | empty_s | %disarmed | turret<5° | σ cx | trigger H̄ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 9 | 243 k | 650 | 9.4 / 13.2 | 0.055 / 0.034 | 0.10 / 0.05 | 20.0 / 13.3 % | 38 / 34 % | 0.80 / 0.43 | 8.4 / 17.3 | 13 / 11 % | 18 / 15 % | 0.59 / 0.61 | 0.17 / 0.12 |
| 16 | 432 k | 1000 | 8.6 / 18.5 | 0.051 / 0.023 | 0.13 / 0.05 | **22.8 / 22.4 %** | 34 / 33 % | 1.05 / 0.47 | 9.0 / 13.3 | 8 / 7 % | 28 / 24 % | 0.48 / 0.62 | **0.18 / 0.09** |
| 20 | 540 k | 1000 | **3.9 / 6.2** | **0.20 / 0.14** | 0.34 / 0.20 | 14.9 / 11.7 % | 24 / 20 % | 1.44 / 1.12 | 4.4 / 5.8 | 37 / 28 % | 30 / 27 % | 0.15 / 0.43 | **0.31 / 0.25** |
| 24 | 648 k | 1000 | 4.8 / 7.7 | 0.14 / 0.08 | 0.27 / 0.13 | 20.1 / 16.5 % | 34 / 30 % | 1.61 / 0.94 | 5.3 / 8.9 | 21 / 16 % | 30 / 25 % | 0.22 / 0.60 | 0.29 / 0.19 |
| 28 | 756 k | 1000 | 5.5 / 10.5 | 0.12 / 0.06 | 0.25 / 0.11 | 20.4 / 20.1 % | 34 / 33 % | 1.63 / 0.91 | 5.6 / 9.9 | 14 / 11 % | 29 / 23 % | 0.22 / 0.60 | 0.26 / 0.16 |
| 32 | 864 k | 1000 | 5.6 / 14.1 | 0.12 / 0.05 | 0.24 / 0.10 | 20.5 / 17.6 % | 32 / 30 % | 1.46 / 0.76 | 7.5 / 11.2 | 18 / 11 % | 30 / 25 % | 0.40 / 0.67 | 0.28 / 0.16 |

Trigger H̄ = mean Bernoulli entropy of p_fire over armed decisions (step ≥ 10). Regulator target: H(0.05) = 0.199.

## Reading

- **Aim does not regress.** At fixed distance, h<5° (32-34 %) and turret<5° (28-30 / 23-25 %) stay at save_16's level from save_24 to save_32. Save_20's trough is transient.
- **The trigger does.** Between save_16 and save_20, off-axis fire is multiplied by 4 to 6 (0.05 → 0.20 and 0.023 → 0.14) and cadence by 1.4-2.4. It never returns to save_16's level: p_off stays 2.5× higher, shots/s 1.5-2×, reserve empty 30-40 % sooner.
- **CSV `hit` decline (24 → 18 %)** comes from these extra off-axis shots. The D move to 1000 also contributes, but at fixed distance save_16's hit/shell (22-23 %) is only found again at 550.
- **Probable mechanism: the trigger's entropy regulator.** At save_16, the selective trigger has mean entropy 0.09-0.18, under target H(0.05) = 0.199. α_fire then rises ×20 (0.0005 at 380 k → 0.011 at 530 k). At save_20, entropy exceeds the target (0.25-0.31), then stabilises around it (0.16-0.29). An α of 0.01 weighs against a mean reward of ~0.005 per step.
- **Same signature in 514**: α_fire 0.0002 at 400 k → 0.0106 at 600 k, then 0.007-0.010 up to 1 M, and a hit plateau at 17-20 %.
- **Structural constraint**: about 40 % of armed decisions are off-axis. A selective trigger therefore necessarily has mean entropy under H(0.05). The regulator forbids this equilibrium and forces p ≈ 0.05-0.14 off-axis, i.e. 0.5 to 1.4 off-axis shots per armed second.

## Not established

- Causality is only temporal: α_fire rises, then trigger loosens. Another cause (economy, critic) not excluded.
- Direct test: relaunch 515 with a lower discrete target, changing nothing else (see run 520 later).

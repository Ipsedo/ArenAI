# Probes 510 save_9 (243 k, D 650) and save_28 (756 k, D 1000)

`build/` binary of 12/11 23:40, commit 1e171bbf; only diff with run code (7422d098) = trainers and sigma_gradients tool ⟹ no inference effect, `--force`. 4 episodes per side, stochastic, 32 tanks. Landmarks: 508 save_9 (same index, same D, memory note on run 508 stabiliser), 500 save_23 (621 k, D 1000, memory note on cruise metrics D 1000 hit/shell dump).

| run | side | p_arm | ratio | hit/sh | h<5° | shots<5° | empty_s | %empty | %disarmed | shots/s | turret<5° | σ cx p10/p50/p90 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 508 s9 | 550 | n/a | 6.8 | 15.2 % | n/a | n/a | 6.5 | n/a | n/a | n/a | 13.1 % | n/a |
| **510 s9** | 550 | 0.11 | 6.4 | **20.9 %** | 37.4 % | 37 % | **9.1** | 77 % | 11 % | 0.89 | **20.8 %** | 0.16/0.57/0.64 |
| 500 s23 | 550 | n/a | 4.58 | 13.6 % | n/a | n/a | 8.0 | 97 % | 37 % | n/a | n/a | n/a |
| **510 s28** | 550 | 0.25 | **5.1** | **21.0 %** | 34.5 % | 48 % | 6.4 | 73 % | 16 % | 1.59 | **28.8 %** | 0.04/0.21/0.68 |
| 508 s9 | 1000 | n/a | 11.6 | 7.9 % | n/a | n/a | 13.8 | n/a | n/a | n/a | 9.6 % | n/a |
| **510 s9** | 1000 | 0.06 | 10.8 | **14.9 %** | 37.6 % | 29 % | **19.6** | 81 % | 13 % | 0.47 | **15.7 %** | 0.41/0.61/0.64 |
| 500 s23 | 1000 | n/a | 4.70 | 7.5 % | n/a | n/a | 13.2 | n/a | 23 % | n/a | n/a | n/a |
| **510 s28** | 1000 | 0.11 | **10.5** | **19.5 %** | 34.4 % | 50 % | 8.6 | 77 % | 14 % | 0.86 | **22.7 %** | 0.05/0.59/0.69 |

Aligned & <40 m counts: 1136 / 533 (save_9), 453 / 349 (save_28): save_28 ratios and side 1000 noisy. Deterministic (p > 0.5), aligned / off-axis: save_9 22/3 % and 9/1 %, save_28 54/11 % and 29/4 %. Angle-only ratio (<5°) save_28: 3.9 / 4.7.

CSV: D 1000 at 428 k (500: 595 k, 502: 538 k; milestones ≡ 508 up to 900, curriculum advances at max speed). At D 1000 (430-760 k): CSV hit 0.20 → 0.17, kill 18 → 23-24, fire 0.2-2.2/s; skip 0.02-0.08, skip_canon 0.15-0.26; joint kl peaks 0.10-0.14 between 200 and 360 k, 0.03-0.07 afterwards; Hc 0-0.4, Hd 0.67 → 0.3-0.4; alpha_fire 1e-4 → 0.005-0.012.

## Reading

- **save_9 vs 508 save_9**: trigger as selective (ratio 6.4 / 10.8 vs 6.8 / 11.6), but everything else better: hit/shell ×1.4 / ×1.9, reserve 9.1 / 19.6 s vs 6.5 / 13.8, turret 20.8 / 15.7 % vs 13.1 / 9.6. The combination 508's trigger + 509's aim seen at save_5 is confirmed.
- **save_28 vs 500 save_23 (D 1000)**: hit/shell 21.0 / 19.5 % vs 13.6 / 7.5 % (×1.5 / ×2.6), record. Ratio 5.1 / 10.5 vs 4.6 / 4.7. No dump: 14-16 % of steps disarmed vs 37 / 23 %, a quarter of tanks never empty. Precision no longer collapses at long range: 19.5 % at 1000 vs 21.0 % at 550.
- **More frequent fire than at save_9**: shots/s ×1.8, p_arm 0.11 → 0.25, reserve emptied sooner (6.4 / 8.6 s). Aligned shots go from 29-37 % to 48-50 % of shots, with h<5° stable (~34 %). With ~21 % hits and +2 shells per hit, a shot returns 0.42 shell: firing more becomes cheaper than before. Plausible reading: agent fires more because it hits more, not a return to dump. Not demonstrated.
- **Turret**: 28.8 / 22.7 % at save_28, best level measured since the stabiliser; still progresses after save_9.
- **σ canon bimodal** at save_28: p10 0.04-0.05, p90 0.68-0.69, 2-6 % at ceiling. State-dependent σ (narrow on some decisions, wide elsewhere), as in 500. μ never saturated.
- **CSV**: CSV `hit` falling at D 1000 (0.20 → 0.17) not confirmed by the probe (hit/shell stable at 21 %). It follows cadence (more shots, kill rising).

## Established / noise

- Established (two sides, two saves, counts > 300): at D 1000, 510 is the best run measured in hit/shell, ratio and disarming. It combines 508's trigger and the precision of the factorised credit.
- Established: D 1000 reached 170 k earlier than 500; curriculum caps, so the gap with 508 (identical up to 900) is not measurable by D.
- Hypothesis: cadence rise between save_9 and save_28 is a rational adjustment to the refund, not a dump. To verify on the save chronology (reserve, p_arm, hit/shell per save).
- One seed per run.

## Next

- Probe chronology (skill `probe-timeline`) on a few saves between 9 and 28 to follow p_arm, reserve and hit/shell after D 1000.
- Deterministic probe of save_28: what the game would see (54 / 29 % of aligned decisions fire).

## Dump or not (13/11, shots before / after the 1st empty reserve, side 550 unless noted)

| save | phase | shots | <5° | ≥20° | hit/shell |
|---|---|---|---|---|---|
| 9 | before 1st empty | 2045 | 43.5 % | 14.7 % | 21.6 % |
| 9 | after | 7096 | 34.0 % | 24.0 % | 21.2 % |
| 28 | before 1st empty | 1925 | 60.8 % | 13.4 % | 27.8 % |
| 28 | after | 9476 | 44.4 % | 26.1 % | 20.2 % |
| 28 (1000) | before 1st empty | 1802 | 61.5 % | 15.8 % | 23.4 % |
| 28 (1000) | after | 7046 | 46.1 % | 25.2 % | 18.8 % |

Armed fire rate (save_28, 550) by time since spawn: 0.42 (1-5 s), 0.39 (5-10 s), 0.29, 0.25, 0.20 (> 40 s); share of armed decisions aligned: 0.45, 0.31, 0.30, 0.27, 0.16.

Reading: the 1st reserve is emptied fast but on the tank's best shots (61 % aligned, 28 % hits), not spraying. High early cadence follows the share of aligned decisions at spawn. Residual waste: a quarter of shots after the 1st empty leaves at ≥ 20° (hit 1-3 %), and armed off-axis p_fire doubled since save_9 (0.066 → 0.131 at 550; 0.063 at 1000, unchanged).

## Duration before the 1st empty reserve, per save (13/11)

| save | index | side | empty p10 / p50 / p90 (s) | % empty | % disarmed | shots/s | hit/shell |
|---|---|---|---|---|---|---|---|
| 3 | 81 k | 550 / 1000 | 2.0/2.5/8.4 · 2.0/2.3/4.4 | 93 / 99 | 54 / 66 | 1.78 / 1.41 | 10.4 / 5.2 % |
| 5 | 135 k | 550 / 1000 | 2.2/3.4/8.7 · 2.1/3.0/6.4 | 95 / 99 | 47 / 55 | 1.44 / 1.09 | 12.8 / 7.3 % |
| 9 | 243 k | 550 / 1000 | 2.6/9.1/39.8 · 3.7/19.6/55.1 | 77 / 81 | 11 / 13 | 0.89 / 0.47 | 20.9 / 14.9 % |
| 13 | 351 k | 550 | 2.2/5.0/14.0 | 80 | 15 | 1.60 | 15.9 % |
| 28 | 756 k | 550 / 1000 | 2.3/6.4/14.0 · 3.0/8.6/30.0 | 73 / 77 | 16 / 14 | 1.59 / 0.86 | 21.0 / 19.5 % |

Saves 13 (side 1000), 17, 21, 25 not probed: NVIDIA driver updated during the session (VK_ERROR_INITIALIZATION_FAILED, NVML version mismatch), reboot needed.

Reading: duration rises from 2.5 to 9-20 s between save_3 and save_9, then falls back from save_13 (5.0 s at 550) and stays at 6-9 s up to save_28. Save_9's peak did not hold: cadence rose (0.89 → 1.6 shots/s at 550) before hit/shell returned to 21 %.

In game, shell shortage is structural: at 21 % hits and +2 per hit, a shot costs 0.58 net shell, ≥ 50 % hits would be needed to hold. In training, empty reserve triggers full respawn + `truncated` (future value cut to 0, no penalty); the game (GAME_SHELL_ECONOMY 10/2/20) does not respawn. The policy never lived a « durably empty » state (led to run 511).

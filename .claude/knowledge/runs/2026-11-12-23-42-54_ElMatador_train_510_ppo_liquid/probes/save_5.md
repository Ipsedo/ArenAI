# Probe 510 save_5 (135 k, D 450) vs 509 and 508 save_5

`build/` binary of 12/11 23:40, commit 1e171bbf; only diff with run code (7422d098) = trainers and sigma_gradients tool ⟹ no inference effect, `--force`. 4 episodes per side, stochastic, 32 tanks. 509 save_5 and 508 save_5: same index, same D (12/11 probes, see run 509 `probes/save_5.md`).

| run | side | p_arm | ratio | hit/sh | h<5° | shots<5° | empty_s | %disarmed | shots/s | turret<5° | σ cx/cy | \|μ\|cx |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 508 s5 | 550 | 0.31 | 1.9 | 9.9 % | 21.3 % | n/a | 4.4 | n/a | 1.26 | 17.4 % | 0.40/0.36 | 0.26 |
| 509 s5 | 550 | 0.70 | 1.12 | 12.2 % | 32.0 % | 26 % | 2.5 | 58 % | 1.42 | 18.8 % | 0.15/0.19 | 0.12 |
| **510 s5** | 550 | 0.42 | **1.86** | **12.8 %** | 27.6 % | 29 % | 3.4 | 47 % | 1.44 | **22.1 %** | 0.38/0.35 | 0.19 |
| 508 s5 | 1000 | 0.26 | 2.2 | 4.3 % | 12.0 % | n/a | 4.4 | n/a | 0.93 | 13.1 % | 0.43/0.39 | 0.26 |
| 509 s5 | 1000 | 0.66 | 1.19 | 6.5 % | 23.9 % | 20 % | 2.4 | 74 % | 0.89 | 10.9 % | 0.22/0.27 | 0.12 |
| **510 s5** | 1000 | 0.34 | **2.12** | **7.3 %** | 18.3 % | 26 % | 3.0 | 55 % | 1.09 | **19.7 %** | 0.46/0.41 | 0.20 |

Aligned & <40 m counts: 835 (550), 417 (1000, noisy). Angle-only ratio (<5°): 1.58 / 1.35. Deterministic (p > 0.5): fires 61 / 32 % of aligned decisions vs 18 / 10 % off-axis. No μ saturation, σ never at ceiling.

CSV 100-120 k (510 / 509 / 508): hit 14.5 / 15.0 / 12.2 %, kill 23 / 24 / 18; skip 0 / 0.46 / 0; joint kl 0.023-0.079 (peaks at 111 and 117 k) / 0.03 / 0.011; Hc ≈ 0 / −0.4 / +0.15.

## Reading

- **Trigger**: caught up with 508 between save_3 and save_5. Ratio 1.05 → 1.86-2.12 (508: 1.9-2.2; 509: 1.12-1.19), aligned/off-axis gap 0.30-0.35. Level stays higher than 508 (p_arm 0.34-0.42 vs 0.26-0.31): partial dump exit, reserve empty at 3.0-3.4 s vs 4.4 s, 95-99 % of tank-episodes still empty.
- **Precision**: best hit/shell of the three runs (12.8 / 7.3 %). h<5° under 509 (27.6 / 18.3 vs 32.0 / 23.9) but above 508.
- **Turret**: 22.1 / 19.7 % vs 17-19 / 11-13 %: best of three, gap holds since save_3 and grows at 1000.
- **σ canon**: back from 0.19 to 0.38-0.46, at 508's level; canon μ more marked (0.19 vs 0.11).

## Established / noise

- Established (two sides, consistent with deterministic): at 135 k, 510 has a selective trigger at 508's level, which 509 does not. With the save_3 probe, lifting the joint skip does not advance the trigger's start (flat at 81 k like 509) but lets it learn afterwards, in ~55 k.
- Nuances save_3's conclusion (« lag ≠ skip »): the initial lag vs 508 is not due to the skip, 509's lack of catch-up probably is. Single seed per run.
- Probable: 510 combines 509's aim and 508's trigger at 135 k (turret and hit/shell ahead). To confirm on later saves.
- Unexplained: joint kl peaks 0.07-0.08 without global stop; to watch (KL regime, cf. 449).

## Next

- save_9 (243 k) vs 508 save_9 (ratio 6.8 / 11.6, reserve 6.5 / 13.8 s, hit/shell 15.2 / 7.9 %): does dump exit continue and does turret keep its lead?
- Follow joint kl and skip_canon in CSV.

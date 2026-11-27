# train_521_ppo_liquid

| | |
|---|---|
| machine | ElMatador |
| user | samuel |
| launch | 2026-11-15 12:04:11 |
| commit at launch | `9d843717` — add trained model for ppo_liquid |
| commits ≤ 90 min after | none |
| diff at launch | unknown (identity recorded after the run ended) |
| raw data | `outputs/train_521_ppo_liquid` on ElMatador |

## Established facts

- 2026-11-15: 521 (launched 2026-11-15 12:04) = 520 config + `shells_per_hit` 3 → 2 (economy-agent-2-per-touch), sole change (commit 9d843717 + uncommitted `shell_economy.h`). Tests whether a binding reserve pushes the trigger toward precision. Curriculum ~25-30 k behind 520 (noise).
- 2026-11-15: save_9 (243 k, D 600) vs 520 save_9: no dump phase (p_off 0.18/0.15 vs 0.62/0.54, ratio 2.8; disarmed 28/35 % vs 42/64 %; 1st empty 4 s vs 2.7 s); precision unchanged (hit/shell 12.4/7.3 %, h<5° 30/22 %, tour<5° 17/13 %); 89-95 % of tanks still empty their reserve.
- 2026-11-15: save_16 (432 k, D 950): no extinction (520 save_16: 0.13-0.29 shot/s); 520 save_32 (864 k) level reached in half the indices (shots/s 1.06/0.50, touches/tank-s 0.235/0.112); but hit/shell 22 % (det 27-28 %) ≤ 520, empty reserve earlier (det 9.5/19 s vs 21/35 s). N=2 removes the oscillation, not yet a precision gain. Best save of 521 on the targets: save_16.
- 2026-11-15: save_23 (621 k, D 1000): cadence ↑ (1.32/0.55 shots/s), hit/shell ↓ 21-22 % (det 21-24 %), median 1st empty 5.6/11 s (save_16: 9/20; 520 save_24: 15/45 s) ⟹ behind 520 on hit/shell and time before empty: N=2 did not create pressure toward precision, 520's volume drift continues.
- 2026-11-15: save_26 (702 k): drift confirmed — hit/shell 21/19 %, 1st empty 5.3/8.6 s, 80-85 % empty, disarmed 15-19 %, side-1000 ratio 17 → 8; CSV hit 0.25 → 0.18-0.20 (600-728 k). Stop recommended ~20:00; run stopped at 752 k (8.06 h, 93 k steps/h, 2.99 M tank-steps/h, 27 saves). Follow-up validated by Samuel: 522 = 521 + empty-reserve penalty P = D/1000 (−0.25 at D 250 → −1 at D 1000) at the truncated step, respawn kept.
- 2026-11-15 (CSV analysis): at D 1000 corr(hit, r_hit) = −0.79 and corr(kill, r_hit) = +0.88 (325 rows); precision peak (hit 0.25, 600-620 k) = reward trough (r_hit 3.6e-3, kill 15), reward max (8.4e-3, 680-720 k) at hit 0.17-0.18 ⟹ volume drift = gradient ascent. At equal r_hit (8e-3), 520 at hit 0.25, 521 at 0.18-0.21. Residual r − r_hit − r_rcv − r_aim = −6e-5 (|max| 1.2e-4): zero reference to read 522's penalty. Estimate: at 521's end behaviour, one emptying every 13-20 s ⟹ penalty 5-7.5e-3 per tank-decision at P = 1, of the order of r (5.4e-3). Empty-reserve rule in 521: truncated in-place respawn, no penalty ⟹ an emptying refunds 10 shells immediately = free reload.

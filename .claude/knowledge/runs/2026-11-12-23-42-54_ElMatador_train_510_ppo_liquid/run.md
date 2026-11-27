# train_510_ppo_liquid

| | |
|---|---|
| machine | ElMatador |
| user | samuel |
| launch | 2026-11-12 23:42:54 |
| commit at launch | `1e171bbf` — * gun laying : real enhancement for the policy => stabilize the canon regarding the movements * continuous action (canon_x, canon_y) optimized separately : compute advantage with phi(s_t) in potential reward fashion (so without critic). other actions remain the same advantage (with critic advantage) * new phi(s_t) : based on theoretic shell trajectory, compute a reward representing the alignment of canon regarding the vision * fix : player controller with new gun laying |
| commits ≤ 90 min after | `7422d098` |
| diff at launch | unknown (identity recorded after the run ended) |
| raw data | `outputs/train_510_ppo_liquid` on ElMatador |

## Established facts

- 2026-11-13: Run = 509 + `GroupKlStop` (per-group KL stop, canon/rest; see memory note on per-group KL stop). Probe save_3 (81 k; binary `1e171bbf` in `--force`, diff = trainers only): p_arm 0.77, ratio 1.05, reserve empty 2.3-2.5 s ≡ 509 save_3. hit/shell 10.4/5.2 % vs 8.1/4.3, turret <5° 20.8/16.6 vs 18.1/12.9 (one seed). CSV: skip_canon 0.07 → 0.17, kl 0.03, hit/kill ≡ 509. Also skip 0 (509: 0.4), σ canon 0.19/0.24 (509: 0.09/0.13), h<5° 24/14 %. Trigger flat + dump 2.4 s identical to 509. D milestones identical at index in 508/509/510: non-discriminating. Lifting the skip does not unblock the trigger at 81 k.
- 2026-11-13: 508 save_3 probed: ratio 1.23-1.24, p_arm 0.55, reserve 2.9-3.0 s ⟹ 508 already sorts at 81 k, 509/510 do not; trigger delay of factorised-credit runs present from save_3, not caused by the joint skip (cause unidentified, one seed per run). 508 behind on aim (turret <5° 12.6/9.9 % vs 20.8/16.6).
- 2026-11-13: save_5 (135 k, D 450): trigger caught up: ratio 1.86/2.12 (508 1.9/2.2; 509 1.12/1.19), p_arm 0.42/0.34, reserve 3.4/3.0 s (508 4.4); turret <5° 22.1/19.7 % (best of three), hit/shell 12.8/7.3 % (best); σ canon 0.38-0.46. Reading: skip does not delay trigger take-off but prevented 509 from catching up. Joint kl peaks 0.07-0.08 at 111-117 k.
- 2026-11-13: save_9 (243 k) and save_28 (756 k, D 1000): D 1000 at 428 k (500: 595 k). save_9 vs 508 save_9: ratio ≈ (6.4/10.8), hit/shell 20.9/14.9 % (508 15.2/7.9), reserve 9.1/19.6 s, turret <5° 20.8/15.7. save_28 vs 500 save_23: hit/shell 21.0/19.5 % (13.6/7.5) record, ratio 5.1/10.5, disarmed 14-16 % (37/23), turret <5° 28.8/22.7 %; cadence ×1.8 vs save_9 (reserve 6.4/8.6 s) — hypothesis: fire profitable at refund (0.42 shell returned/shot), not a dump. σ canon bimodal (p10 0.05, p90 0.69). Analysis `probes/save_9_28.md`.
- 2026-11-13: « Dump » at save_28: not in the historic sense. Shots before 1st empty: 61 % <5°, 13-16 % ≥ 20°, hit 23-28 % (after: 44-46 % / 25-26 % / 19-20 %). Fast emptying = life start rich in opportunities. Residual: armed off-axis p_fire 0.066 → 0.131 at 550 between save_9 and save_28.
- 2026-11-13: 1st empty reserve per save, p50 at 550: 2.5 (s3) → 3.4 (s5) → 9.1 (s9) → 5.0 (s13) → 6.4 s (s28); save_9 peak not held, cadence back to 1.6 shot/s. In game tanks end dry: shot = −0.58 net shell at 21 % hits, and training respawns full at empty reserve (truncated, V cut) while the game does not ⟹ train/game mismatch (decision for Samuel; led to run 511). Best run per memory index (D 1000 at 428 k, save_28 hit/shell 21/19.5).

# train_509_ppo_liquid

| | |
|---|---|
| machine | ElMatador |
| user | samuel |
| launch | 2026-11-12 21:36:21 |
| commit at launch | `1e171bbf` — * gun laying : real enhancement for the policy => stabilize the canon regarding the movements * continuous action (canon_x, canon_y) optimized separately : compute advantage with phi(s_t) in potential reward fashion (so without critic). other actions remain the same advantage (with critic advantage) * new phi(s_t) : based on theoretic shell trajectory, compute a reward representing the alignment of canon regarding the vision * fix : player controller with new gun laying |
| commits ≤ 90 min after | none |
| diff at launch | unknown (identity recorded after the run ended) |
| raw data | `outputs/train_509_ppo_liquid` on ElMatador |

## Established facts

- 2026-11-12: Run = 508 + factorised aim credit (`aim_gae_lambda` 0) (first test of the factorised credit + aimed Φ design; follows run 508). CSV at 40 k: hit 14.6 % / kill 21.9 vs 508 11.5 / 17.6, gap from 4.5 k; D 300 identical. Hc −0.48 (508 −0.08), skip 0.02 → 0.19 monotone, kl 0.030 (3× 508). Probe save_1 vs 508 save_1 (same binary `1e171bbf`, 508 in `--force`): aligned shots more frequent (21/19 % vs 18/13 %) but h<5° unchanged ⟹ gain = more frequent alignment, not per-shot precision. Trigger flat (ratio 1.02) in both runs, dump 2.5 s. Probe save_1: turret <5° 15.6/14.2 % (508: 13.6/9.1), hit/shell +20-38 %, σ canon 0.13 (508: 0.23).
- 2026-11-12: save_3/save_5 (81/135 k, D 400/450, ~23:30): h<5° 32.0/23.9 % (508 save_5: 21.3/12.0) and hit/shell 12.2/6.5 % (508: 9.9/4.3) ⟹ gain = aligned shot precision; turret <5° 18.8/10.9 % ≈ 508 (17.4/13.1), save_1 lead did not hold. Trigger late: ratio 1.12-1.19 vs 1.9-2.2, p_arm 0.66-0.70 vs 0.26-0.31, reserve empty 2.4 s vs 4.4 s. CSV: skip 0.40-0.46 from 108 k (508: 0), kl 0.03; σ canon trough 0.09 at save_3, back to 0.15-0.27 (alpha_canon ×4). D milestones identical to 508. Unverified hypothesis: joint ratio dominated by canon ⟹ skip deprives trigger of ~40 % of updates.
- 2026-11-12: KL by group, save_5 (23:35): tool `arenai_agent_sigma_gradients --kl_by_group` (`LiquidPpoTrainer::kl_by_group` + `load_actor_optimizer`, not committed). At the stopping step: canon 77 % of joint KL, displacement 12 %, discrete 1 %; displacement+discrete never > 0.075 over 304 steps ⟹ the skip is caused by the canon. Trigger KL 0.0002-0.0018 per step even without cut ⟹ skip → trigger delay link not demonstrated. Replay cuts earlier than the run (~75 % vs 40 %): magnitude to take with caution. (Led to run 510 per-group KL stop.) Related notes: see comparisons for factorised credit design.

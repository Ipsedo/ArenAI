# PPO liquid (IMPALA) train 449 — KL regime from first rollout, skip 96 % (2026-10-29)

Host: ElMatador (memory notes); raw data no longer on disk.
Source note date: 2026-10-29 (ppo_liquid_impala, 472 k steps; run 2026-10-28/29). 449 = 448 + target_kl 0.05 → 0.02 + continuous entropy bonus 0.01 → 0.001 and discrete 0.001 → 0 (commits « lower target kl » and « backtrack to original entropies »). 448 already had actor_epochs 4 → 8.

The actor learns nothing because it performs almost no gradient steps:
- skip ≈ 0.3 within the first 10 rollouts (σ still 0.5), 0.6 at 30 k, 0.96 at 400 k. 80 planned steps per rollout (160 chunks of 30 / 17 per batch × 8 epochs) ⟹ ~3 executed at the end.
- Hc 0.57 → −1.1 ⟹ σ 0.5 → ≈ 0.08; Hd 0.69 → 0.21 (fire/zoom almost deterministic, factor 0).
- Loop: KL per step ∝ (Δμ/σ)² ⟹ small σ ⟹ threshold 1.5 × 0.02 crossed at step 2 ⟹ early stop; the rare steps still shrink σ (bonus 0.001 too weak, see continuous bonus 0 = σ collapse).
- Game unchanged: fire 0.85/s, 24 s episode (reserve emptied then death, see run 437), hit 1-2 % < 6 % ⟹ D stuck 250.
- Comparison: 447 (epochs 4, target 0.05, entropy 0.01/0.001) skip 0, Hc stable 0.6; 448 (epochs 8, target 0.05) already skip 0.65 at 100 k. 8 epochs + tight target are the cause.
- Same KL regime as 416 and 441, worse: threshold reached before the policy moved. Do not read absence of progress as a plateau when skip > 0.3; look at skip and Hc first. For 450 (proposed): back to target_kl 0.05 (or epochs 4) and bonus 0.01/0.001; one change at a time.

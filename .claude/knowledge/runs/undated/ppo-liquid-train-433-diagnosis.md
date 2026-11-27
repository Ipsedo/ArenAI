# PPO liquid train 433 (older, 2026-10) — no continuous entropy + epochs 6

Host: ElMatador (memory notes); raw data no longer on disk.
Note: older run than the local `train_433` of 2026-11-24 (see runs/2026-11-24-18-43-54_ElMatador_train_433_ppo_liquid). Source note dated 2026-10-24/25; folder `outputs/train_433_ppo_liquid_no-entropy`.

Continuous entropy factor 0, but also epochs 2→6, Hd 1e-4→1e-3 and initial_fire 0.1→0.25 at the same time. Result over 1.8 M steps: D never left 250 (probe hit/fire never ≥ 0.06; sliding max 0.050 at 300 k then falls to 0.03). Hc 0.50 → −0.88 (σ ≈ 0.5 → 0.1, canon_y 0.08), kl 0.004 → 0.083 (> target 0.05 from 630 k), skip 0 → 68 %, clip 0.05 → 0.53. corr(Hc, kl) = −0.96. Hd_fire 0.68 → 0.05 despite its active 1e-3 bonus.
Reference 430 (entropy 1e-3, epochs 2): D=250 for 790 k, then 350 at 1.4 M, kl 0.003-0.004, skip 0, but Hc drifts anyway 0.53 → 0.30.
- Over the first 125 lines (~110 k), 435 (= 433 + continuous entropy 1e-3, epochs 6) is indistinguishable from 433: kl 0.010, clip 0.15, Hc 0.46 vs 0.44; while 430/431/432 (epochs 2) have kl 0.003 and clip 0.035 at the same point. The KL/clip regime is due to epochs 6, not entropy; a 1e-3 bonus retained neither Hc in 430 nor Hd_fire in 433.
- Do not attribute 433 to the entropy removal alone. Pending then: let 435 pass ~500 k (where 433 has Hc < 0 and skip > 25 %). If 435 collapses too: it is epochs 6 (see runs 416, 417).
- Related: entropy regulation known gaps; run 430.

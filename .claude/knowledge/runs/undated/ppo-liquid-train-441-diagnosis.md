# PPO liquid train 441 — regression masked by the curriculum, KL regime at the stop (2026-10-27)

Host: ElMatador (memory notes); raw data no longer on disk.
Source note date: 2026-10-27 (run 2026-10-26/27, ppo_liquid, 3.26 M steps, 30 saves). No config.json in the folder; presumed identical to 440's config (continuous entropy 0.001, gamma 0.994, p_init fire 0.01, epochs 4, target_kl 0.05, minibatch 1536, lr 1e-4 actor and critic). Entropy 0.001 restored.

- Trajectory: D=250 until 1.1 M (kill 0.2, hit 4 %), then fast climb: D 300 (1.2 M), 350 (1.4 M), 400 (1.6 M), 450 (2.1 M) — 430 had needed 4.7 M for D=450. Peak 1.3-1.5 M: kill 0.7-0.8, hit 6.6 %. At D=450 kill falls 0.45 → 0.2; curriculum (probe ratio < 3 %) retreats: 400 (2.7 M), 350 (3.1 M), 300 (3.2 M). End: kill 0.25, hit 4 %, level of 1.2 M.
- Visible cause: optimisation regime degrading from ~2.8 M. kl 0.02-0.03 → 0.069 (> target 0.05), clip 0.2 → 0.37, skip 0-5 % → 41 % over the last 100 k. Hc 0.3 → −0.53 (σ ≈ 0.15 if Hc is the per-dim mean). Critic: erratic ev, often negative (−1.5 at 2.8 M), v 0.7 → 0.3. Same pattern as 439 (Hc < 0 after 1 M, skip rising), 416 (KL regime) and 429 (peak then D falls).
- vs 430 (same 0.001 bonus): kl 0.004 and skip 0 all run, slow but monotone progress, kill 0.9-1.2 at D=450. 441 is 2× faster early but unstable — the lever is not the reward but the KL regime (runs 416, 417).
- Samuel saw « convergence stagnating »; in fact the curve goes down and the curriculum masks it (D drops so the task simplifies, kill rises a bit). Rule: on a liquid run read D/side with kl/skip: a D decrease is a regression, not a plateau. When skip > 10 % and kl > target, the other metrics are no longer interpretable.
- Related: continuous bonus 0 = σ collapse; run 429.

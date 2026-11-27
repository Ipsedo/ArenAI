# PPO liquid (IMPALA) train 450 — audit, no bug, window too short (2026-10-29)

Host: ElMatador (memory notes); raw data no longer on disk.
Source note date: 2026-10-29 (run 450, config 447, 65 k steps).

Audit: no bug found in ppo_liquid_impala — train.cpp samples (sample=true), the stored liquid state is the one opening the step and reopening each chunk, fire mask identical collection/update, GAE with bootstrap on truncated, advantages normalised on the rollout, chunk row mapping correct. 443 (impala, hyperparams of 442) followed exactly 442's trajectory ⟹ the IMPALA architecture is not the cause.

Facts measured:
- 441 (only run that learned): D leaves 250 around 800 k, kill 0.7 at 1.2 M, 3.27 M in 17.5 h. Since 443 no run exceeded 470 k (443 311 k, 444 196 k, 447 179 k, 450 65 k) and throughput fell from 186 k/h (441) to 104 k/h (443, IMPALA cost) then ~50 k/h (447-450).
- Only visible learning before 500 k in 441/442/443: σ drops (Hc 0.5 → 0.15-0.3) and kill rises (0.1 → 0.25-0.36). Since the 0.01 continuous bonus (446+), Hc rises (0.55 → 0.63) and kill falls (0.1 → 0.04): the bonus beats the surrogate for σ.
- kl per step flat 0.003 in 450 despite lr 3e-4 (441: 0.004 → 0.013 at 1e-4), clip decreasing: consistent with an actor gradient direction that is noise (rollout 300, batch 512 ⟹ ~2 deaths per minibatch, death −1 = 85 % of the return).
- Why: every run since 444 changes 2+ parameters and is judged on 1-8 h where nothing can be seen (see run 449).
- Baseline to hold ≥ 1.2 M steps (~24 h at current throughput) before any verdict: config 443 (lr 1e-4, epochs 2, rollout 900, batch 512, bonus 0.001/0, target_kl 0.05); then one change per run.

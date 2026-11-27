# PPO liquid (IMPALA) train 451 — reserve-empty truncation, but 449's hyperparams (2026-10-29)

Host: ElMatador (memory notes); raw data no longer on disk.
Source note date: 2026-10-29. 451 = patch « empty reserve = truncation » (jolt_enemy_tank.cpp, is_timeout = starved || out_of_shells) + header liquid_impala_ppo_hyperparams.h back to 449's values (file rewritten 10:17, before the 10:20 build; 451's config.json confirms: actor_epochs 8, target_kl 0.02, bonus 0.001/0, lr 3e-4, rollout 300, batch 512, 16 tanks).

Measured over 72 k steps (82 lines):
- Patch effect: r = 0.000/step (449/450/437: −0.003), r_rcv −0.001, s 24 s, fire 0.8/s, hit 1.4 %, kill 0.05 ⟹ the penalty is gone, the game unchanged (empty in ~24 s then end).
- Critic: v 0.43 → 0.16, negative ev (−1.7 mean, −16 in places): the target is only a rare noise (hit +0.2 / received −0.3) amplified by the ReturnScaler.
- Actor: kl 0.016 at the stop, clip 0.21, skip 0.13 → 0.65, Hc 0.52 → −0.07 (σ 0.5 → ≈ 0.25), faster than 449 at equal index (mean Hc 0.22 vs 0.45).
- Why: the event-density lever (event density collapsed since 10-21) can show nothing while the optimiser is in the KL regime of 449. Check a run's config.json before interpreting; judging the truncation needs config 447/450 (target_kl 0.05, epochs 4, bonus 0.01/0.001) or 443.

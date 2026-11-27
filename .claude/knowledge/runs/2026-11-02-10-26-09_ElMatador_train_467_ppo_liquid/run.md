# train_467_ppo_liquid

| | |
|---|---|
| machine | ElMatador |
| user | samuel |
| launch | 2026-11-02 10:26:09 |
| commit at launch | `d02a9b6b` — * backtrack lambda and gamma * out of shells penalty at -0.5 |
| commits ≤ 90 min after | none |
| diff at launch | unknown (identity recorded after the run ended) |
| raw data | `outputs/train_467_ppo_liquid` on ElMatador |

## Established facts


- 2026-11-02: offline probe on save_3 settles the question of the ammo-dump note: the policy is not blind, it reads the bearing (turret of right sign, p_fire 0.88 ahead / 0.70 behind) but the trigger is flat vs the most centred enemy (0.74-0.81 from 0° to 180°), fires at 0.86 with 1 shell left, and the turret commands only 20 % of its speed with a dead zone < 6° ⟹ lock = shape of the reward, not perception (aim 0.068 at 26 m miss vs 0.1 perfect, a shot always positive). Details: `probes/save_3.md`.
- 2026-11-02: Samuel refused the sliding baseline. Static track proposed (not validated at that time): tol = 2.5 m + 3°·d ⟹ at 40 m a 20° shot pays 0.004, 5° 0.047, 2° 0.074 (contrast ×15 instead of ×2.5); remove fire cost 0.01 (revenue/shot falls toward 0.015). **Applied 2026-11-02 (validated by Samuel)** for run 468. Success criterion for the next run: p_fire depends on `best_angle`.

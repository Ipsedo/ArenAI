# train_476_ppo_liquid

| | |
|---|---|
| machine | ElMatador |
| user | samuel |
| launch | 2026-11-04 07:08:46 |
| commit at launch | `12e06985` — more shells in reserve + more shells added at hit + no more out of sells death |
| commits ≤ 90 min after | `324d21a9` |
| diff at launch | unknown (identity recorded after the run ended) |
| raw data | `outputs/train_476_ppo_liquid` on ElMatador |

## Established facts


- 2026-11-04: config 475 with ADR thresholds 0.05/0.08. D 250 → 500 in 866 k, but probe on save_8 shows a saturated trigger: p_fire = 0.99 at every step, P(fire | can_fire) = 0.99, a shot at every cooldown window (~5/s).
- 2026-11-04: progress 0 (30/20/60, 250 m): reserve emptied ≥ 1 time for 78/128 tanks, 37 % of living time empty, 99 shots/tank (42 in first 10 s), hit 12 % (18 % in 0-10 s, 3 % after 30 s). Progress 1/3 (23/15/45, 500 m): 119/128 tanks empty, 1st empty at 4.6 s median, 70 % of time empty, 45 % of shots in the first 5 s. 1.3-1.8 % of shots aligned (< 5°, < 40 m) as in 470: aim did not move.
- 2026-11-04 why: break-even (1/shells_per_hit) = 5 % at progress 0 and 6.7 % at 1/3, under the spam hit (8-9 %): spraying stays free or profitable over the whole visited range. The spam hit/fire ratio does not depend on the economy, so ADR rises as soon as its high threshold is under that plateau (476) and stalls when above (475): D progress measures no selectivity. Bernoulli at 0.99 has gradient p(1-p) ≈ 0.01: trigger as frozen as in 471 (p ≈ 0), at the other bound. Do not read 476's D rise as progress; an ADR on hit/fire cannot force restraint while spam reaches the threshold (see run 475, scarcity-economy-tied-to-competence, ammo-dump-optimum, ammo-economy-break-even notes).

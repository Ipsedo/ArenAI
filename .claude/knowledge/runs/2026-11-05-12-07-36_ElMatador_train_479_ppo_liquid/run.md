# train_479_ppo_liquid

| | |
|---|---|
| machine | ElMatador |
| user | samuel |
| launch | 2026-11-05 12:07:36 |
| commit at launch | `fc5b0444` — add trained ppo liquid |
| commits ≤ 90 min after | none |
| diff at launch | unknown (identity recorded after the run ended) |
| raw data | `outputs/train_479_ppo_liquid` on ElMatador |

## Established facts


- 2026-11-05: launched 12:07 = 478 config unchanged + `out_of_shells` restored in jolt_enemy_tank.cpp (death −1 in training when nb_shells ≤ 0 and no shell in flight; Samuel's decision « Mort à vide quand même » despite 437/470). The model change was **not committed** at launch (cmake-build-release binary 12:07:35, run 12:07:36: it embeds it). Read 2026-11-05 at 156 k steps. (see docs/decisions.md: empty reserve as death removed 2026-11-22)
- 2026-11-05 CSV vs 478 at equal index: fire 0.57 (0-30 k) → 0.30 (87 k) → 0.09 (100 k) → 0.05/s after (478: 0.4-0.6 throughout); kill 0-0.2 then 0 from 100 k (478: 0.2-1.1); s 24 → 28 s = starvation; r_hit 0.000; Hd 0.38 → 0.25; Hc 0.19 → 0.15; D 250 → 300 at 110 k because CSV hit (7-10 % on a handful of shots) passes threshold 0.08: **curriculum on a mute agent, like 465**.
- 2026-11-05 probe save_1 (108 k, side 250, 4 ep., 35 k armed decisions) vs 478 save_1 same protocol: p_fire(can_fire) **0.005** (median 0.002, p95 0.019, 0 % > 0.5); 478 save_1: 0.243 (4 % > 0.5, then 0.76 at save_3). Flat vs angle, distance, reserve. Spawn reflex 0.037 in the first second, 0.004 after. Reserve: can_fire 99 %, **0 % of tanks empty their reserve** (478: 84 %, median 5.8 s), 0.06 shot/s (478: 0.72). The 4 episodes are exactly 330 steps = everyone dies of starvation. Turret intact, even a bit better: < 5° 8.7 % (478 save_1: 6.2 %), < 15° 33 % (26 %), σ canon 0.30/0.29. The 174 landed shells hit 9.8 %.
- 2026-11-05 why: empty-reserve death is a uniform cost on firing (each shell brings death closer by the same amount whatever the angle). Under centred reward (shot neutral in expectation) the −1 at the 10th shell dominates: the policy cuts the trigger in 100 k, exactly 465/471. With 437/470 (dump then death) this is the 3rd confirmation: a cost on firing gives dump, uniform throttling or extinction, never conditioning on angle. Stop 479 (D will rise artificially); no more cost/death tied to firing without a good/bad shot differential. Scripts: scratchpad session 60e24df7 (probe479/).
- 2026-11-05 15:00 (Samuel: « on backtrack la mort + pénalité »): model code restored to committed state (git checkout of 3 files), cmake-build-release binaries rebuilt, model tests 80/80. « What if we lower the penalty? » answer: no — −0.5 (470) = dump then death, −1 (479) = extinction, 0 (478) = dump then throttling; an intermediate value stays flat vs angle. Levers proposed for 480 on the 478 base, one at a time: **actor_epochs 4 → 2** (precedent 417 vs 416, recommended first: turret regresses under KL regime) or slow regen 1 shell / 5 s.
- 2026-11-05 15:30 (Samuel: « Epochs 2 »): for run 480, default `actor_epochs` 4 → 2 in liquid_ppo_hyperparams.h (critic_epochs stays 4), sole change vs 478; cmake-build-release binaries rebuilt, `--help` shows default 2, agent tests 2280/2280 (109 skipped). To compare with 478 at equal index: skip (0 expected), kl, Hc, turret < 5° at save_3 (11.2 % on 478).

# train_556_ppo_liquid — CSV analysis (2026-11-27, run in progress)

Launched 2026-11-27 02:00 on ElMatador, last row 568 k at 08:51, 83 k steps/h (2.65 M tank-steps/h). Effective
commit `5367a45e` (clean tree at launch, high confidence).

## What the run tests

Vs 555 (`cfcc1e81`): panorama encoder shared actor/critic, trained supervised on the enemy visibility mask
(`d75fec10`; `panorama_mask_batch_size` 512, `epochs` 1, `lr` 3e-4), mean aggregation over rows (`e318fb2d`),
plus two render commits (`876fbe9e`, `44f12436`, perf). Several changes ⟹ confounded; one seed.

## CSV reading (train rules)

- Milestones: D 1000 at 461 k (554: D 950 at 404 k, never 1000 by 546 k; 555: stuck 850 from 366 k to 680 k).
- Index window 430-546 k: kill 21.1 (554 18.0; 555 18.0), hit 0.17 (0.27; 0.29), fire 0.86 (0.40; 0.44),
  miss 69 m (59; 42), Hc −0.54 (−0.33; −0.27).
- `mask_loss`: 0.28 at 40 k → 0.18-0.21 at 450-568 k, noisy, no clear plateau.
- Healthy KL regime: kl 0.015-0.02, skip ≤ 0.07, α at floor 1e-4 from 450 k.

## Diagnosis

- Established (match, `matches/save_21.md`): in game, s17 and s21 empty the reserve (dry at 10 s 0.27 / 0.48),
  hit/shell 0.18-0.26, gun low; below 554 s17 and 555 s17. Training kill and curriculum progress are carried by
  fire volume refilled when dry (553 profile).
- Undecided: whether the mask makes the bearing decodable from the trunk (554 measure: 0.51-0.58); whether mask
  loss ~0.2 is a learning ceiling or the irreducible part.

## Open options (no decision) and markers

- Decode out-of-field bearing from the trunk at save_17/21, same probe as 554/555.
- `probe-checkpoint` s21: reserve emptying, trigger selectivity, gun elevation.
- Match at save_25+ for a 3-save trajectory.

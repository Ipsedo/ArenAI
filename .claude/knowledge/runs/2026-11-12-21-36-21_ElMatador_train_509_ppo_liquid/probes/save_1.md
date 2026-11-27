# Probe 509 save_1 (27 k, D 300) vs 508 save_1

`build/` binary of 12/11 21:06, commit 1e171bbf = code of run 509, no inference diff. 508 probed with the same binary in `--force`: the diff (25a75a34 → 1e171bbf) contains the stabiliser, which 508 already had uncommitted, plus the reward's Φ and the probe geometry. It changes neither observation nor actions. Stochastic, economy of config.json. 509 side 550: only 2 episodes (remnant of an interrupted probe), side 1000: 4 episodes. 508: 4 episodes per side.

Only config difference between the two runs: `aim_gae_lambda 0` (canon optimised on the λ-return of Φ, no critic).

| run | side | p_arm | ratio | hit/sh | h<5° | shots<5° | empty_s | %disarmed | shots/s | turret<5° | σ cx/cy | \|μ\|cx |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 509 | 550 | 0.72 | 1.02 | 6.7 % | 13.7 % | 21 % | 2.5 | 68 % | 1.18 | 15.6 % | 0.13/0.14 | 0.085 |
| 508 | 550 | 0.63 | 1.02 | 5.6 % | 12.8 % | 18 % | 2.6 | 62 % | 1.41 | 13.6 % | 0.24/0.22 | 0.084 |
| 509 | 1000 | 0.72 | 1.02 | 3.3 % | 7.9 % | 19 % | 2.4 | 71 % | 1.16 | 14.2 % | 0.13/0.14 | 0.076 |
| 508 | 1000 | 0.63 | 1.02 | 2.4 % | 7.5 % | 13 % | 2.6 | 65 % | 1.33 | 9.1 % | 0.24/0.23 | 0.076 |

## Reading

- **Aim**: turret<5° +15 % (550) and +56 % (1000); shots leave aligned more often (21/19 % vs 18/13 %). 509 at 27 k already reaches 508 save_9's level (13.1/9.6 % at 243 k) and 507 save_9's (14.9/13.4 %).
- **Fire**: hit/shell +20/+38 % with fewer shots per second. An aligned shot hits as often as before (h<5° 13.7 vs 12.8 %; 7.9 vs 7.5 %). Gain comes from alignment frequency, not aligned-shot precision. Consistent with a credit on the canon gesture.
- **Trigger**: flat in both runs (ratio 1.02), dump at 2.5 s, 62-71 % of time disarmed. Normal at 27 k: trigger stays on the usual GAE 0.95 advantage, nothing changed for it.
- **Heads**: σ canon 0.13 vs 0.23. Canon tightens its law twice as fast, explaining the Hc drop in CSV (−0.48 at 40 k vs −0.08). No saturation (|μ| 0.08, nothing at ceiling).

## Established / noise

- Established at this index: turret and canon better placed than in 508 at same index, and hit/shell higher. Gap consistent on both sides and with CSV (hit 14.6 vs 11.5 %).
- Single seed, 2 episodes side 550: exact magnitude (+15 or +56 %) unreliable.
- To watch: σ canon. If it keeps going toward 0.05, canon will not explore any more, resembling the 433-439 collapse but limited to canon. KL skip also rising.

## Next

- save_3-5 (81-135 k): turret<5°, σ canon, and first sign of trigger selectivity, vs 508 save_5 (turret<5° 17.4/13.1 %, ratio 1.9-2.2).
- D milestones: only final judge (D identical at 300 so far).

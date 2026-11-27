# Run 554 — ppo_liquid, panorama

Launched 2026-11-25 21:06, running; read at index 104 k (2.08 h, 50 k steps/h, 1.6 M tank-steps/h), D 400. Effective commit `b6f61755` (panorama), training binary of 11-25 21:06 launched before the 21:30 commit; clean tree, following commits = docs. Good confidence (config.json carries `panorama_*`).

## What the run tests

550's code (single advantage without Φ; presence head / spatial softmax / spatial aim removed by `61344578`) + **360° panorama as second vision input** (`panorama_channels` 3 blocks, 4 output channels). Reference = 550 (one seed). Also in the diff: vision size as compile-time constant, thread pool.

## CSV (training rules)

| D | 554 kill / hit / fire | 550 | 551 | 552 |
|---|---|---|---|---|
| 300 | 19.7 / 0.124 / 1.55 | 21.4 / 0.134 / 1.40 | 20.4 / 0.119 / 1.49 | 20.6 / 0.134 / 1.39 |
| 350 | 23.5 / 0.135 / 1.51 | 23.2 / 0.142 / 1.61 | 22.5 / 0.121 / 1.68 | 22.8 / 0.128 / 1.62 |
| 400 | 23.3 / 0.136 / 1.48 | 25.0 / 0.141 / 1.57 | 23.1 / 0.109 / 1.61 | 22.3 / 0.108 / 1.38 |

- D milestones 300 @ 21 k, 350 @ 48 k, 400 @ 67 k: identical to 551/552/545 (550: 48/67/114 k). The "D blocked at 400" signal is none: 551/552 pass to 450 at 114 k.
- kl 0.010-0.013, skip < 1 %, ev 0.43-0.60, Hc −0.04 to −0.20 (550: −0.36 at D 400): healthy regime, no collapse. All within the twins' noise at equal D.

## Game-rules match, save_3 (81 k): see matches/save_3.md

Bad in absolute (50-56 % of tanks dry at 10 s, 0.87 touch/tank on side 1000) like any 81 k checkpoint; at equal index 550 save_3 is worse on reserve (87-88 % dry at 10 s), equal on touches/kills.

## Diagnosis

Established: the panorama breaks nothing (throughput, KL, curve ≡ twins). Not decided: the expected panorama effect (contact / approach at D ≥ 800) is not readable at D 400.
Markers: match save_10 (270 k) vs 550 s10: dry at 10 s ≤ 0.07, touches/tank ≥ 2.95 / 5.24, armed < 5° ≥ 0.13. D 1000 entry: armed time within 100 m of a target and cos velocity/target (`outputs/comparisons/2026-11-25_exploration/approche.py`, pandas absent from system python) vs 550 s28.

## Update 2026-11-26 06:36: 475 k, D 950

- Milestones: D 450 @ 114 k … D 900 @ 366 k, D 950 @ 404 k (550: 900 @ 404 k, 950 @ 680 k; 551: 950 @ 595 k; 545: 950 @ 404 k). D 950 held since 71 k without passing to 1000.
- At D 950: kill 17.5, hit 0.255, fire 0.435/s, kl 0.009, skip < 1 %, Hc −0.23, Hd 0.30 (550 at D 950: 17.2 / 0.284 / 0.31; 551: 17.2 / 0.297 / 0.31). Same level, reached earlier (one seed).
- Trigger rarefied like twins: fire 1.3 (D 650) → 0.43 (D 950).
- Matches save_10 and save_17: `matches/save_17.md`. In game 554 s17 ≡ 550 s26; no approach (cos ≈ 0), contact < 100 m 0.19 of armed time (550: 0.12-0.16).
- Diagnosis: the panorama produced no approach at D 950; the curriculum lead (~240 k) is not attributable on one seed.
- Markers: passage to D 1000; match at D 1000 entry then +200 k, same contact table.

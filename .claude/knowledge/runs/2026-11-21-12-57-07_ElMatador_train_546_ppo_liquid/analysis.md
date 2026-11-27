# Run 546 — arena-exit death + received-shot direction (point at 548 k, D 1000)

## Header

`ppo_liquid`, launched 2026-11-21 12:57, last line 18:11, **running**, index 548 k, D 1000 since 429 k, save_20. Throughput 105 k steps/h (3.35 M tank-steps/h). Code: `1720af12`, committed 3 min before launch, clean working tree. High confidence.

## What the run tests

Vs 545 (`d85cc90f`, same `config.json`): the tank dies on leaving the square of half-side 2000, and the proprioception gains 3 inputs (direction of the last shot received, held 2 s). Two **confounded** changes, plus the seed.

## CSV

D milestones identical to 545, 522 and 510 to the thousand (D 1000 @ 429 k): max curriculum cadence, separates nothing.
At equal D:

| D | kill 545 / 546 | fire (shots/s) | hit | r_hit ×1e-3 | r_rcv ×1e-3 | r ×1e-3 | Hc | kl |
|---|---|---|---|---|---|---|---|---|
| 300 | 24.6 / 24.6 | 1.25 / 1.79 | 0.170 / 0.163 | 10 / 11 | −3.4 / −3.8 | 4.2 / 4.5 | −0.53 / −0.49 | 0.037 |
| 400 | 24.7 / 26.5 | 1.00 / 1.72 | 0.151 / 0.146 | 8.8 / 13 | −2.9 / −4.3 | 2.9 / 4.1 | −0.73 / −0.96 | 0.068 |
| 500 | 21.5 / 25.1 | 0.86 / 1.25 | 0.128 / 0.136 | 4.7 / 8.9 | −1.6 / −3.0 | 0.4 / 1.3 | −0.69 / −0.85 | 0.064 |
| 600 | 23.1 / 24.8 | 0.99 / 1.06 | 0.136 / 0.141 | 5.1 / 7.4 | −1.7 / −2.5 | −0.1 / 0.2 | −0.66 / −0.57 | 0.054 |
| 700 | 23.6 / 23.2 | 0.53 / 0.90 | 0.147 / 0.135 | 6.4 / 5.9 | −2.1 / −2.0 | 0.0 / −0.5 | −0.26 / −0.37 | 0.027 |
| 800 | 23.9 / 23.4 | 1.09 / 0.73 | 0.162 / 0.158 | 5.3 / 5.7 | −1.8 / −1.9 | −0.2 / −0.6 | −0.67 / −0.22 | 0.023 |
| 900 | 24.4 / 24.9 | 0.84 / 1.14 | 0.213 / 0.194 | 6.6 / 7.0 | −2.2 / −2.3 | 0.7 / 0.1 | −0.06 / −0.16 | 0.018 |

At D 1000, same index windows:

| index | kill 545 / 546 | fire | hit | r_hit ×1e-3 | r ×1e-3 | miss (m) |
|---|---|---|---|---|---|---|
| 429-470 k | 22.5 / 23.8 | 0.49 / 0.55 | 0.285 / 0.244 | 6.2 / 7.0 | 1.8 / 1.1 | 78 / 86 |
| 470-510 k | 21.3 / 21.2 | 0.38 / 0.38 | 0.319 / 0.310 | 5.2 / 5.1 | 2.0 / 1.7 | 72 / 73 |
| 510-549 k | 19.9 / 23.0 | 0.25 / 0.43 | 0.336 / 0.304 | 4.4 / 5.5 | 1.9 / 2.0 | 62 / 67 |

Phases: 0-110 k (D ≤ 400): kill peak 27.3 at 67 k, held 26-27 to D 400, shots 1.7/s. 110-270 k (D 450-650): kl above target (max 0.130 at 134 k, mean 0.106 at D 450), Hc −0.97, no skip; kill slides 26 → 23.7, r falls to 0. 270-430 k (D 700-950): kl 0.02, Hc back toward 0, r slightly negative (min −0.7e-3 at 307 k), hit 0.135 → 0.213. D 1000: fire halved (0.93 → 0.45), hit 0.21 → 0.31, kill 22-24, r back to 2.0e-3. α_canon_x rises to 0.031 (545: 0.037 at same index).

## Diagnosis

Established: stronger start at D 400-600: kill +1.7 to +3.6, r_hit +45 to +90 %, shots +7 to +70 %, held over five consecutive levels. 545 dipped to 21.5 (D 500); 546 never under 24.7. More combat on both sides in that range: r_rcv −2.5 to −4.3 vs −1.6 to −2.9. Same trajectory afterwards: D 650-950 identical ±1 kill, same trigger closing at D 1000, same α_canon_x rise. More marked kl episode: 0.106 at D 450 (target 0.05), without skip nor kill drop.
Noise / not settled: at D 1000 546 fires more (0.43 vs 0.25 at 510-549 k) and kills more (+3) on the last window only; hit/shot 3 points lower; one window, one seed. 545 held kill 19-22 and fire 0.2-0.3 from 549 k to 873 k: the marker for what follows. Share of arena-exit deaths and use of the received-shot direction are not readable in the CSV.
Markers: kill ≥ 22 and fire ≥ 0.4 at D 1000 beyond 600 k (545: 19-22, 0.2-0.3). Probes save_9 / save_16 / save_20 vs 545 save_9 / save_16: probes/timeline.md.

## Update at 743 k

| index | kill 545 / 546 | fire | hit | r_hit ×1e-3 | r ×1e-3 |
|---|---|---|---|---|---|
| 539-594 k | 18.8 / 22.1 | 0.22 / 0.26 | 0.374 / 0.321 | 3.7 / 5.4 | 1.8 / 2.0 |
| 594-649 k | 21.7 / 21.7 | 0.27 / 0.33 | 0.387 / 0.358 | 4.7 / 5.2 | 2.4 / 2.4 |
| 649-704 k | 19.9 / 21.9 | 0.21 / 0.28 | 0.378 / 0.343 | 5.1 / 5.7 | 2.5 / 2.5 |
| 704-743 k | 20.3 / 17.4 | 0.26 / 0.17 | 0.403 / 0.413 | 4.6 / 3.2 | 2.3 / 1.7 |

Kill 22-24 held to 680 k, then 19 at 690 k and 17 from 700 to 740 k, with fire 0.10-0.22 and r_hit −45 %. kl 0.018, skip 0, Hc > 0, ev 0.6-0.8: no KL regime nor σ collapse. Probes save_24 / save_27 (probes/timeline.md) give the cause: the trigger closes at constant shape while aim progresses.
Verdict: the D 400-600 lead is not found at D 1000. 546 follows 545's trajectory (closing trigger, kill ~20) and goes under at 700 k. Effect of the two additions on the cruise regime: not established.

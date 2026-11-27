# Run 545 — isolated control of Φ₂ (point at 456 k, D 1000)

## Header

`ppo_liquid`, launched 2026-11-21 03:51, last line 08:22, **running**, index 456 k, D 1000 since 429 k, save_16. Throughput 101 k steps/h (3.23 M tank-steps/h). Code: `d85cc90f`, committed 03:50:52, one minute before launch; `config.json` matches its message (target_kl 0.05, 2 shells per touch, canon entropy target 0). High confidence.

## What the run tests

Separate canon credit on potential Φ₂ (miss bell + bearing slope, weighted by urgency 50 m / distance, max over all enemies), rest GAE, reward unchanged. Settings back to 522's: target_kl 0.05, 2 shells per touch, empty penalty −D/1000, canon entropy target 0, canon σ ceilings removed.
Gaps confounded with 522: Φ₂ instead of Φ, linear command (no a·|a|), flat CNN, current image, actor batch 2400, kills curriculum. Vs 544 (killed at 109 k): target_kl, shells per touch, canon entropy target, empty penalty.

## CSV

D milestones identical to 510 and 522 to the thousand (D 1000 @ 428.8 k in the three): the curriculum advances at max cadence, separates nothing.
At equal D, first 31 rows of D 1000 (≈ 27 k steps):

| D | run | kill | hit | fire (shots/s) | r ×1e-3 | r_hit ×1e-3 | penalties / r_hit | skip_canon |
|---|---|---|---|---|---|---|---|---|
| 700 | 545 | 23.6 | 0.147 | 0.53 | 0.04 | 6.4 | 0.66 | 0.04 |
| 700 | 522 | 18.0 | 0.276 | 0.19 | 1.9 | 4.6 | 0.23 | 0.26 |
| 700 | 510 | 21.2 | 0.218 | 0.40 | 4.9 | 7.6 | — | 0.25 |
| 800 | 545 | 23.9 | 0.162 | 1.09 | −0.2 | 5.3 | 0.70 | 0.02 |
| 800 | 522 | 18.1 | 0.295 | 0.27 | 1.6 | 3.8 | 0.22 | 0.32 |
| 800 | 510 | 21.0 | 0.219 | 1.01 | 4.1 | 6.5 | — | 0.27 |
| 900 | 545 | 24.4 | 0.213 | 0.84 | 0.6 | 6.6 | 0.57 | 0.01 |
| 900 | 522 | 15.2 | 0.359 | 0.25 | 1.4 | 2.9 | 0.15 | 0.31 |
| 900 | 510 | 20.3 | 0.212 | 0.87 | 4.4 | 7.0 | — | 0.25 |
| 1000 | 545 | 22.7 | 0.273 | 0.51 | 1.7 | 6.4 | 0.40 | 0.002 |
| 1000 | 522 | 12.3 | 0.367 | 0.19 | 1.1 | 2.3 | 0.16 | 0.32 |
| 1000 | 510 | 17.9 | 0.203 | 0.79 | 3.8 | 6.0 | — | 0.24 |

(510 has no empty penalty: its `r` is not comparable.)
Trajectory of 545: kill 21.5-26.3 at all D levels, no trough; 22.1-22.6 on the last rows. hit 0.13-0.15 until D 800, then continuous rise: 0.18 (D 850), 0.21 (D 900), 0.24 (D 950), 0.27 (D 1000), 0.30 at 456 k (run max, still rising). fire 1.0 → 0.5 shot/s between D 800 and D 1000 while r_hit stays 6.4-6.6e-3: fewer shots, as many touches. Interval between two emptyings per tank (penalty / |rest| × 0.1 s): 17 s (D 700), 21 s (D 800), 24 s (D 900), 39 s (D 1000), 45 s on the last row; 522 at D 1000: 276 s. kl 0.044-0.052 from D 350 to D 600 (at target without skip), 0.015-0.023 from D 650; skip and skip_canon ≈ 0; ev 0.65-0.73 from D 900 (max 0.87). α_canon_x ×25 (1e-3 → 0.025, flat since 446 k), α_canon_y ×9, α_dir_x ×4; Hc −0.7 (D 350-650) → +0.25 at 452 k. α_fire and α_zoom at floor 1e-4, Hd 0.69 → 0.40.

## Diagnosis

Established: best D 1000 entry of the three runs: kill 22.7 vs 17.9 (510) and 12.3 (522), at intermediate hit. The criterion "kill ≥ 17 at D 1000" is held, over 27 k steps only. No trigger closing at 240-430 k, whereas all families showed one: fire never lastingly below 0.5, kill has no trough. The canon KL stop no longer bites (skip_canon 0.002 vs 0.24-0.32 in 510/522). The dump is stronger than in 522 at each D (penalties = 40-70 % of r_hit vs 15-23 %) but resorbs from D 900; it keeps `r` near 0 up to D 850.
Noise / not settled: attribution to Φ₂ confounded with the five other gaps vs 522, one seed; entropy regulation pushes the canon (α_canon_x ×25, Hc from −0.7 to +0.25): the CSV does not say whether canon σ widened on both axes nor whether precision suffers; hit rises meanwhile; CSV says nothing of the turret with closed trigger (marker ≥ 30 % < 5°) nor of p_al / p_off.
Options/markers: let it run: 522's plateau only came at 960 k; judge holding of kill ≥ 17 and hit over 100-200 k more at D 1000. Probe save_9 (243 k) and save_16 (432 k), sides 550/1000, vs 510 s9/s28 and 522 s9/s16. Alerts: α_canon_x back above 0.025 with falling hit; fire < 0.25 shot/s side 550; kill < 18; emptying interval back under 30 s.

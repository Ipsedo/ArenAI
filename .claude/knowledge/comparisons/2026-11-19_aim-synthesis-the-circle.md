# Aim synthesis 2026-11-19 — « the circle »: 2×2 CNN × aim credit on kills at D 1000

Runs: 500, 510, 514, 520, 522, 524, 525, 528, 529, 538, 523-538

2026-11-19: 2×2 table CNN × aim credit on kills at D 1000; only separate credit on Φ ever held aim, 523-538 = variations without it; reread before any proposal on aim.

Synthesis requested by Samuel 2026-11-19 (« on tourne trop en rond, je perds le fil »), CSVs reread at equal D:

| | separate aim credit (Φ, λ 0, no critic) | single advantage |
|---|---|---|
| spatial softmax | 510/514/520/522: kill 19-22 at D 1000, hit 0.18-0.38 | 500: kill 7 |
| flattened CNN 7 convs | control 430 other PC (cfb76a2e): kill 17.8, hit 0.38, tour<5° 40 % | 524/525/528/529: kill 8-13 from D 750-1000 |

- 538 (white noise, batch 2400, linear command) at D 650 (279 k): kill 20.8, hit 0.17, fire 0.57 falling = trajectory of 528 at the same D (22.4 / 0.19 / 0.45, then 12.7 at D 750). Prediction made to Samuel: drop before D 1000; if kill ≥ 17 at D 1000, the synthesis is false.
- The circle: removed and never restored = separate credit (2026-11-16, 5c338159) then Φ (2026-11-17); the 15 runs 523-538 varied target_kl, batch, kills curriculum, economy 5, σ ceiling, short head, γ, critic lr, coloured noise, softmax, linear command.
- Concordant measurements: ρ(events, gesture) ≤ 0.03 under five angles; critic blind to alignment (530/531/532); sweep = acquisition (comparisons/2026-11-19_aim-gesture-state-vs-events-signal.md); no run ever steered the turret toward the target, the best sweep slowly and fire selectively.
- Samuel's hypothesis of 2026-11-16 (« avec un CNN qui localise, l'avantage unique suffira », single-advantage run 523) is refuted by 524-538.

Options presented 2026-11-19 (Samuel's decision awaited): A restore separate credit + Φ on the current tree, 538 as control; B same structure with a potential learned from shot outcomes (events only, classed « pas ouf » on 2026-11-16, bound ρ 0.08-0.09, one-step aim critic measurement 522 s68), measurable offline before code; C stay pure-event = no measured lever remaining.

**Rule:** no longer propose tuning under single advantage as the solution to aim; judge any run on kill and hits/tank-s at D 1000 against this table. (see docs/decisions.md 2026-11-22: separate aim credit removed; 2026-11-24: separate aim credit on Φ refused)

**Measurement of B (2026-11-19 afternoon, report `outputs/comparisons/2026-11-19_bruit_avantage_visee/rapport.md` §7, script `hhead.py`)**: Samuel sets A aside (« Φ biaise la politique de visée ») and asks for the measurement of B. On 538 s12, H = share of the fire's shells that hit: read from geometry ρ 0.147 at one step (> Φ); learned from the image (7-block CNN from scratch, 36 k shots) ρ ≈ 0 at one step, 0.056 at two steps, flat from 9 to 36 episodes while hit AUC rises 0.68 → 0.75 (geometry 0.865). Criterion 0.08 not reached. B's lock is reading the image, not the label. Pitfall: one episode = 1 801 decisions (max_episode_steps + 1), `credit.load` assumes 1 800; image key = (t // 1801, t % 1801, tank). Tool: `sigma_gradients --frames_fired`, columns `fired_shells;landed_shells;nb_hits` (uncommitted). 538 at 379 k: D 700, kill 18.3, fire 0.46.

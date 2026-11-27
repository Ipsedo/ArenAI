# Run 539 — current image (538 config)

`train_539_ppo_liquid`, ppo_liquid, launched 2026-11-19 16:13, **running**; read at index 351 k (19:49), D 800. Throughput 97 k decisions/h (3.1 M tank-steps/h). Code: b83da585 + uncommitted tree, binary `cmake-build-release` recompiled 16:12 (high confidence: binary and `libarenai_view.so` mtime after the fix).

## What the run tests

One change vs 538: `VulkanOffscreenRenderer::draw_and_get_frame` renders the current step's image instead of the previous decision's. Config diff reduces to `noise_correlation` (option removed; 538 was already at 0, white noise). Removing the coloured-noise code has no behavioural effect. One run per arm.

## CSV

D milestones: identical to 538 per index up to D 700 (281 k), both runs passing every step at first check. Then 538 stays 120 k at D 700 (D 750 at 404 k, D 800 at 429 k); 539 chains D 750 at 305 k and D 800 at 328 k, a 100 k lead.
Means at equal D (538 / 539):

| D | kill | hit | fire (shots/s) | miss (m) | r excluding hit and received |
|---|---|---|---|---|---|
| 350 | 22.9 / 24.8 | 0.113 / 0.122 | 1.73 / 1.85 | 35 / 39 | −0.0035 / −0.0033 |
| 450 | 21.7 / 25.8 | 0.137 / 0.113 | 1.19 / 1.85 | 37 / 49 | −0.0027 / −0.0050 |
| 550 | 22.6 / 25.2 | 0.153 / 0.115 | 1.05 / 1.59 | 44 / 59 | −0.0025 / −0.0053 |
| 600 | 22.0 / 25.1 | 0.167 / 0.128 | 0.95 / 1.16 | 44 / 60 | −0.0023 / −0.0048 |
| 650 | 20.7 / 24.0 | 0.176 / 0.157 | 0.56 / 0.56 | 44 / 60 | −0.0018 / −0.0036 |
| 700 | 17.4 / 20.8 | 0.203 / 0.170 | 0.56 / 0.65 | 45 / 61 | −0.0009 / −0.0021 |
| 750 | 19.5 / 20.5 | 0.176 / 0.181 | 0.93 / 0.58 | 47 / 61 | −0.0017 / −0.0014 |
| 800 | 18.8 / 17.8 | 0.175 / 0.193 | 0.62 / 0.30 | 48 / 61 | −0.0017 / −0.0010 |

539 trajectory: 0 to 145 k (D 250 to 400): climb, kill max 26.8 (138 k), fire up to 3.2/s (142 k). 145 to 267 k (D 450 to 600): kill held 24-26, but r falls to 0.0005 (D 550) while r_hit + r_rcv stays at 0.006: the rest of the reward (kill, death, empty-reserve penalty) costs twice as much as in 538; ev hollowed (0.22 mean at D 500, min −3.6 at 200 k). 267 to 351 k (D 650 to 800): fire 1.16 → 0.56 → 0.30/s, kill 25.2 → 15.5, hit per shot 0.13 → 0.20, ev 0.80-0.92.
Healthy optimisation throughout: kl 0.007-0.018, skip < 1 %, clip 0.09-0.19. Hc −0.08 to −0.61, α canon at floor; α dir left floor then returned.

## Diagnosis

Established: curriculum faster: D 800 at 328 k vs 429 k, without the 120 k plateau at D 700. From D 350 to D 700, 539 makes 2 to 4 more kills per episode than 538, with lower hit per shot and higher cadence: more volume, not more precision. The cadence then collapses as in 538 (0.39/s at D 850, kill 15.9 then 11.5). At D 800 539 averages 17.8 and ends at 15.5: at 538's level, not above. Probes save_3 and save_5: dwell under 5° doubled, corrective gesture and touches per shell unchanged.
Probable, not verified by probe: the r trough between D 450 and D 600 comes from reserves emptied more often (cadence 1.6-2.0/s, penalty −D/1000 per empty reserve). The trigger closing from D 650 would be the answer to this penalty, which grows with D; miss at 60 m vs 45 m: shots from farther or less aligned.
Not settled: whether doubled dwell still holds at D 800, whether the closing trigger is selective or only rarer; arrival at D 1000: climb criterion is 0.5 kill per tank in probe episode = 16 kills; 539 at 15.5, at threshold; 538 stalled at 850 in the same situation.
Markers: probe save_12 (324 k, D 800), sides 550 and 800; kill and fire at D 800-850, under 9.6 kills (0.3/tank) the curriculum recedes; at D 1000: separate credit 17.8-22 kills; single advantage 7-13.

## Conclusion at 447 k (2026-11-19 ~21:00, probes save_3, 5, 8, 10, 12), side 550 (4 episodes, stochastic)

| save | index | D | kills | dwell < 5° | corrective gesture | σ canon | off-axis p_fire | hit/shell | disarmed |
|---|---|---|---|---|---|---|---|---|---|
| s3 | 81 k | 350 | 24 | 0.62 s | 0.59 | 0.11 | 0.73 | 7.5 % | 53 % |
| s5 | 135 k | 400 | 20 | 0.73 s | 0.59 | 0.08 | 0.67 | 9.1 % | 38 % |
| s8 | 216 k | 550 | 25.5 | 0.49 s | 0.56 | 0.22 | 0.18 | 12.9 % | 15 % |
| s10 | 270 k | 650 | 23 | 0.28 s | 0.52 | 0.23 | 0.05 | 18.9 % | 6 % |
| s12 | 324 k | 800 | 21 | 0.24 s | 0.51 | 0.20 | 0.03 | 20.9 % | 4 % |

Touches per shell by angle and distance at save_12 (sides 550 and 800 merged), with armed p_fire:

| angle | < 40 m | 40-100 m | 100-200 m | > 200 m |
|---|---|---|---|---|
| < 1° | 100 % (p 0.66) | 70 % (0.35) | 53 % (0.08) | 12 % (0.05) |
| 1-2° | 100 % (0.60) | 72 % (0.37) | 38 % (0.09) | 23 % (0.04) |
| 2-5° | 92 % (0.61) | 53 % (0.26) | 17 % (0.05) | 14 % (0.03) |
| 5-10° | 79 % (0.47) | 20 % (0.11) | 6 % (0.04) | 4 % (0.04) |

- The image fix gives doubled target holding as long as the trigger is open everywhere (s3, s5), faster curriculum (D 800 at 328 k vs 429 k) and 2 to 4 more kills up to D 700. It does not change the corrective gesture (0.51-0.59, never learned) nor the outcome.
- At fixed side 550 the policy does not collapse: 20 → 25.5 → 21 kills. The CSV fall comes from the curriculum, which raises D while probe episodes exceed 16 kills: D regulates around 750-850, where kills fall to 16. 538 stabilised the same way.
- Between s5 and s8 the trigger becomes selective and σ canon rises to 0.2 at the same time; target holding then erodes as the trigger closes (0.49 → 0.28 → 0.24 s).
- Beyond 100 m only a shot under 2° pays for the reserve (threshold 20 %, +5 shells per touch). The trigger is closed there (0.04-0.09) and flat in angle; the turret spends 4 % of armed decisions there. This is the regime missing at D ≥ 800.

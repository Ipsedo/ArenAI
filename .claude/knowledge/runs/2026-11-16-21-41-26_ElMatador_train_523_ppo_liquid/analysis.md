# train_523_ppo_liquid — analysis at 69 k (2026-11-16 22:36, RUNNING)

## Header

ppo_liquid, launched 2026-11-16 21:41, last CSV line 22:35: index 69 k, D 400, 0.91 h. Throughput 76.4 k steps/h (2.44 M tank-steps/h); 522: 86.7 k steps/h over its whole run (−12 %, to attribute to the 7-conv CNN, not isolated). Effective commit **5c338159** (high confidence). Binary `cmake-build-release` 21:39:50, started 21:41:25, committed 21:50; `config.json` without `aim_gae_lambda`, CSV without `skip_canon`.

## What the run tests

Two changes vs 522, **confounded** on this PC: (1) CNN: spatial softmax → 7 conv blocks + Flatten (cfb76a2e); (2) single advantage: separate aim credit removed (one ratio, critic GAE for all actions, one KL stop at 1.5 × target_kl = 0.075); Φ stays in reward. The run isolating (2) is the other PC's (cfb76a2e as is): without it nothing below is attributable to either. Other code diffs 9d843717..5c338159 already in 522 (empty penalty ∝ D, 2 shells per touch): identical config there.

## CSV

D milestones: 300 @ 20 677, 350 @ 47 647, 400 @ 67 425 — identical per index to 510, 520, 522 (and 514). In this lineage equal D = equal index: curriculum does not discriminate runs at this stage.

| D (index) | run | kill | hit | fire | r_hit | s | miss | ev | Hc | kl | clip |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 300 (21-47 k) | **523** | 13.4 | 0.094 | 1.02 | 4.4e-3 | 61 | 24.7 | 0.56 | +0.30 | 0.030 | 0.29 |
| | 522 | 16.6 | 0.095 | 1.53 | 6.2e-3 | 55 | 26.4 | 0.31 | −0.14 | 0.050 | 0.36 |
| | 520 | 17.5 | 0.093 | 1.70 | 6.4e-3 | 46 | 30.4 | 0.31 | −0.18 | 0.067 | 0.37 |
| | 510 | 17.6 | 0.119 | 1.70 | 7.9e-3 | 55 | 26.0 | 0.36 | −0.38 | 0.045 | 0.36 |
| 350 (48-67 k) | **523** | 15.9 | 0.112 | 1.21 | 5.5e-3 | 57 | 28.2 | 0.54 | +0.24 | 0.030 | 0.28 |
| | 522 | 19.6 | 0.103 | 1.62 | 6.8e-3 | 47 | 36.2 | 0.18 | −0.26 | 0.065 | 0.40 |
| | 520 | 22.4 | 0.112 | 1.69 | 8.2e-3 | 48 | 34.1 | 0.38 | −0.27 | 0.077 | 0.40 |
| | 510 | 22.0 | 0.130 | 1.76 | 9.8e-3 | 48 | 33.8 | 0.30 | −0.55 | 0.112 | 0.44 |

D 250 (0-20 k): kill 10.9 (522: 11.1; 520: 8.8; 510: 9.5): start within band.
523 trajectory: kill 7.9 → 12.3 (14 k) → plateau 12.5-13.4 to 41 k → 16.9 (69 k), still rising. fire 0.85-1.1 to 41 k then 1.2-1.4 (max 1.72 @ 68 k). Hc rises to +0.34 (28 k) then falls to +0.16 (69 k). The 6 α reach floor 1e-4 between 48 and 52 k.
`skip`: 256-rollout window (≈ 77 k steps) not yet full ⟹ column is a cumulative mean. Instantaneous value rebuilt (difference of cumulatives): 22 k 0.03, 28 k 0.05, 33 k 0.16, 44 k 0.31, 49 k 0.40, 60 k 0.42, 66 k 0.45, 70 k 0.46. 522 same index: skip (rest) 0.000, skip_canon 0.12-0.15.

## Diagnosis

Established: precision equal, volume lower (hit ≡ 520/522 at both levels; kill deficit −19 to −29 % at D 350 follows fire deficit −25 to −31 %: delay in firing volume, not aim). The single KL stop cuts ~45 % of actor minibatches from 49 k although mean kl (0.030) and clip (0.28) are lower than in 510-522; in 522 the cut touched only the canon group (12-15 %) and never the rest; here the whole policy (trigger, direction) is cut with it: structural effect of 5c338159 (single KlStop). Better critic: ev 0.54-0.56 vs 0.18-0.38, v 0.036 vs 0.07-0.10. No σ contraction wave: Hc stays positive where 510-522 go to −0.14…−0.55.
In the noise / not attributable: higher ev (localizing CNN or sparser reward: both readings hold); kill delay (learning speed reduced by skip, or simple shift of fire rise: 41 k here, from 20 k in 510-522; kill still rises at 69 k); longer s (57 vs 47) and shorter miss (28 vs 34-36): consistent with fewer shots, not proof of better aim.
Not settled by CSV: trigger selectivity, dump, turret alignment (tour<5°), yaw decodability — the question this run poses, needs a probe.
Options noted (no decision): let it run (useful verdict at D 400-450 (67-145 k) where 510/520/522 plateau at kill 22-24; if 523 rejoins them with fire < 1.5 it does so at lower volume); probe save_2/save_3 with `--features` (yaw decoding vs 522 s68 0.56-0.63, tour<5°, decision ratio); if instantaneous skip > 0.6-0.7 or kill stagnates under 20 at 110 k the single KL stop becomes suspect (levers target_kl, actor_epochs to discuss, not before); comparison with the other PC's run (CNN alone) at same index is the only way to separate CNN and single advantage.
Markers: skip: read instantaneous value to 77 k, the column after; Hc +0.16 and falling: passage below 0 with α at floor = delayed σ contraction; fire > 1.6 lasting = volume catch-up of previous runs (dump possible, probe); planned measure ~200-300 k: ρ(advantage, corrective gesture), threshold 0.08.

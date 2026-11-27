# Run 423 — PPG

- Launched 2026-11-12 23:46 → last row 2026-11-13 08:15, **running** (8.5 h).
- Final index 581k, 68.4k steps/h (2.19 M tank-steps/h, 32 tanks); 422 ran at 74.8k steps/h.
- Effective commit **802b1b32** (high confidence): commit 23:45:43, binary 23:46:37, `config.json` 23:46:38, clean tree.
- Saves: save_21 ≈ 581k (~27.7k steps/save); D=1000 reached around save_15.

## What this run tests

Config identical to 422. Only code change: **7422d098, per-action-group KL stop**.
Before, a single `target_kl` on the joint KL cut the whole actor; now the aim group
(canon_x, canon_y, critic-less φ advantage) and the rest group (displacement, fire, zoom, critic
advantage) are each stopped on their own KL (> 1.5 × 0.05). `skip` = skip of the rest group,
`skip_canon` = skip of the aim group, `kl` = joint KL.

Inherited from 422 (1e171bbf), hence **confounded** with 423 against 421: canon stabilisation w.r.t.
chassis movements, aim advantage = potential φ (theoretical shell trajectory) with
`aim_gae_lambda = 0`, entropy targets [0, 0, 0, 0], return to delta aim.

422 was stopped at 148k (D=500): there is no long 422 to isolate the per-group stop.

## CSV reading

D milestones: 300 @ 21k, 400 @ 67k, 500 @ 145k, 600 @ 208k, 700 @ 267k, 800 @ 305k, 900 @ 366k,
**1000 @ 429k** — first run to reach the curriculum's final bound (421: D=400 at 145k,
450 at 566k; best SACs: D=400).

| D | hit | fire (/s) | kill | s (s) | miss (m) | ev | Hc | skip | skip_canon |
|---|---|---|---|---|---|---|---|---|---|
| 400 | 0.126 | 1.64 | 22.6 | 52 | 36 | 0.71 | −0.46 | 0.15 | 0.10 |
| 500 | 0.126 | 1.55 | 19.5 | 52 | 39 | 0.75 | 0.11 | 0.01 | 0.00 |
| 700 | 0.142 | 1.14 | 21.8 | 59 | 54 | 0.75 | 0.11 | 0.03 | 0.01 |
| 900 | 0.171 | 1.01 | 22.1 | 45 | 58 | 0.74 | 0.05 | 0.09 | 0.02 |
| 1000 (429k → 581k) | 0.193 | 0.81 | 21.2 | 48 | 64 | 0.74 | 0.06 | 0.03 | 0.03 |

- **hit rises with D** instead of dropping: 0.126 (D=400) → 0.193 (D=1000), last 0.217 @ 581k,
  run record. Slices at D=1000: 0.186 → 0.199 → 0.194, still slightly rising.
- **fire drops** 1.6 → 0.8 /s while hit rises. Hits/s per living tank ≈ fire × hit:
  0.21 (D=400) → 0.16 (D=1000), in a 6× larger arena; kill/episode flat ~21.
- miss 22 → 64 m grows with D: geometric, not an aim degradation.
- Critic stable (ev 0.70–0.75 all run, v ~0.04).
- Hc: trough −1.34 @ 59k (D=350), then rises to ~+0.1 from D=450 and stays. Canon α active
  (0.01–0.035, above floor); dir α at floor 1e-4 since ~400k; fire/zoom α at floor.
- KL: KL regime only at D=300–400 (kl 0.14, clip 0.5, skip 0.11–0.15), then kl 0.02–0.05,
  clip ~0.21, skip < 0.1.

### Against 422 (same config, joint KL stop) over 0–148k

D milestones identical to the log point (300 @ 21k … 500 @ 145k), hit identical within noise
(D=400: 0.126 vs 0.123; D=450: 0.131 vs 0.119). Clear differences:
- skip: 423 ≤ 0.15 vs 422 0.47 (D=400) → 0.58 (D=450) — in 422 the canon KL cut half
  of the rest group's steps;
- Hc: 423 rises to +0.16 at D=450, 422 stays at −1.05;
- clip 0.24 vs 0.38 at D=450.

## Diagnosis

**Established**
- Qualitative jump vs 421: curriculum unlocked up to D=1000 and hit ≈ 2.5× 421's level
  at D=400 (0.077), no more chronic KL regime, no σ collapse. Attributable to the package
  1e171bbf + 7422d098; the two are confounded beyond 148k.
- 422's joint KL stop really cut the rest group (skip 0.5–0.6); the per-group stop
  frees it (skip < 0.15).
- The curriculum is saturated: hit 0.19 ≫ ratio_high 0.08. D measures nothing since 429k; only
  hit, fire and kill at D=1000 remain informative.

**Probable, not proven by the CSV**
- fire ↓ with hit ↑ looks like a trigger becoming selective (fire less, better). The CSV does not
  prove it: `fire` is blind to the episode-start dump and says nothing about alignment at fire.
- Up to 145k, the per-group stop did not change the curriculum speed; unknown whether 422
  would also have climbed to 1000 (stopped too early).

**In the noise**
- fire oscillations 0.39 → 1.89 /s within a few k (256-decision window); Hc oscillates ±0.3 at the end of run.
- Throughput loss −8 % vs 422: consistent with the double gradient pass when a group is stopped,
  but at this level it may also be machine noise.

## Options (Samuel's decision)

1. **Probe a checkpoint at D=1000** (save_18–21): trigger selectivity (angle at fire vs off-fire,
   armed p(fire)), corr(μ canon, error), hit/shell — the question the CSV does not settle,
   and the first run where the probe has something to find. Compare to save_5 (D≈450).
2. **Let it run** to see whether hit plateaus at D=1000 (marker: next slice ≥ 0.20).
3. The curriculum has no margin left: extending `final_spawn_*` or hardening the task otherwise is a
   design decision.
4. To attribute the gain: replay 422 (joint KL stop) beyond 300k.

## Markers

- hit at D=1000: progress or plateau over the next 100k.
- fire: if it continues under 0.6 /s with kill falling, watch for trigger extinction.
- skip_canon / canon α: rise = aim starts moving strongly again.
- ev < 0.6 or Hc < −0.5 sustained: regime change.

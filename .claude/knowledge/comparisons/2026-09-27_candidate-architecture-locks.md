# Candidate architecture locks (liquid cell low-pass, CNN squashed to 1×2) — none demonstrated

Runs: 391, 392, 405_zoom, 407, 409, 414

2026-09-27 — liquid cell = low-pass τ≈0.6 s, CNN squashed to 1×2; but common hit ceiling ~8-9 % for vanilla/liquid and CNN 1×2/2×4 ⟹ none demonstrated.

Samuel's question: « il n'y a pas des verrous dans l'architecture ? ». Reliable measurements (parameters and geometry, no assumed state distribution — cf. the error of comparisons/2026-09-24_dead-trigger-deterministic-threshold.md).

**Liquid cell = low-pass filter.** `liquid_cell.cpp`: semi-implicit Euler LTC, 6 sub-steps of 1/180 s, `x ← (x + h·f·a)/(1 + h(1/τ + f))`, and the heads see **only** x (no direct input → output path). Effective constant `τ_eff = 1/(1/τ + f)`, fraction updated per frame `1 − exp(−(1/τ+f)/30)`: 5 % at f=0, 8 % at f=1, 29 % at f=10. τ measured 414 save_155: **0.45-0.77 s (median 0.60)**, barely moved from init `softplus(0)` = 0.69. Input weights ×8 (row norm 0.196 → 1.62). Real f unknown without real observations.

**Vision.** 128×256, vertical FOV π/4 ⟹ horizontal ≈ 79°, ≈ 3.2 px/°. 7 convs stride 2 ⟹ final map **1×2** (128 px cells ≈ 40°), the aim point falls on the boundary of the two. Required precision: tank ~5 m at 60-90 m (nearest-neighbour distance, 32 tanks) ⟹ ±1.6-2.4° ≈ ±5-8 px.

**What prevents concluding** — hit at equal curriculum: 405_zoom (CNN 6 convs, map 2×4) 0.071-0.088; 407/409/414 (1×2) 0.073-0.096; **vanilla PPO 391 (no recurrence, hence no delay)** 0.08-0.095 at D=525-800. Common ceiling ~8-9 %; 392 (vanilla, 0.114-0.125 at D=250-300) is confounded by the old reward with aim shaping (before 2026-09-13, comparisons/2026-09-09_event-only-reward-refactor.md).

**Rule:** do not blame the liquid delay nor the CNN squashing without measurement. Proposed test (not validated): C++ harness rendering real frames with the bearing of the nearest enemy ⟹ (a) linear probe bearing ← frozen CNN features (error > ~1.5° ⟹ vision is the lock), (b) real step response of the actor (delay in frames). See run 414 (plateau, σ init). (later: 2026-11-16 CNN bench comparisons/2026-11-16_cnn-bench-spatial-softmax.md; docs/decisions.md 2026-11-10 perception ruled out)

# train_493_ppo_liquid

| | |
|---|---|
| machine | ElMatador |
| user | samuel |
| launch | 2026-11-07 20:11:47 |
| commit at launch | `b5309399` — reward redesign : damage fractions, aim potential, fire cost, no starvation |
| commits ≤ 90 min after | `401b7874` |
| diff at launch | unknown (identity recorded after the run ended) |
| raw data | `outputs/train_493_ppo_liquid` on ElMatador |

## Established facts

- 2026-11-07: Launched by Samuel 20:11 from `cmake-build-release`; only difference vs 492: `--curriculum_probe_window 8 --curriculum_boundary_proba 0.5` (became `main.cpp` defaults, with `save_every` 5 min). Throughput 85 k/h.
- 2026-11-07: Curriculum (read at 243 k): D 250 → 300 at 47 k (20:45), → 350 at 67 k (20:59), → 400 at 207 k (22:38). First two steps 15 min each, then 140 k (1 h 40) on 350: CSV `hit` (all distances) stayed 0.085-0.09 but ratio of probe episodes played at the boundary was lower and lingered in the dead band 0.05-0.08. Intended behaviour. `side` oscillates between boundary and easy (268-350) as expected with boundary 0.5.
- 2026-11-07: Probes (4 episodes, side 250 unless noted), p_fire aligned (<5°,<40 m) / off-axis (>20°): save_3 (81 k) 0.812/0.781; save_6 (162 k) 0.743/0.648; **save_9 (243 k) 0.809/0.615, gap 0.19**; at side 400: 0.837/0.626, gap 0.21. Armed level 0.79 → 0.68 → 0.69, self-regulated; p rises with full reserve (0.864 at 0.8-1). Dump unchanged: reserve emptied 2.3-2.6 s, 11-15 respawns/episode, armed 35-42 %, 1.3-1.6 shot/s; hit/shell 10.4 → 11.1 → 12.0 % (8.8 % at side 400), aligned 26-30 %, shots <5° 12 → 15 %; share of decisions <5° 11 → 16 %. Canon: σ 0.27/0.23, |μ| > 0.9 on 1 % ⟹ **CSV Hc 0.005 = σ ≈ 0.25 per dim (differential entropy), not a collapse**. CSV: kill 13-14.6 at D=400, kl 0.02, skip 1.3 %. Comparison: 480 save_8 had wider gap (0.70 vs 0.17) but with epochs 2 and a turret that later started spinning; 492 plateaued 0.08-0.09 on save_3-6; 493 at equal data reaches 0.19 at save_9.
- 2026-11-08: Result at 450 k (save_15 at 405 k, save_16 at 432 k). Curriculum: → 450 at 267 k then nothing over 183 k (~9 windows), probe ratio at boundary 0.078 just under 0.08. CSV at D=450 per 22 k slice: hit 0.087 → 0.065, kill 11.7 → 9.6-10.7, fire 1.2, s 70-78; **skip 0.03 → 0.18, kl 0.024 → 0.031 (target 0.05), Hc −0.04 → −0.20, clip 0.29-0.31**: KL regime settling in like 441 (skip > 0.02 at 256 k, > 0.10 at 378 k, > 0.15 at 420 k). Probes side 250, p_fire aligned (<5°) / off-axis (>20°): save_9 0.748/0.615, save_15 0.833/0.681, **save_16 0.851/0.726 ⟹ gap 0.13 → 0.15 → 0.13 (0.17-0.19 with aligned <40 m), frozen**; armed level 0.69 → 0.76 → 0.79, p at empty reserve 0.82-0.84; dump 2.6 → 2.4 s, armed 42 → 33-37 %, 94-97 % of tanks empty at t = 5-10 s. **Aligned-shot enrichment = 1.07** (shots <5° 13-15 % vs armed decisions <5° 12-14 %): a 0.15 gap on a 0.75-0.85 level does not sort shots. Turret stable: decisions <5° 15-17 %, hit/shell 12.0 → 11.6 → 11.1 % (side 250), 8.8 % save_9 at 400 vs 7.2 % save_16 at 450 (aligned 29.5 → 27.6 %); σ canon 0.27/0.23 → 0.23/0.17, |μ| > 0.9 in x 1 → 5 → 9 % (μ drifts to bounds, start of 429). Verdict: second case of the open question: gap freezes ~0.15-0.2, contrast of centred hit ×3 does not carry the trigger under the dump; 480 save_8 (epochs 2) had 0.70 vs 0.17, a 3× larger contrast at a much lower level.
- 2026-11-08: **Structural cause found and fixed (validated « Oui, code-le »; fix ships in run 494).** In all three trainers δ = r + γ·V(s′) − V(s) with s′ = observation of the respawned tank (full health, 10 shells): emptying the reserve was worth a fresh tank's value, shells had no scarcity. Evidence in probes: p_fire at 1-2 shells > p_fire at 3-5 shells in every save (0.712/0.673; 0.786/0.742; 0.823/0.764; 0.804/0.733 at 450). See run 494.

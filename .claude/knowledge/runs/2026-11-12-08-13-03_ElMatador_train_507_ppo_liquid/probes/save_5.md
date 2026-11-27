# Probe 507 save_5 (135 k, D 450) vs 504 save_5 (D 400)

`build/` binary of 12/11 08:01, compiled from the tree committed afterwards as 25a75a34 (`--force`: script dates the binary to c02137c6). 4 episodes per side, stochastic, economy 10 / +2 / 20. 504 save_5 re-analysed from its CSVs (`--analyze-only`).

| run | side | p_arm | ratio | p_al / p_off | hit/sh | h<5° | empty_s | shots/s | turret<5° | σ cx/cy | \|μ\|cx | corr(yaw, μ_cx) ±40° (slope /10°) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 507 | 550 | 0.15 | 2.9 | 0.30 / 0.11 | 10.0 % | 18.4 % | 6.1 | 0.75 | 13.2 % | 0.45/0.49 | 0.15 | −0.38 (−0.051) |
| 504 | 550 | 0.20 | 3.5 | 0.53 / 0.15 | 8.4 % | 16.2 % | 5.9 | 0.86 | 13.8 % | 0.48/0.40 | 0.16 | −0.39 (−0.057) |
| 507 | 1000 | 0.12 | 3.7 | 0.33 / 0.09 | 5.1 % | 12.1 % | 8.2 | 0.60 | 9.6 % | 0.49/0.52 | 0.14 | −0.23 (−0.026) |
| 504 | 1000 | 0.15 | 5.0 | 0.61 / 0.12 | 3.9 % | 7.7 % | 5.7 | 0.69 | 12.7 % | 0.50/0.42 | 0.15 | −0.28 (−0.035) |

CSV at 135 k: 507 D 450, hit 12.4 %, kill 14.5; 504 D 400, hit 11.3 %, kill 11.

- **Turret: no better than 504.** turret<5° equal side 550 (13.2 vs 13.8 %) and lower side 1000 (9.6 vs 12.7 %). Target tracking in field is the same (corr −0.38 vs −0.39, gain ≈ 0.05 μ per 10°). The 10° potential did not strengthen the corrector at 135 k.
- **Hits per shell: better.** hit/shell ×1.2-1.3 and hits of aligned shots <5° more frequent (18 vs 16 %, 12 vs 8 %). Comes from what happens under 5° (fine precision, or pitch) not the share of aligned decisions. Undecided: probe does not measure pitch error.
- **Trigger: selective but less than 504.** Ratio 2.9-3.7 vs 3.5-5.0, via lower p_al (0.30 vs 0.53-0.61). 507 fires less (0.6-0.75 vs 0.7-0.86 shots/s), reserve empty at 6-8 s vs 6 s; dump persists (91-96 % of tanks empty).
- σ and μ in healthy regime in both runs (0 % at ceiling, 0 % saturated).

## Provisional verdict

The 10° aim potential did not, at 135 k, produce the turret it aimed at. CSV lead (D 450 vs 400, kills ×1.3) is explained by more profitable shots and a more economical trigger, not alignment. Next point: save_9 (243 k) vs 504 save_9 (turret<5° 12.5 %, hit/shell 16.1 %, ratio 6.1, reserve 22.4 s); if turret<5° stays ≤ 504 there, lever 3 does not act on the turret.

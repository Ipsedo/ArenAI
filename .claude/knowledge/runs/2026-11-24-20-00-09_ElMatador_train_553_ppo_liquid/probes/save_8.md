> Correction of 2026-11-25: the conclusions "better aim", "low canon gone", "dual active, not a pathology" come from the training environment, which respawns tanks at empty reserve. In game Samuel judges s36 bad (reserve empty very fast, canon pointing down). Deterministic, s36 empties its reserve in 5.9/6.2 s. The link between H_offset and α is not established (inconsistency on s36).

# Probe 553 save_8 — entropy of the vertical aim offset

## Header

Run `train_553_ppo_liquid`, save_8 = index 216 k, D 500 (CSV read at 236 k, D 550). Binary `build/arenai_agent/arenai_agent_probe` compiled 2026-11-24 23:13 at commit `95f1812c` (the run's), plus an uncommitted addition in `probe.cpp`: columns `aim_offset_entropy_x/_y` (= `SpatialAimLaw::offset_entropy()`, the estimator used by the α dual). No inference diff. Stochastic mode. save_8: sides 550 and 1000 (4 episodes each), 300 and 400 (2 episodes each). save_2 and save_5: side 550, 2 episodes. GPU shared with training. Offset analysis: scratchpad script `aim_offsets.py`, 8×16 grid (cells 0-127, offsets rebuilt in [−1, 1]). Decisions kept: step ≥ 10.

## Question

In the CSV `alpha_aim_dy` rises without interruption (1e-3 → 4.4e-3 between 0 and 233 k) while `alpha_aim_dx` and `alpha_aim_cell` fall to floor. Is it a σ_dy collapse or a tanh saturation?
The dual is $\mathcal{L}_\alpha=\log\alpha_k\,\big(\bar H_k-H^\star_k\big)$, with $\bar H_k=\mathbb{E}_{s}\big[\sum_c\pi(c\mid s)\,\hat H(\delta_k\mid c,s)\big]$ (one-draw estimator, tanh Jacobian included) and $H^\star_k=-1$. α rises only if $\bar H_k<H^\star_k$ on the training data.

## Measures

| | side | $\bar H_x$ | $\bar H_y$ | \|off_y\| > 0.95 | median canon elev. | tour<5° | hit/shell |
|---|---|---|---|---|---|---|---|
| s2 (54 k) | 550 | −0.17 | −1.09 | 0.00 | −23.6° | 0.104 | 0.096 |
| s5 (135 k) | 550 | −0.18 | −1.23 | 0.00 | −12.2° | 0.203 | 0.156 |
| s8 (216 k) | 300 | −0.74 | **−1.11** | 0.00 | −2.6° | **0.310** | **0.202** |
| s8 | 400 | −0.64 | −0.94 | 0.00 | −2.2° | 0.298 | **0.202** |
| s8 | 550 | −0.51 | −0.70 | 0.00 | −2.1° | 0.282 | 0.171 |
| s8 | 1000 | −0.24 | −0.25 | 0.00 | −2.4° | 0.213 | 0.087 |

- No saturation: $\tanh\delta_y$ quantiles 10/50/90 −0.19/−0.04/+0.18 at side 300 and −0.34/−0.03/+0.37 at side 1000. The share beyond 0.95 is null everywhere.
- $\bar H_y$ grows monotonically with the side (−1.11 → −0.25). During training the sides run from 255 to 550 (mean 445 beyond 200 k). Interpolated over that range, $\bar H_y\approx-0.95$, i.e. **at target** and slightly under on small arenas.
- $\bar H_x$ stays between −0.24 and −0.74, always above −1, consistent with α_dx at floor. Cell entropy is 3.6-4.1 nats for a 0.69 target, hence α_cell at floor.

## Reading

- Neither collapse nor saturation. The vertical entropy constraint is active and holds $\bar H_y\approx H^\star$. The policy wants to tighten the vertical offset and α_dy rises to prevent it: the expected regime of an active dual. With Adam a weak but persistent bias of $\bar H_y-H^\star$ makes $\log\alpha$ drift at nearly constant speed, which explains the CSV's regular slope.
- Order of magnitude: $H^\star=-1$ corresponds, for μ ≈ 0, to $\sigma\approx0.09$ pre-tanh, i.e. a std of about 0.011 in vertical NDC (a cell is 0.25). The constraint already allows very fine aim.
- Aim: tour<5° rises from 0.10 to 0.20 then 0.28 (side 550) between s2, s5 and s8, vs 0.065 for 552 s8. The canon rises from −23.6° to −2.1°, whereas 552 stayed between −9 and −18° to s10. 552's low-canon defect has gone by s8.
- Trigger: ratio goes 1.09 → 1.13 → 1.43 (aligned close n 148 / 270 / 464, under the 300 threshold for s2 and s5).
Conclusions: established: α_dy's rise is not a pathology. The dual fills its role, $\bar H_y$ stays at target and σ_dy is not at floor (offset varies without saturating). Signal to watch: the term $\alpha_y\bar H_y$ stays ≈ 4e-3, negligible vs the surrogate; it would become significant if α_dy exceeded ≈ 5e-2 (+2.4 in log; at the current pace of +1.5 per 230 k that would take ≈ 370 k steps, around 600 k); reread from save 15. Established: aim progresses clearly faster than in 552 at equal save, and the canon is no longer low.

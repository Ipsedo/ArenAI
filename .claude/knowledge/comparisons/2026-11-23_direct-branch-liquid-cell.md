# Direct branch encoder → heads in the liquid cell (run 551 test of cell delay)

Runs: 550 (twin), 551

2026-11-23: direct branch encoder → heads (Linear 320→128 + LayerNorm + SiLU, concatenated at the cell output) coded in `LiquidActor` and `LiquidCritic`, uncommitted; tests the ~1.5-decision delay of the cell, without Φ.

After the analysis of run 550, Samuel chose « terme direct seul, sans Φ », then validated piece by piece: actor **and** critic, same form as the skip of 416-418 (commit `1604dc5c`), standard init (`init_hidden_weights` on the branch, heads unchanged).

Code (uncommitted at the time): module `direct` in `arenai_agent/src/networks/recurrent/liquid_actor.{h,cpp}` and `liquid_critic.{h,cpp}`; heads μ, σ, fire and `to_value` as `Linear(2 × neuron_number → ·)`; `features` of `LiquidActorOutput` = concatenation (256). Compiled in `build/` (2442 tests OK, 109 pre-existing ignored), real 4-min trial without error. Checkpoints ≤ 550 incompatible: 550 probe binaries kept in `outputs/train_550_ppo_liquid/probe_bin` (probe, sigma_gradients, libarenai_agent.so). `cmake-build-release/` not rebuilt (550 was running there).

**Why:** heads wired to the cell state only ⟹ μ canon_x reaches 48/76/89 % of a change in 1/2/3 decisions; every delay removed from the loop improved aim (stabiliser 2026-11-12, up-to-date image 2026-11-19). 417/418 had this skip but confounded with bigger CNN and LTC, before these fixes.

**Test plan:** twin = 550 (same config, no Φ). Criteria at saves 3-11 then at equal D: μ response to a crossing target (`probes/reponse.py`), directed command decomposed by side, « < 5° in 1 s », dwell < 5°. Note: without Φ the gesture gradient is zero (ρ ≈ 0), this run tests the delay alone; a null result does not condemn the branch with Φ.

**Outcome (same day):** branch **removed 2026-11-23** at Samuel's request: delay removed but behaviour ≡ 550 (see comparisons/2026-11-23_turret-bimodal-per-pass-engagement.md; run 551 file). Probe 551 with `outputs/train_551_ppo_liquid/probe_bin`.

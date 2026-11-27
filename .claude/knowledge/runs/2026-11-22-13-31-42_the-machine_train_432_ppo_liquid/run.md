# train_432_ppo_liquid

| | |
|---|---|
| machine | the-machine |
| user | samuel |
| launch | 2026-11-22 13:31:42 |
| commit at launch | `dcca274e` — bootstrap the value on the last state at the episode time limit (GAE and return scaler no longer run into the next episode), continuous target entropy -1 on the four actions, shells_per_hit default read from GAME_SHELL_ECONOMY, critic value and per-step rewards in probe exports |
| commits ≤ 90 min after | `7bccaa04` |
| diff at launch | unknown (identity recorded after the run ended) |
| raw data | `outputs/train_432_ppo_liquid` on the-machine |

## Established facts

_run 432 liquid, single optimisation group for the policy — bearing aim learned before 27k then flat gain; method and sign convention to measure the canon servo_

Run 432 (ppo_liquid, launched 2026-11-22, uncommitted on top of dcca274e): a single optimisation group
for the policy (no more aim advantage nor per-group KL), confounded with continuous entropy target −1
and time-limit bootstrap.

Probes side 350, save_0 / save_1 (27k) / save_3 (81k): turret<5° 0.053 / 0.154 / 0.188, hit/shell
0.029 / 0.122 / 0.135. corr(μ_canon_x, bearing error) at 10-20°: +0.08 / −0.30 / −0.26, i.e.
~5 % of the error closed per decision, identical at 27k and 81k. Bearing aim is thus
learned early then its gain stagnates; the turret<5° progress comes from canon σ (0.19 → 0.09). Weakly
selective trigger (ratio 1.27), 83 % empty the reserve in 3.4 s.

save_6 (162k, D=450): aim gain still flat (3rd probe), turret<5° 0.175; it is the trigger
that learns (ratio 2.43, p_off 0.59 → 0.32, reserve emptied at 6.9 s, hit/shell 0.162); canon σ
spread 0.02-0.48 depending on state.

save_13 (351k, D=800): trigger closed (CSV fire 1.40 → 0.35 from 260k, kill 25.8 → 15.6); p_off
0.02-0.03 whatever the reserve, p_fire <5° 0.24-0.33 while an aligned shell hits 1/2;
at equal side 350 hits/s ÷2, turret<5° 0.175 → 0.077, tanks 2× farther, canon σ 0.48.
Empty-reserve penalty = −D/1000 terminal (ee1ca624): makes off-axis deficit-making from D=250
(0.025/shell vs ~0.01 expectation), but does not explain the closure in aligned state, and 428 (fixed
penalty) had closed the same way.

**How to apply:** to measure aim, regress μ_canon_x on `yaw` (signed error towards the
nearest target) per error slice in the probe CSV, and always probe save_0 as the
reference. Measured convention: canon_x action +1 ⟹ error +12.7°/decision, the right direction is a
**negative** correlation (run 423's « corr ≈ 0 » were global, whereas the
signal exists only under ~45°). The `pitch` column does not react to canon_y: pitch is not
measurable this way. Script: disposable scratchpad, to rewrite (20 lines pandas, `.venv/bin/python`).

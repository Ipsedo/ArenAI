# Hypothesis: canon σ / target holding — holding pays but σ 0.5 + sweep make it unreachable (refuted by run 529)

Runs: 520, 522, 524, 525, 528, 529

2026-11-17: target holding pays (1 s = 3 hits) but canon σ 0.5 + sweep make it unreachable; 520/522 hold because the turret is still at σ 0.2; untested hypothesis, options posed. **Refuted 2026-11-18 by run 529** (see end).

Search for the lever requested by Samuel 2026-11-17 after run 528 (« les 5 obus n'ont pas résolu, cherche le vrai levier avec ce CNN et le reward événementiel »). Report: `outputs/train_528_ppo_liquid/levier_visee.md`. No code modified at first.

Measured facts (probes side 550):
- hits per pass < 5° & < 100 m by duration (528 s9): 0.07 (0.1 s), 0.34 (0.3-0.4 s), 1.04 (0.5-0.9 s), 3.05 (1-1.9 s), 7.5 (≥ 2 s); outside a pass 0.076 hit/s. Cadence in pass ~5 shots/s. The event reward thus pays holding; 0.6-3 % of passes last ≥ 1 s.
- plant: 9.2° + 9.0° per unit of a·|a| over two decisions. Exploration noise: 3.0°/decision at canon σ 0.5 (524/525/528), 1.6-1.7° at σ 0.2 (520/522). Mean command: sweep 3-4°/decision (528, 525) vs 1.0-1.1 (520/522).
- dwell < 5°: 0.37 s untrained network (525 s1) → 0.20 s (528 s14); 520/522: 1.0-1.1 s ≈ simulation of a still turret at σ 0.2 (1.43 s); at σ 0.5 still = 0.38 s, ideal controller 0.96 s.
- canon_x entropy at probe ≈ 0 = target `continuous_target_entropy` for σ 0.53-0.60; alpha_canon_x leaves the floor around 200-300 k (1.3-3e-3 in 524/525/528): the regulator holds σ. Initial rise 0.30 → 0.58 before 135 k = policy gradient (α at floor). 520/522: α_canon_x 0.010-0.017 from 81 k with σ 0.2.

Hypothesis (Claude's, correlational, untested): at σ 0.5 stopping the sweep gains only +0.15 s on target vs +1.2 s at σ 0.2 ⟹ holding gradient ÷10; separate credit perhaps acted mostly by tightening σ. Counter-indications: ideal controller holds 0.96 s at σ 0.5; 498/433/439 at low σ did not aim (but spatial softmax).

Options posed to Samuel, nothing decided: (1) σ ceiling ≈ 0.25 for both canon actions + lowered canon entropy target; (2) canon target −1 only; (3) temporally correlated exploration noise. Refutation criterion of (1) at save_5: dwell > 0.6 s and tour<5° > 20 %, otherwise the lock is the sweep.

**Rule:** in every probe, read dwell < 5°, canon σ and mean command in °/decision (scripts to rewrite: passes by duration, regression Δyaw on a·|a|); do not present the hypothesis as established.

**Coded 2026-11-17 (option 1 validated by Samuel, uncommitted, build `build/` + 2 442 agent tests ok):** σ ceiling 0.25 for canon_x/canon_y (`CANON_LOG_SIGMA_MAX`, `FIRST_CANON_ACTION`, `NB_CANON_ACTIONS` in networks/constants.h; `SoftClamp` with one ceiling per action; `log_sigma_upper_bounds` in networks_utils/init) in Actor, SacActor and LiquidActor (Samuel's choice: all actors); initial canon σ = min(initial_sigma, 0.20) (Samuel's choice: 0.20); default `continuous_target_entropy` = [0, 0, −1, −1] in ppo, ppg, ppo_liquid. Old checkpoints load but their canon σ is reinterpreted by the new binaries: stochastic probe wrong on ≤ 528. Previous probe binaries saved in `outputs/train_528_ppo_liquid/probe_bin/` (launch with `LD_LIBRARY_PATH` on this folder). `cmake-build-release` not rebuilt (528 running there).

**Refuted 2026-11-18 by run 529 (σ cap):** canon σ held at 0.24 over 729 k, the sweep disappears at the start (s5: 0.9°/decision, dwell 0.47 s) then returns through the command (s27: 3.3°/decision, dwell 0.17 s, below 528) and the trigger closes the same. The lock is the sweep itself, not exploration noise. (see docs/decisions.md 2026-11-21: canon σ cap 0.25 removed)

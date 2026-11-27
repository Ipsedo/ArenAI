# train_415_sac

| | |
|---|---|
| machine | the-machine |
| user | samuel |
| launch | 2026-11-10 08:30:12 |
| commit at launch | `a15ee8f1` — sac : follow develop (log-sigma constants, alphas written to the CSV only) |
| commits ≤ 90 min after | `15b32ece` |
| diff at launch | unknown (identity recorded after the run ended) |
| raw data | `outputs/train_415_sac` on the-machine |

## Established facts

_SAC run 415 (2026-11-10, α₀ 5e-4) — dead phase removed, hit 0.12 at D=350 then 0.11 at D=400 (91k steps); probe: near-unconditional fire, aim uncorrelated with error, hit = fortuitous alignment (<2° → 0.30, >30° → 0.01); Q drifts negative (−0.34) while realised return is positive_

Run 415 (config 414 except α₀ 5e-4, lr α 3e-5, 32 tanks, 128×64, D 250→400) analysed at 91k steps / 6 h
(still running). Saves every 27k: save_1 = 27k, save_2 = 54k, save_3 = 81k.

CSV: learning starts from ~3k (hit 0.03 at 2.7k, 0.08 at 19k) — **414's dead phase
was indeed α's decay**. Hc 2.0 → −3.5 at 43k then plateau −3.5/−4.2 (α_c 4.8e-4 → 5e-5 at 44k,
then back up to 1.3e-4: regularisation acts). Curriculum 300 (21k), 350 (48k), 400 (67k). Hit peak
0.121 / kill 14.7 at 58–66k (D=350); at D=400: hit 0.109–0.112, kill 12.3, miss 31 → 28 m. Not
414's regression (0.117 → 0.067). α_d0 (fire) rises without stopping 5e-4 → 3.7e-3: fire entropy
stays below its target. **Q (q1 at replay actions) decreases from −0.06 to −0.34 while r/decision
rises to 0.0038 (V ≈ r/(1−γ) ≈ +0.4)**: underestimation ≈ 0.7, mirror of 414's +0.4
overestimation; unexplained (entropy bonus ≈ 0.1 at most). Critic loss q ×10 (0.0017 → 0.019).

Probe (D=350, 2 episodes, sampled) on save_1/2/3:
- near-unconditional fire: p(fire | can_fire) 0.79 / 0.43 / 0.81, barely depends on angle
  (save_3: 0.91 at <2°, 0.72 at >30°). Empty reserve 83 / 53 / 71 % of decisions, can_fire 0.10–0.34.
  save_2 (best, shell hit 0.122) is the only selective one (p 0.29 beyond 30°).
- hit per shell by alignment at fire (save_3): <2° 0.31 (miss 10 m), 2–5° 0.25, 5–10° 0.15,
  10–30° 0.08, >30° 0.01 (miss 45 m); 59 % of shells leave at >10°. Distance: <100 m 0.19, >100 m
  0.04–0.06; 45 % of shots at <100 m.
- **corr(mu_canon, yaw/pitch error) ≈ −0.05**: the aim command does not react to the target offset;
  median |yaw| to the nearest tank 80°. Hit = fortuitous alignment on the best of 31 tanks.
- pre-tanh σ 0.42–0.50 with |μ| 0.77–0.95 (saturation |act|>0.9: 17 % yaw, 25 % pitch): the
  σ 0.1 entropy target is reached by tanh saturation, not by a small σ.

**Why:** the 414 → 415 gain comes from the initial α; the mechanism remains "fire what passes in front",
like the PPOs (see ppo-actor-audit-2026-10-29). The `arenai_agent/probe` now also loads
the SAC actor (`is_sac` branch = absence of `neuron_number` in config.json; change uncommitted
as of 2026-11-10).

**How to apply:** options to validate with Samuel, not decided: check Q's sign (MC return vs
Q in the probe), understand why p(fire) rises back to 0.8 despite growing α_d0, test save_2 in
game. See run 414, sac-agent-2026-11.

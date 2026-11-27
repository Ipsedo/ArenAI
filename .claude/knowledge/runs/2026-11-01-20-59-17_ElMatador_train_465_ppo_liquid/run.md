# train_465_ppo_liquid

| | |
|---|---|
| machine | ElMatador |
| user | samuel |
| launch | 2026-11-01 20:59:17 |
| commit at launch | `7ea94d83` — * fire cost * initial fire/zoom proba at 0.1 |
| commits ≤ 90 min after | none |
| diff at launch | unknown (identity recorded after the run ended) |
| raw data | `outputs/train_465_ppo_liquid` on ElMatador |

## Established facts


- 2026-11-02: config = 464 (liquid, 32 tanks, epochs 4, target_kl 0.05, entropy 0.001/0.001, reserve 10 → +10/hit, max 30) + fire cost 0.03/shell (commit `7ea94d83`) + p_init fire/zoom 0.1. Analysis at 760 k. (see docs/decisions.md: fire cost / miss cost removed 2026-11-07, refused 2026-11-09)
- 2026-11-02: until D=550 (~300 k) 465 ≥ 464: fire 1.5/s, kill 3.9/episode at D=450, hit 11 %. p_init 0.1 had no lasting effect (fire joins 464 by 90 k). From D=600 (~314 k): fire 1.5 → 0.7 → 0.2 → 0.03/s, kill → 0, s (episode duration) 12 s → 30 s = everyone dies of starvation. 464 at equal D keeps fire 1.1-1.3 and kill ~1.9.
- 2026-11-02: yet gain per shot (aim + hit)/shot ≈ 0.08-0.09 constant = 3× the cost: expectation does not kill firing, the signal does. Cost is immediate, deterministic and glued to the step where fire=1; gain is delayed (flight), stochastic, read through a critic with ev < 0 from 270 k. Only the −0.03 survives advantage noise ⟹ fire head falls to p ≈ 0.
- 2026-11-02: simultaneous switch: Hc 0.2 → 0 at D 550-600 (σ ≈ 0.24) then −0.25; skip 0 → 33 % at 950 (kl 0.03 < 0.05 but KL regime like 441); Hd ≈ 0.30 = zoom frozen at p_init 0.1 (like 439).
- 2026-11-02: curriculum raised D to 950 (464: 850 at same step) because its criterion is hit/fire over 24 probe episodes: an agent firing only its best shots (hit 12 %) passes it at 0.2 shot/s. D is no longer a competence measure when fire collapses.
- 2026-11-02: refined mechanism: initial spam is rational (D ≤ 500: 32 tanks over 250-500 m, each shot has a sampled enemy, net EV +0.055/shot; empty reserve = −0.1 once ≈ 2 shots; +10 shells/hit at 11 % ≈ self-financing). At D=600 (316-327 k) fire 1.0 → 0.5 in 10 k while aim/shot and hit/shot RISE (0.045 → 0.073): the agent cuts shots without enemy in range (density ÷6), worth −0.03 flat. Then loop: superlinear kills (5 hits on same part) 2-3 → 0.2-0.5/episode between D 600 and 650, hit/shot 0.04 → 0.023, aim/shot 0.025 < cost 0.03 ⟹ a no-hit shot is surely net negative, the hit (+0.2, 1 in 10, delayed) is invisible. Not firing is free: starvation = truncation without penalty. Hc → 0, ev < 0 and skip 33 % come after (~470 k) and lock, they do not trigger.
- 2026-11-02: second failure of the fire cost (see agent-does-not-aim note); at p(fire) ≈ 0.003 and bonus 0.001 return is impossible and the deterministic game will never fire (dead-trigger / deterministic-threshold failure). Do not wait for a rebound; a fire cost holds only if the critic follows (ev > 0) or the gain is less delayed. Read fire AND D together: D rising with fire < 0.3/s is a hit/fire-criterion artefact.

# Lock: trigger closed at range (single advantage = hits at < 50 m only)

Runs: 522 (reference), 524, 525, 526

2026-11-17: single advantage (524-526) = hits at < 50 m only (60-75 %), trigger closed beyond 50 m before long-range aim is learned, hit/shot curriculum satisfied by this closure ⟹ kills ÷2 when D rises.

Measurement 2026-11-17 on existing probes, by target distance at the shot (`outputs/train_526_ppo_liquid/diagnostic_portee.md`), 524/525/526 vs 522 (separate aim credit):

- hits/tank-s side 550: single advantage 0.05-0.06 at < 50 m and 0.002-0.007 at 50-100 m (falling from save to save); 522: the 50-100 m band rises from 0.023 (s5) to 0.055 (s24) and equals the close one;
- hit of an aligned shot < 5°: > 70 % at < 50 m everywhere (41 % even without aiming); at 50-100 m 25-35 % without progress (single advantage) vs 38 → 62 % (522);
- p_fire on aligned target: 525 s9 0.38 at < 50 m, 0.14 at 50-100 m, 0.04-0.07 beyond; 522 s9 0.11 / 0.14 / 0.06-0.09 with 6-10× more aligned states at 50-100 m;
- enemy at < 50 m: 40 % of the time side 550, 9-18 % side 1000.

Reading (Claude's, by run comparison, not intervention): the trigger, the clearest signal, closes beyond 50 m between 135 k and 243 k; fewer shells at range ⟹ fewer events to learn long-range aim; the curriculum criterion (hits/shots > 0.08, `curriculum_ratio_high`) is satisfied by this closure, D rises, close encounters become rare, kill 18 → 8. The batch 2400 of run 526 changes nothing in this mechanism; the « per-step noise » hypothesis of the run 525 audit (event-only, Φ removed) is thus insufficient.

**Rule:** read any probe by distance band (hits/tank-s and aligned hit at 50-100 m), not only tour<5° nor the ratio aligned & < 40 m, which rewards exactly the « point blank » policy. Options posed to Samuel 2026-11-17, not decided: curriculum criterion on hit throughput or kills; keep the trigger open at range while aim is learned; separate aim credit.

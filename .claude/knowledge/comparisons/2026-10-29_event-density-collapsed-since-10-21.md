# Event density collapsed since 2026-10-21 — why convergence is slower than in September

Runs: 417, 419, 428, 429, 430, 437, 441, 450

kills/episode 11 (417-419) → 1.4 (428) → 0.3-0.9 (430) → 0.1-0.3 (437+); random hit 6 % → 2.6 % → 1.8 %; the event reward no longer has events.

Figures (metrics.csv, blocks 0-100 k / 100-300 k):
- 417/419 (2026-09-28, 10-04): 32 tanks, network 512, shell regen, 30 Hz — hit 6 % → 10 % in 100 k, kill 5 → 11 per episode, D leaves 250 at 300 k, D 418 at 1.2 M.
- 428/429 (2026-10-21): 16 tanks, network 128, spatial softmax, can_fire mask, 10 Hz, new aim reward — hit 2.6 % → 5 %, kill 0.3 → 1.4, D leaves 250 at 300 k-1.2 M.
- 430 (2026-10-23, regen removed): kill 0.1 → 0.9, fire 1.1 → 0.55/s.
- 437/441 (2026-10-25, death at empty reserve −1): r = −0.003/step (92 % of lives end with the −1 at ~24 s), kill 0.1 → 0.5, hit 2.8 % → 5.8 % at 1.2 M only, D 250 until 800 k.
- 450 (2026-10-29, IMPALA + bonus 0.01): random hit 1.8 %, kill 0.04.

**Why:** the reward is purely event-based; the aim gradient is carried by hits (0.4 per life today, ~4 in 419). Death at empty reserve is a −1 identical for all, without gradient (cf. comparisons/2026-10-27_ammo-economy-break-even-threshold.md, run 437). Halving the tanks halved the random hit.

**Rule:** before touching the optimiser, compare event density (hit at 0 k, kill/episode) with 419's; levers = tank density / arena size, empty-reserve rule, hit vs death scale. Decisions for Samuel.

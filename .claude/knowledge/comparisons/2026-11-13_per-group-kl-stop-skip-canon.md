# Per-group `target_kl` stop (canon vs rest) — GroupKlStop

Runs: 509 (cause), 510 (first run with it)

2026-11-13 — `target_kl` stop per group (canon vs rest) in ppo/ppg/ppo_liquid via `GroupKlStop` (ppo_objective); skip = rest; `skip_canon` CSV-only; same threshold 1.5 × target_kl; uncommitted at the start.

Samuel's choice: option 1 (stop per group), the three trainers, same threshold per group, `skip_canon` CSV-only. A stopped group loses its surrogate and its entropy bonus for the rest of the update; the step where it stops is recomputed without it; `kl` stays the joint KL. Companion tool: `arenai_agent_sigma_gradients --kl_by_group`.

**Why:** run 509 (comparisons/2026-11-12_factored-aim-credit-phi-aimed.md) — the canon made 77 % of the joint KL at the stop step and cut the trigger's update.

**Test plan:** first run = 509 + this single change (p_init fire left at 0.5); read skip vs skip_canon, dump exit (reserve, probe ratio) against 508/509 at equal D. Rebuild cmake-build-release before launching.

(see docs/decisions.md 2026-11-22: separate aim credit incl. per-group KL stop removed, single optimisation group)

# Agent economy: 2 shells per hit (`GAME_SHELL_ECONOMY` 10/2/30)

Runs: 520 (config base), 521 (first run with N=2)

2026-11-15 — Samuel's choice: lever « constraining reserve », `GAME_SHELL_ECONOMY` 10/2/30 (2 shells returned per hit instead of 3) for the agent in game and in training; the player has no shell economy; uncommitted.

Reason: reward = ⅔ r_hit ∝ hits per tank-step, and at 3 shells per hit a reserve of 25-30 % hit/shell suffices for 10-30 hits, death comes before the empty reserve ⟹ volume wins (520's drift after save_32). At 2 per hit the budget falls to 5-7.5 hits at 25-30 %, and shot sorting (< 5°: 35 %, < 1°/1°: 55-63 %) becomes the optimum (hits-per-life model, validated on 520's drift).

A single constant `GAME_SHELL_ECONOMY` (shell_economy.h): agent tanks in game (game_session.cpp), CLI defaults of training and probe. The player (`make_player_tank`) has no economy. Empty-reserve rule always different: truncated respawn in training, disarmed until death in game.

**Why:** Samuel prefers lever 1 (same rule game/training) over precision in the reward; choice N = 2, player untouched.

**Test plan:** next run = config 520 + N = 2 only. Markers: hit/shell > 30 % held at D 1000, stable cadence from save_24 to save_37, late 1st empty reserve; failures = extinguished trigger or dump; watch very early empties (respawn as leak). Related: run 520 (discrete target 0.01), comparisons/2026-10-27_ammo-economy-break-even-threshold.md. (not in docs/decisions.md as an entry; see 2026-08-06 shell reserve validated)

# Game rule at empty reserve: no respawn / no truncation in training (run 511)

Runs: 510 (base), 511 (first run), 491, 437, 479 (earlier rules)

2026-11-13 — Samuel's decision: at empty reserve, no more respawn nor truncated in training; the tank stays alive disarmed until death or end of episode, as in game (uncommitted, run 511 from scratch planned).

Respawn/truncated block removed from `TrainTankEnvironment::step` (train_environment.cpp). Model/core API (`respawn`, `respawn_tank`, `is_out_of_shells`, `has_shells_in_flight`) left in place, unused outside tests. Trainers unchanged: truncated is never true any more. No empty penalty (difference with 437/479).

**Why:** in game tanks end dry: at 21 % hits, a shot costs 0.58 shell net, and training offered a free exit (full respawn, V cut to 0) never seen in game. Samuel: « je veux que la politique apprenne à économiser ses obus ». Follows run 510 (per-group KL stop save3), cancels the run 491 empty-reserve truncation/respawn (option A).

**Test plan:** run 511 = 510 + this single change, from scratch after reboot (NVIDIA driver). Read at probe: % disarmed, armed decisions (floor 490: 2-3 %), 1st empty reserve, ratio, hit/shell vs 510 at equal D. Fallback validated by Samuel if the trigger starves: delayed respawn of N s (see comparisons/2026-11-13_delayed-respawn-cut-at-respawn.md). Probe follows the new rule: % disarmed not comparable to 510 probes. Open point: episodes where all the living are dry run to max_episode_steps without events (throughput). (see docs/decisions.md 2026-11-22: no respawn (11-13, reverted); current rule refill + constant penalty)

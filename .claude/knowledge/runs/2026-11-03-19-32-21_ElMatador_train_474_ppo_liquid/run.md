# train_474_ppo_liquid

| | |
|---|---|
| machine | ElMatador |
| user | samuel |
| launch | 2026-11-03 19:32:21 |
| commit at launch | `088ff389` — remove potential reward |
| commits ≤ 90 min after | `12e06985` |
| diff at launch | unknown (identity recorded after the run ended) |
| raw data | `outputs/train_474_ppo_liquid` on ElMatador |

## Established facts


- 2026-11-03: config = reserve 30, +20/hit, max 60, no empty-reserve death. Analysed 2026-11-03 at ~205 k (save_1, probe `arenai_agent_probe --spawn_side 300`, 6 episodes). Reference run of the new economy, to compare to 417-419.
- 2026-11-03: events returned: kill 7-9/episode (464-473: 0.2-4), hit 8 %, r_hit 0.006. Kill per episode-second ≈ 464/465: the rise comes mostly from 3× longer episodes (s 40 s vs 13-16 s).
- 2026-11-03: dump confirmed: p_fire 0.75-0.83 all episode, firing at cooldown ceiling (3.5-4/s); empty reserve median at 11 s, 85 % of tanks end at 0; unarmed life ~23 s; median death 29.9 s = starvation; **57 % of tank-steps unarmed**.
- 2026-11-03: trigger: slight selectivity (p_fire 0.84 at < 2° → 0.65 at > 90°, better than flat 470) but all > 0.5. Aligned shots (< 5°, < 40 m) 1.1 %, hit at 39 %; median angle at fire 17°, distance ~100 m.
- 2026-11-03: dump = optimum: break-even +20/hit = 5 %; hit 8-9 % in the first 10 s (density) then 3 % after 20 s ⟹ firing early is the best use of a shell; empty reserve free (starvation = truncation). Side effects: 70-110 s episodes ⟹ curriculum (24 probe episodes) ~100 k steps per update, D 250 for 150 k then 300; Hc 0.17 → 0.036 at 200 k (faster than 470); negative ev on the last lines.
- 2026-11-03: proposed to Samuel, not applied: run on to ~500 k and reprobe, or end the tank on empty reserve as truncation (removes zombies; 451 inconclusive). Do not read the CSV `fire` column as a trend (window 256 steps < 1 episode: oscillates 0 ↔ 3 by phase).

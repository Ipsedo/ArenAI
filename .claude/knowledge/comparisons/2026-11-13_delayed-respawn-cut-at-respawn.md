# Delayed respawn, trajectory cut at respawn (validated, pending run 511 save_9 verdict)

Runs: 510, 511 (save_5/9), 491, 481

2026-11-13 — scheme validated by Samuel, awaiting the verdict of 511 save_9: at empty reserve, 10 s disarmed and vulnerable without cut, then respawn with 10 shells as truncated + V cut; D (episode end at 0 armed) judged insufficient.

Validated scheme (not coded): empty reserve and no shell in flight ⟹ the tank stays in place 10 s (fixed duration), disarmed and vulnerable, trajectory not cut (`r_rcv`/death in the return); then respawn with 10 shells, `truncated` + V cut at respawn; death before ⟹ normal `done`. Open point: respawn in place or at a new spawn.

**Why:** 511 save_5: dump faster than 510, flat trigger (see run 511 save_1 note). Retained hypothesis: everyone dry in ~3 s ⟹ disarmed without danger. Cutting at respawn (not at empty as in 491-510) puts the danger of waiting into the return without ever paying the reload, coherent with the refusal of regen (run 481 slow regen). Variant D discarded: neutral on returns, throughput only; at threshold « ≤ 1 armed » it would punish the thrifty.

**Test plan:** code only after 511 save_9, if its trigger stays flat. Replaces the « delayed respawn » fallback of comparisons/2026-11-13_game-rule-empty-reserve-no-respawn.md. Risk: trigger extinction of the 437/479 type, to watch at probe (armed p_fire ≪ 0.1) in addition to the dump. (see docs/decisions.md 2026-11-22: empty reserve rule = refill + constant penalty)

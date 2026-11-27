# train_433_ppo_liquid

| | |
|---|---|
| machine | the-machine |
| user | samuel |
| launch | 2026-11-23 22:49:10 |
| commit at launch | `f7435cc2` — update harness |
| commits ≤ 90 min after | none |
| diff at launch | unknown (identity recorded after the run ended) |
| raw data | `outputs/train_433_ppo_liquid` on the-machine |

## Established facts

_run 433 (4-block CNN + spatial softmax + presence head) — probe save_14 at D 850, comparison with the 552 report from the other PC_

Run 433 `ppo_liquid` (launched 2026-11-23, code = f7435cc2 + staged uncommitted diff): 4-block CNN + spatial softmax (x, y, amplitude) + presence on the read map. Reference = run 552 of the other PC (7 blocks flattened), known only through `2026-11-24_rapport_visee_552` (no CSV/checkpoint here).

Probe save_14 (378 k, D 850), 2026-11-24:
- position at the policy input 0.87° / 1.51° / 4.18° (60-100 / 100-200 / 200-400 m), ≥ frozen 552 (1.26 / 2.19 / 5.34°) despite a 2× smaller readout set; fine-tuned 0.78 / 0.94 / 1.75°; softmax temperature stayed ≈ 1;
- but aim unchanged: turret<5° 0.185 (550) / 0.133 (1000) ≡ 552; corr(μ canon_x, yaw) ≈ 0 at 200-400 m, −0.1 to −0.26 at 100-200 m; gesture credit ρ ≈ 0; hit/shell 0.15-0.16;
- selective trigger (ratio 23 side 550) but does not follow an oracle aim (0.16-0.36 shells/tank-s vs 1.2-1.3 for 552 s41).

**Why:** representation was not the only lock; available precision ≠ used precision, consistent with run 432 and cnn-perception-validated.
**How to apply:** compare to 552 only at D 1000; position-bench figures are comparable only at equal set size (« 7 fresh blocks » reference to rerun). `probe_checkpoint.py --check-only` does not see a binary older than a staged diff: check the binary time by hand.

Follow-up 2026-11-24: D 850 → 800 at 429 k, kill 21 → 12-15, fire 1.0 → 0.2-0.6, hit rises to 0.19. Probe save_17 (459 k) vs save_14: aim unchanged; trigger closed everywhere except < 40 m (aligned p_fire at all distances 0.30 → 0.19 side 550, 0.091 → 0.041 side 1000; 0.74 unchanged at < 40 m); under oracle aim it fires even less (0.25 / 0.09 shells/tank-s). Same pattern as run 428 (hit ↑ by extinction, kill ↓). Cause not established.

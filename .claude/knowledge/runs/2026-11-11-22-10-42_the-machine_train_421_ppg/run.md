# train_421_ppg

| | |
|---|---|
| machine | the-machine |
| user | samuel |
| launch | 2026-11-11 22:10:42 |
| commit at launch | `c02137c6` — absolute angle for turret |
| commits ≤ 90 min after | none |
| diff at launch | unknown (identity recorded after the run ended) |
| raw data | `outputs/train_421_ppg` on the-machine |

## Established facts

_Run 421 PPG (c02137c6, aim as absolute command yaw cos/sin + pitch, MIN_ALPHA 1e-4, vision 256×128) — D=400 at 145k then stuck; hit/fire flat 0.086 → 0.077, kill ×2 by cadence alone (fire 1.1 → 2.1/s), Hc → −1.0 without gain_

Run 421 PPG launched 2026-11-11 22:10 on c02137c6 (high confidence), analysed at 515k / 9.2 h (running).
Confounded changes: absolute aim command (5 continuous actions, yaw atan2(sin, cos) in the
chassis frame, step limited), entropy targets [0, 0, −1, −1, −1], MIN_ALPHA 1e-3 → 1e-4, big
network/vision 256×128. First PPG since the 06/11 reward redesign → references = SAC 415-420.

- D: 300 @ 21k, 350 @ 114k, 400 @ 145k then stuck. Hit peak 0.103 @ 133k (D=350).
- At D=400: hit 0.086 → 0.077, miss 31 → 36 m, but kill 7 → 19 and r ↑ by **volume** (fire 1.1 → 2.1/s).
- Hc 0.23 → −1.0; α all at the 1e-4 floor until ~200k (dir) / ~440k (aim) → free contraction, no gain.
- skip ~0.4 and clip ~0.3 since 200k; ev 0.9 → 0.4.
- Camera mounted on the barrel: with absolute command the actor must add the turret angle (proprio).

**Why:** 1st test of the absolute command since the 24/09 revert (aim-absolute-position-control).
**How to apply:** do not read kill/r as progress; options proposed: probe save_5/save_19,
stop, or replay 421 in delta to isolate the command. See run 415.

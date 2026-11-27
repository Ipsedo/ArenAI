---
name: ppo-train-389-diagnosis
description: "Run 389 (388 + recharge +10 shells/hit, commit dba4e002) — read at 202k steps: kill 1.5-2.1 vs 0.8-1.7 at equal D, shots per life ~14 vs ~10, hit held but miss unchanged; volume gain, not aim; 388's dips at 237k not yet reached"
metadata:
  node_type: memory
  type: project
  originSessionId: f4f60a9b-baf5-487c-aea1-2d1d583ff538
  modified: 2026-11-01T12:59:10.303Z
---

Run `outputs/train_389_ppo` (launched 2026-11-01 10h15, read at 202k steps) = [ppo-train-388-diagnosis](ppo-train-388-diagnosis.md)
with `shells_recharged_per_hit` 5 → 10. Samuel decided to let it continue.

- Per D (389 / 388): kill 1.88/1.29 at D=300, 2.06/1.63 at 350, 1.78/1.68 at 400, 1.79/0.74 at 450,
  1.52/0.78 at 500. 388 is averaged over its first 300k steps, 237k dip included at 450-500.
- Shots per life ~14 vs ~10, s 12-14 vs 10-12, fire 1.1-1.2 vs 0.9-1.0.
- hit 0.09 vs 0.10 at D=300-400, no fall back towards the random floor. miss not improved.
- Hc identical to 388 (σ decreases the same), skip 0.045 at 194k vs 0.058, clip 0.34, ev ~0.6.

**Why:** recharge raises kills through fire volume; aim has not progressed.

**How to apply:** the decisive test is the 237k-490k stretch (388's `fire`/`hit` dips) and a
possible hit > 0.10 at D ≥ 500, the reserve autonomy threshold.

## From ElMatador memory

Distinct run from the 2026-11-01 one above (this 389 dates from early September 2026). Note dated 2026-09-05: aborted at 115 lines, relaunched unchanged as run 390 (commits made in between).

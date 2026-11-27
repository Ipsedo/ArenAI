# train_423_ppg

| | |
|---|---|
| machine | the-machine |
| user | samuel |
| launch | 2026-11-12 23:46:38 |
| commit at launch | `7422d098` — skip separately between policy action groups (canon skip and prevent fire action to learn) |
| commits ≤ 90 min after | none |
| diff at launch | unknown (identity recorded after the run ended) |
| raw data | `outputs/train_423_ppg` on the-machine |

## Established facts

_Run 423 PPG (802b1b32: gun laying + critic-less aim advantage φ + per-group KL stop) — 1st run at D=1000 (429k), hit 0.126 @ D400 → 0.19-0.22 @ D1000, fire 1.6 → 0.8/s, ev 0.74, Hc ~0; 422 = same config with joint KL stop, stopped at 148k_

Run 423 PPG launched 2026-11-12 23:46 on 802b1b32 (high confidence), analysed at 581k (running).
Tested vs 422: per-group KL stop (canon aim vs rest), 7422d098. Inherited from 422 (1e171bbf):
canon stabilisation, canon advantage = trajectory potential φ, aim_gae_lambda 0, delta aim.

- D: 500 @ 145k, 1000 @ 429k (final bound, never reached before; 421 stuck at 400).
- hit 0.126 (D400) → 0.193 (D1000), 0.217 @ 581k; fire 1.6 → 0.8/s; kill flat ~21; miss ∝ D.
- ev 0.70-0.75, Hc trough −1.3 @ 59k then ~+0.1; canon α active 0.01-0.035.
- 422 (joint KL stop, stopped at 148k): same D milestones and hit up to 145k, but skip 0.5-0.6 and Hc −1.05;
  423 skip ≤ 0.15. Gain beyond 148k not attributable between the two commits.

**Why:** first real unlocking of the curriculum, see run 421.
**How to apply:** curriculum saturated (hit ≫ 0.08) → judge on hit/fire at D=1000; trigger
selectivity not proven by the CSV → probe save_18-21 proposed.

**Probe save_21 (567k, D=1000, 2026-11-13)**: selective trigger, ratio 4.9 (side 550) / 7.5 (side 1000),
hit/shell 0.23 / 0.19, disarmed 6-10 % (dump removed), turret < 5° 24-28 %; but corr(μ canon, error) ≈ 0
→ alignment does not come from a proportional canon command. Deterministic: fires in only 9-24 % of
aligned states. The probe now handles PPG (DistillationActor branch in probe.cpp).

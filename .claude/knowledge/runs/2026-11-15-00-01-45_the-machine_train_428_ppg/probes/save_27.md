# Probe train_428_ppg / save_27

- Checkpoint: index ≈ 729k, D = 1000 (curriculum saturated since 429k), just after the CSV hit rise (0.26 @ 724k).
- Binary: `cmake-build-release/arenai_agent/arenai_agent_probe`, recompiled 2026-11-15 at 10:47 at 9d843717. This commit only adds models and the desktop preset relative to the run code (0e5093a1): **no inference diff**. The old probe binary (7422d098) lacked the canon a·|a| mapping and would have biased the probe.
- Protocol: sides 550 and 1000, 4 episodes each, stochastic mode, economy 10 / +3 / 30.

## Summary, compared to 423 save_21 (567k, D = 1000)

| side | run | ratio | p_al (n) | p_off | p_arm | hit/shell | h<5° | shots<5° | enrich | empty_s | %empty | %disarmed | fire/s | turret<5° | σx | σy | p_zoom |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 550 | 423 | 4.86 | 0.359 (836) | 0.074 | 0.139 | 0.229 | 0.369 | 0.500 | 2.04 | 9.7 | 0.70 | 0.095 | 1.08 | 0.277 | 0.37 | 0.37 | 0.47 |
| 550 | **428** | **8.60** | 0.146 (1441) | **0.017** | 0.054 | **0.268** | 0.345 | **0.649** | 2.00 | 31.9 | **0.49** | **0.026** | 0.50 | **0.340** | 0.35 | 0.34 | 0.40 |
| 1000 | 423 | 7.53 | 0.293 (592) | 0.039 | 0.061 | 0.191 | 0.327 | 0.467 | 2.15 | 20.0 | 0.73 | 0.060 | 0.53 | 0.235 | 0.58 | 0.56 | 0.39 |
| 1000 | **428** | **21.9** | 0.149 (888) | **0.007** | 0.025 | **0.270** | 0.352 | **0.656** | 2.60 | 46.1 | **0.39** | **0.011** | 0.23 | 0.264 | 0.44 | 0.43 | 0.34 |

Armed p_fire by angle to the best target ([0-2[, [2-5[, [5-10[, [10-20[, [20-45[, [45-180[ °):
- side 550: 0.142 / 0.092 / 0.047 / 0.026 / 0.023 / 0.012
- side 1000: 0.097 / 0.051 / 0.020 / 0.010 / 0.008 / 0.006

## Reading

- **Trigger.** Clearly more selective than 423: ratio ×1.8 (side 550) and ×2.9 (side 1000). Still monotone with angle. But this gain comes mostly from **the extinction of off-axis shots** (p_off divided by 4 to 6). The fire probability when the barrel is aligned also drops, by about half (0.36 → 0.15). The global trigger level p_arm is divided by 2.5.
- **In deterministic mode** (p > 0.5), the trigger fires in only 6.8 % (side 550) and 4.1 % (side 1000) of aligned states, vs 24 % and 9 % in 423. **In deterministic game, the agent is nearly mute.**
- **Precision.**
  - Hit/shell goes from 0.19–0.23 to 0.27.
  - Hit/shell of aligned shots (h<5°) does not move: 0.35 vs 0.33–0.37.
  - The gain therefore comes from **the composition of the shots**: 65 % of shots are aligned, vs 47–50 % in 423. Not better ballistics at equal angle.
- **Dump.** Strongly reduced:
  - the share of tank-episodes emptying their reserve goes from 70–73 % to 39–49 %;
  - the first empty reserve arrives at 32–46 s instead of 10–20 s;
  - the share of disarmed steps goes from 6–10 % to 1–3 %.
- **Aim.**
  - Turret at < 5° of a target: 0.34 / 0.26 vs 0.28 / 0.24 in 423. Slightly better, within noise on side 1000.
  - median canon σ: 0.35–0.44, never at ceiling. No canon μ saturation (≤ 1 %).
- **Zoom.** Correlated with the trigger (corr 0.43–0.48). The zoom share rises with alignment, as in 423.
- **Lifetime.** Median 96 s (side 550) and 180 s (side 1000): more passive play, consistent with the CSV kill drop.

## Conclusions

**Established**
- The CSV switch around 560–620k (fire ↓, hit ↑) is a **selection** learning: off-axis shots are removed, and the reserve is no longer emptied.
- It is neither better precision per aligned shot, nor clearly better aim.
- The price is a globally low trigger, including when the agent is aligned. Hence fewer kills and a mute deterministic mode.

**Undecided**
- The origin of the change: discrete target 0.01 (α_fire at floor, nothing holds the trigger low) or γ 0.998. The two are confounded.
- The canon μ / angular error correlation did not come out of the probe. It is therefore unknown whether the canon command became proportional (it was not in 423).

## Useful next steps
- Probe `save_19` (≈ 510k, before the switch) to have the « before » at the same D in the same run.
- Probe a later save to see whether p_al rises back (selective and active trigger) or keeps dropping (towards an extinguished trigger).

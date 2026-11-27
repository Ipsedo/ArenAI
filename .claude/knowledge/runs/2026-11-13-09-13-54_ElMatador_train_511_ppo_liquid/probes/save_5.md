# Probe 511 save_5 (135 k, D 450) vs 510 save_5

`build/` binary of 13/11 09:13, commit deca02d4 (no respawn).
- 511: no inference diff.
- 510 save_5: re-probed in `--force`, only diff = respawn removed. Both policies compared under the same rule.
- Old 510 save_5 probe (with respawn: ratio 1.86/2.12, reserve 3.4/3.0 s) archived in `train_510_ppo_liquid/probes/binaire_1e171bbf_avec_respawn/`.

4 episodes per side, stochastic, 32 tanks.

| run | side | p_arm | ratio | <5°-only ratio | hit/sh | h<5° | empty_s p10/p50/p90 | %empty | %disarmed | shots/s | turret<5° | σ cx/cy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 510 s5 | 550 | 0.58 | **1.58** | **1.46** | 20.7 % | 37.4 % | 2.2/**3.1**/7.8 | 90 % | 79 % | 0.113 | 44.6 % | 0.25/0.26 |
| 511 s5 | 550 | 0.82 | 1.06 | 1.08 | 19.0 % | 34.3 % | 1.9/2.3/6.7 | 94 % | 86 % | 0.103 | 44.2 % | 0.38/0.34 |
| 510 s5 | 1000 | 0.47 | **1.52** | **1.41** | 12.2 % | 24.2 % | 2.2/**3.0**/5.2 | 98 % | 92 % | 0.077 | 32.7 % | 0.39/0.35 |
| 511 s5 | 1000 | 0.80 | 1.09 | 1.02 | 12.8 % | 29.0 % | 1.9/2.2/4.0 | 99 % | 92 % | 0.084 | 29.9 % | 0.49/0.42 |

Aligned & <40 m counts: 511 124/63, 510 120/109; <5° only: 805/499 vs 967/791.

CSV at 135 k: 511 hit 25 % (510: 15 %), skip_canon 0.19 (0.13), kl 0.038 (0.045). Kills not comparable (zombies).

## Reading

- **Reserve**: 511 empties **faster** than 510 at same index (2.2-2.3 s vs 3.0-3.1 s). The no-respawn rule did not push the dump back, it is even more marked.
- **Trigger**: 510 took off between save_3 and save_5 (ratio 1.05 → 1.5, p_arm 0.80 → 0.47-0.58). 511 stayed flat and even rose (p_arm 0.72-0.75 → 0.80-0.82, ratio 1.13-1.22 → 1.06-1.09). Save_3's onset did not hold.
- **Precision**: 511's lead at save_3 closed (h<5° 34/29 % vs 37/24 %, hit/shell equal). CSV hit stays higher in 511. Hypothesis: composition effect, as its shots are concentrated in the first seconds, at short range.
- **Canon**: σ wider than in 510, no trough. Turret equivalent.

## Established / noise

- Established (three saves, two sides, same binary): without respawn, the 1st empty reserve does not recede. At 135 k, 511's trigger lags 510's (ratio 1.1 vs 1.5).
- One seed per run: the trigger lag could be seed noise. 509 had the same kind of lag vs 508 at same index.
- Unverified hypothesis: when almost everyone is dry in ~3 s, being disarmed costs almost nothing. Nobody can hit you any more (probed median life 180 s), so emptying is not punished. Dump would then be a collective equilibrium rather than an individual flaw. To verify: share of `r_rcv` received while disarmed in 511's episodes, or a probe where only a few tanks are the agent facing armed tanks.

## Next

- save_9 (243 k, around 13:30-13:40) vs 510 save_9 (ratio 6.4/10.8, reserve 9.1 s with respawn): does 511 take off late or not at all?
- Decision for Samuel if save_9 stays flat: fallback « respawn delayed by N s », validated in principle 13/11.

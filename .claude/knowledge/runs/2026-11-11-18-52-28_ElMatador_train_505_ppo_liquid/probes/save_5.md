# Probe 505 save_5

- Run: train_505_ppo_liquid (turret + canon absolute setpoint, yaw (cos, sin), MIN_ALPHA 1e-4)
- Save: save_5, index 135 k, D 400 (reached at 67 k; 504: 114 k; D 450 at 145 k for both)
- Binary: `build/arenai_agent/arenai_agent_probe`, compiled 11/11 19:55, commit f9df436a + 505 changes; only change since run = rename `probe/main.cpp` → `probe.cpp`, no inference diff
- Sides 550 / 1000, 4 episodes, stochastic, economy 10 / +2 / 20

## Summary (505 save_5 vs 504 save_5, same index)

| side | run | ratio | angle-only ratio | p_arm | hit/sh | h<5° | empty_s | %disarmed | shots/s | turret<5° | σ yaw (cos/sin) | \|μ\| yaw (cos/sin) | σyaw° |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 550 | 505 | 3.3 | 1.25 | 0.13 | 7.1 % | 18.2 % | 11.9 | 20 % | 0.91 | 14.3 % | 0.44 / 0.35 | 0.21 / 0.16 | 72.6 |
| 550 | 504 | 3.5 | n/a | 0.20 | 8.4 % | 16.2 % | 5.9 | 46 % | 0.86 | 13.8 % | n/a | n/a | n/a |
| 1000 | 505 | 3.2 | 1.06 | 0.12 | 2.6 % | 6.5 % | 10.4 | 20 % | 0.85 | 14.3 % | 0.45 / 0.36 | 0.21 / 0.15 | 73.0 |
| 1000 | 504 | 5.0 | 1.57 | 0.15 | 3.9 % | 7.7 % | 5.7 | 47 % | 0.69 | 12.7 % | n/a | n/a | n/a |

## Reading

- **Trigger**: level fell since save_2 (0.53 → 0.13), hence a reserve lasting 2× longer than 504's (11-12 s vs 6 s, 20 % disarmed vs 47 %). Selectivity is mostly a distance matter: angle-only ratio 1.25 / 1.06 (504 side 1000: 1.57). Shot enrichment 1.05 / 1.00: shots not more aligned than decisions.
- **Aim**: turret <5° = 14 %, equal to 504 and down from save_2 (16 %). Absolute angle brings no measurable aim gain at 135 k.
- **Yaw setpoint**: σ_yaw rose 0.28 → 0.44 / 0.35, |μ_yaw| 0.13 → 0.21 / 0.16. |μ| stays 2× under σ ⟹ setpoint dispersion 73° (77° at save_2). Absolute yaw learned almost nothing.
- **Why σ rises**: alpha_yaw_sin and alpha_pitch rose to 0.004 (65-90 k) because per-dim entropy fell under target 0, then fell back to floor 1e-4 once H ≈ 0.2-0.3. With target 0, PID keeps σ around 0.25 at minimum (Gaussian H = 0 ⟺ σ ≈ 0.24, more with tanh compression).
- **Hit/shell**: 7.1 / 2.6 %, under 504 (8.4 / 3.9 %). Aligned shots hit a bit better (18 vs 16 %), within noise.

## Established vs noise

- Established: yaw setpoint stays almost random at 135 k (|μ| < σ); aim no better than 504's at same index; dump less pronounced (reserve 2× longer).
- Hypothesis to verify (not measured): in (cos, sin), angular precision ≈ σ/|μ| with |μ| ≤ 1, and entropy target 0 per dim holds σ ≳ 0.25 ⟹ ~14° dispersion floor even at |μ| = 1. In velocity the same σ cost only a few degrees per decision.
- Not established: D lead (400 at 67 k); both runs at D 450 at same index.

## Next

Read save_8-9 (216-243 k) vs 504 save_9 (hit/shell 16 %, reserve 22 s, ratio 6.1). Indicators: |μ_yaw| vs σ_yaw, σyaw°, turret<5°. If |μ_yaw| stagnates under σ, the question to settle is the entropy target of yaw dims.

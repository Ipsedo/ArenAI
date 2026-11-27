# Aim credit in the advantage — 508 save_9 (243 k)

`arenai_agent_sigma_gradients` (`build/` binary of 12/11 evening) replays full training rollouts with the trainer's policy, critic and GAE (300 steps × 32 tanks, stochastic). The tool also exports per transition the drawn pre-tanh action `u`, the reward decomposition (hit, kill, received, death; Φ = reward − 3·hit − kill − received − death) and aiming geometry. 2 episodes per side: 66.8 k transitions side 550, 68.6 k side 1000.

## Method

- At the first PPO step (ratio = 1), the loss gradient on a dimension's μ is A·ε/σ per sample, with ε = (u − μ)/σ the drawn noise.
- d = −sign(gap to target) × sign of command gain. ε·d > 0 means noise brought the barrel closer to the target.
- **ρ = corr(A, ε·d)** is the part of the μ gradient pushing toward correction, per sample. The t of a 512-minibatch is ρ·√512: below ~2, the direction of each Adam step on μ is noise.
- Gains measured in these rollouts: +8.4° yaw per turret action unit; +3.6 to +4.0° pitch gap per canon action unit, measured with target on axis (|yaw| < 10°, pitch sensitivity = cos(yaw)).
- Retained samples: target at 1-40° gap, outside last 30 steps of each rollout. Turret: n = 17-19 k per side. Canon: n = 8 k per side. Standard error of ρ about 0.007 for turret and 0.011 for canon.

## Results

Trainer advantage (critic + GAE λ 0.95, normalised):

| | turret 550 | turret 1000 | canon 550 | canon 1000 |
|---|---|---|---|---|
| ρ | +0.003 | +0.020 | +0.015 | +0.017 |
| t per minibatch 512 | 0.06 | 0.45 | 0.33 | 0.39 |

Trainer advantage shares 50-69 % of its variance with the λ-sum of hits and kills, 5-11 % with received damage, 0-0.6 % with Φ's.

**Reward Φ reacts well to the gesture, from the decision itself.** Correlation between ε·d and the same decision's Φ reward: +0.12 to +0.14 for turret (gap 5-40°), +0.11 to +0.23 for canon. At the next decision correlation is slightly negative (−0.03 to −0.09).

**ρ as a function of the advantage's λ horizon (λ-sum without baseline):**

| summed reward | dim | 550: λ 0 / 0.5 / 0.8 / 0.95 | 1000: λ 0 / 0.5 / 0.8 / 0.95 |
|---|---|---|---|
| Φ alone | turret | +0.116 / +0.072 / +0.031 / +0.005 | +0.099 / +0.069 / +0.040 / +0.021 |
| Φ alone | canon | +0.114 / +0.090 / +0.041 / +0.003 | +0.129 / +0.112 / +0.062 / +0.025 |
| aim (Φ + 3·hit + kill) | turret | +0.051 / +0.025 / +0.006 / −0.004 | +0.075 / +0.052 / +0.024 / +0.012 |
| aim (Φ + 3·hit + kill) | canon | +0.053 / +0.045 / +0.027 / +0.010 | +0.058 / +0.050 / +0.029 / +0.011 |
| total reward | turret | +0.048 / +0.023 / +0.006 / −0.002 | +0.073 / +0.052 / +0.026 / +0.014 |
| total reward | canon | +0.044 / +0.036 / +0.019 / +0.006 | +0.053 / +0.045 / +0.026 / +0.010 |

## Established

1. **At the current horizon (λ 0.95), the μ gradient contains almost no aim information.** ρ is 0.003 to 0.02, i.e. 0 to 2.5 standard errors. Each Adam step on μ_canon_x / μ_canon_y goes in an almost random direction (t per minibatch < 0.5). Measured cause of μ's random walk (state white noise, growing variance, 4-14 % tied to the gap) and of a correction gain that does not move.
2. **The information exists in the reward, at the step itself.** Φ reacts to the gesture with correlation 0.10 to 0.13. That is 10 to 15 standard errors, and ρ·√512 ≈ 2.3-2.9 per minibatch.
3. **The time horizon destroys it.** ρ decreases monotonically with λ, for both dimensions, both sides and all three rewards. The sum of ~20 decisions adds the variance of following gestures, target motion, and above all hits and kills, which dominate the advantage.
4. **A separate « aim » advantage at the same horizon (lever B as proposed) does not help.** At λ 0.95, the aim λ-sum gives ρ −0.004 to +0.012, no better than the current advantage. Separating rewards is not enough: the horizon matters.
5. Even the total reward at λ 0 without baseline carries 3 to 20× more aim signal than the current advantage.

## Limits

- First-order gradient measurements, on a single checkpoint. They show where information is lost, not that short-horizon training would learn better.
- Alternative λ-sums are without baseline. A dedicated critic would remove variance, but the TD error it adds at low λ is not measured here.
- Fire and movement need the long horizon: hits arrive after shell flight and the kill reward is deferred. A short horizon would only apply to aim dimensions.

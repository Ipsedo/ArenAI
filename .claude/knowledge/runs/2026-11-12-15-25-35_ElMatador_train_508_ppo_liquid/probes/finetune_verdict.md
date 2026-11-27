# Causal test of the aim-credit horizon — 508 save_9

12/11, 19:20-20:05. `arenai_agent_aim_finetune` (`build/`, uncommitted): from save_9, 40 on-policy PPO actor updates (rollouts of 300 steps × 32 tanks, side 550), critic frozen. `LiquidPpoTrainer::train_actor` replaces the GAE advantage by the supplied advantage, normalised the same way. Two arms, two trials each:

- **trainer**: usual GAE advantage (λ 0.95, critic);
- **immediate**: reward Φ of the decision alone (λ 0, no baseline), applied to all dimensions.

Probe afterwards: 3 episodes side 550 (4 for the start).

| arm | turret<5° (per episode) | median best_angle | nearest <5° | hit/shell | p_fire |
|---|---|---|---|---|---|
| start save_9 | 13.1 % (11.4-14.2) | 22.4° | 9.0 % | 15.5 % | 0.27 |
| trainer a | 8.3 % (7.7-9.2) | 29.7° | 8.9 % | 11.9 % | 0.22 |
| trainer b | 13.6 % (12.5-14.8) | 21.3° | 8.3 % | 15.1 % | 0.30 |
| immediate a | 18.8 % (17.1-20.6) | 11.5° | 9.2 % | 11.7 % | 0.17 |
| immediate b | 18.1 % (15.3-20.6) | 14.0° | 10.6 % | 14.5 % | 0.16 |

## Established

- Long-horizon advantage does not make aim progress: in 40 updates it stagnates or regresses (8-14 %). Immediate credit makes it progress (18-19 %) in both trials. Each episode of the immediate arms exceeds each episode of start and controls, except one episode at 15.3 %. The credit horizon is a causal bottleneck of turret learning.
- Immediate credit removes the constant bias of μ_canon_x when the gap is null (+0.06 to +0.10, about 1°/decision rotation).

## Not established

- The gain aims at *an* enemy (median best_angle 22 → 12-14°), not the nearest: its share <5° unchanged (9-11 %), correction gain too (−0.03 to −0.05 per 10°). This is what Φ rewards.
- No gain in hits per shell (11.7-14.5 % vs 15.5 %). Test is confounded: Φ advantage also applied to trigger and movement, and p_fire falls from 0.27 to 0.16.
- 40 updates indicate a direction of variation, not a ceiling.

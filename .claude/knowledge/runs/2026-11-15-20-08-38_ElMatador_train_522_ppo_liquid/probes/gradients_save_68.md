# Aim credit by short-horizon critic: 522 save_68 (1 836 k, D 1000)

Question: should the critic be put back into the aim advantage, over a single step horizon?

## Method

`build/arenai_agent/arenai_agent_sigma_gradients` (2026-11-15 20:06, run code), unmodified: the tool reads `gae_lambda` from `config.json`, so save_68 was replayed with four config copies (λ = 0, 0.3, 0.6, 0.95). Column `advantage` is then the trainer's advantage with the run's critic at that horizon; at λ = 0 it is r + γV(s′) − V(s). 2 episodes per side and per λ, 45-67 k transitions per file, different rollouts across λ. The tool's environment does not apply the empty penalty.
Measure of 2026-11-12 (508 save_9): ρ = corr(A, ε·d), ε the noise drawn on the dimension, d the direction that approaches the nearest enemy. Gain sign estimated in data (corr action → gap variation 0.32-0.43 yaw, 0.58-0.63 site). Samples outside the last 30 steps of each rollout. Threshold retained 2026-11-12: ρ ≥ 0.08.

## Results (550 / 1000)

Turret, gap 1-40° (n = 11.5-16 k, s.e. 0.008-0.009):

| advantage | λ 0 | λ 0.3 | λ 0.6 | λ 0.95 |
|---|---|---|---|---|
| run's critic | +0.037 / +0.003 | +0.042 / +0.010 | +0.019 / +0.017 | +0.015 / +0.006 |
| Φ at same step (current) | +0.129 / +0.110 | +0.115 / +0.099 | +0.107 / +0.110 | +0.097 / +0.086 |
| step reward, no critic | +0.055 / +0.043 | +0.060 / +0.031 | +0.059 / +0.035 | +0.054 / +0.026 |

Canon, yaw < 10° and site gap 1-40° (n = 5.5-7.5 k, s.e. 0.012-0.013):

| advantage | λ 0 | λ 0.3 | λ 0.6 | λ 0.95 |
|---|---|---|---|---|
| run's critic | +0.035 / +0.024 | +0.011 / +0.028 | −0.005 / +0.007 | +0.001 / 0.000 |
| Φ at same step (current) | +0.123 / +0.070 | +0.111 / +0.108 | +0.112 / +0.102 | +0.080 / +0.095 |

Turret, nearest enemy out of field (≥ 40°, n = 26-44 k, s.e. 0.005-0.006):

| advantage | λ 0 | λ 0.3 | λ 0.6 | λ 0.95 |
|---|---|---|---|---|
| run's critic | −0.001 / −0.008 | +0.001 / +0.006 | +0.019 / +0.005 | +0.005 / +0.006 |
| Φ at same step (current) | −0.012 / 0.000 | −0.007 / 0.000 | 0.000 / +0.010 | +0.003 / −0.013 |

## Signal received by the trigger (same data)

Normalised advantage gap between firing and not firing, armed decisions, λ 0.95 (the run's):

| state | 550 | 1000 |
|---|---|---|
| aligned < 2°, target < 100 m | +1.12 ± 0.10 | +1.43 ± 0.13 |
| aligned < 2°, target 100-200 m | +0.71 ± 0.14 | +0.84 ± 0.20 |
| 5-20° | +0.04 ± 0.10 | −0.22 ± 0.14 |
| ≥ 20° | −0.41 ± 0.06 | −0.50 ± 0.09 |

At λ = 0 the gap is positive everywhere, off-axis included (+0.35 to +1.96): the critic's one-step advantage does not separate good shots from bad ones.

## Established

1. The one-step critic does not carry the aim signal: ρ 0.00-0.04, under threshold 0.08 and 3 to 30× under Φ. No intermediate λ does better.
2. The critic removes information instead of adding: the step reward alone (0.03-0.06) correlates better than the same reward corrected by γV(s′) − V(s). The run's critic does not resolve one alignment step.
3. No target-search credit, with or without critic: ρ ≈ 0 out of field at all horizons (one point at +0.019, not reproduced at the other side).
4. Φ at the same step holds its 2026-11-12 level (0.10-0.13 turret, 0.07-0.12 canon).
5. The trigger receives a clear, well-oriented signal at λ 0.95: +1.1 to +1.4 std for an aligned shot < 100 m, negative off-axis.
Limits: ρ measures only the immediate corrective gesture toward the nearest enemy, with the critic as at save_68 (ev 0.55-0.65); says nothing of what a differently trained critic would learn. Trigger advantage gap biased by selection: at equal alignment, shots leave in states the policy already judges better.
Reading: putting the critic back into aim, even over one step, would dilute Φ without adding coupling. Point 5 shifts the question: the trigger receives the right gradient and does not follow it (aligned p_fire < 100 m outside a burst 0.08). Most probable reading, not demonstrated: neither the trigger head nor the critic finely distinguishes "aligned" in features, so raising p_fire on aligned states also raises it on neighbouring states where advantage is negative.

---

# Addendum: dedicated critic head for aim (2026-11-16 evening)

The measure above uses the existing critic head (full return, all rewards, long horizon). A dedicated head is another object; measure on the same 8 files (script `tete_dediee.py`).

Two cases:
- Head dedicated to aim reward Φ: moot. The reward is a potential difference, its return = −Φ(s); an exact critic cancels the one-step advantage (γΦ′ − Φ − γΦ′ + Φ = 0). Φ at the same step works because it has no baseline.
- Head dedicated to touches and kills: V(s) = discounted return of 3·hit + kill, one-step advantage g + γV(s′) − V(s). The measured case.

Method: simulated head, V fit by table on the nearest enemy's geometry (|yaw| 16 classes, |site| 7, distance 4, armed), target = discounted 3·hit + kill return over 100 decisions (γ 0.998), fit on 7 files and evaluated on the 8th. **Upper bound**: the head receives the exact geometry; a real head must read it in features.
Learned V (armed, site < 1°): at 0-0.5° yaw 2.80 / 2.13 / 1.18 / 0.20 for a target at 0-50 / 50-100 / 100-200 / 200+ m; at 4-5° 2.09 / 1.35 / 0.37 / 0.10; at 20-30° 1.07 / 0.75 / 0.21 / 0.02.

Results (mean ρ over 4 files per side, min..max):

| sample | side | dedicated head, 1 step | current Φ | step hit + kill, no critic | Φ + dedicated head |
|---|---|---|---|---|---|
| turret 1-40° | 550 | +0.080 (0.067..0.098) | +0.112 | +0.006 | +0.124 |
| turret 1-40° | 1000 | +0.090 (0.069..0.116) | +0.104 | −0.001 | +0.122 |
| turret 5-40° | 1000 | +0.117 (0.100..0.151) | +0.091 | +0.003 | +0.134 |
| turret 1-5° | 550 / 1000 | +0.082 / +0.076 | +0.125 / +0.141 | ≈ 0 | +0.131 / +0.137 |
| canon 1-40° | 550 / 1000 | +0.075 / +0.082 | +0.108 / +0.094 | ≈ 0 | +0.117 / +0.112 |
| turret out of field ≥ 40° | 550 / 1000 | +0.009 / +0.019 | −0.004 / −0.001 | ≈ 0 | +0.004 / +0.012 |

Established: (1) a dedicated head changes everything vs the existing head: 0.075-0.09 vs 0.00-0.04; reaches the 0.08 threshold, worth 70-90 % of Φ, exceeds it at 5-40° side 1000. (2) Without critic the step's touches carry nothing (ρ ≈ 0): all signal comes from V(s′) − V(s). (3) The two signals add: Φ + dedicated 0.112-0.134 = +10 to +30 % over Φ alone. (4) Out of field a weak but correct-sign signal: +0.015 to +0.024 on the 4 files side 1000 (3-4 s.e. each), +0.005 to +0.016 side 550; Φ is null there; 0.2 to 0.4 std per 512-minibatch. (5) V weights by what the game pays: a target aligned at 50-100 m is worth 10× one aligned beyond 200 m; Φ does not make this difference.
Limit: upper bound. The existing head, which must read the image, does worse than the raw reward; nothing guarantees a learned dedicated head reaches the table's level. Possible intermediate test without touching the trainer: export features (`--features`), fit offline a linear head on the hit + kill return, recompute ρ.

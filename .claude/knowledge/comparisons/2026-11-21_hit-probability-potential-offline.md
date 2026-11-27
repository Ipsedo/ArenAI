# Offline measurement — cannon credit on the hit probability (2026-11-21)

Runs: 543 (save_8, save_13), 541 (save_12).
Question: does a "hit probability" potential P, alone or summed with the event advantage, credit the aiming gesture at one step? No training code touched. Scripts and data: `outputs/comparisons/2026-11-21_potentiel_proba_touche/` (`proba.py`, `flanc.py`, `data/*.csv`, raw output `data/proba_resultats.txt`).

## Method

- Data: `arenai_agent_sigma_gradients` (`build/` binary of 20/11 18:39), 4 episodes, 32 tanks, on 543 save_8 (sides 550 and 1000), 541 save_12 (550), 543 save_13 (550).
- P = 1 − ∏(1 − p_i), p_i = share of a Gaussian of dispersion σ centred on the cannon that falls within the angular width of enemy i (half-width 3.5 m). 2D positions of all living tanks; angle and distance of the nearest taken as the game gives them.
- Φ: lateral replica of the removed potential (max over enemies, 3 m bell).
- One-step advantage γ·X(s′) − X(s), γ 0.998. Event GAE = column `advantage` (run's critic, λ 0.95).
- Measure: correlation with the corrective gesture (definitions of `credit.py`), standard error by bootstrap over tank-episodes. Usefulness threshold fixed 2026-11-12: 0.08.

Limits: no terrain occlusion, no ballistics; elevation (site) known only for the nearest enemy; the Φ replica gives 0.05 where the real Φ measured 0.085-0.12 on the 522 policies, so absolute values are not comparable to this threshold from one run to another. The reliable comparison is P against Φ on the same rows.

## Results

Yaw gesture toward the target, 1-40°, target within 100 m (± 0.01):

| signal | 543 s8 s550 | 543 s8 s1000 | 541 s12 s550 | 543 s13 s550 |
|---|---|---|---|---|
| event GAE | +0.007 | −0.004 | +0.010 | +0.011 |
| Φ (replica) | +0.046 | +0.056 | +0.050 | +0.080 |
| P σ 1.6° | +0.051 | +0.052 | +0.050 | +0.113 |
| P σ 3° | +0.062 | +0.054 | +0.053 | +0.141 |
| P σ 6° | +0.076 | +0.056 | +0.049 | +0.167 |
| P σ 10° | +0.082 | +0.063 | +0.042 | +0.182 |
| P σ 1.6° + 0.25 × GAE | +0.051 | +0.049 | +0.050 | +0.112 |
| P σ 1.6° + 0.5 × GAE | +0.048 | +0.043 | +0.048 | +0.105 |
| P σ 1.6° + 1 × GAE | +0.040 | +0.031 | +0.041 | +0.085 |

Other gestures:

| gesture | signal | 543 s8 s550 | 543 s8 s1000 | 541 s12 s550 | 543 s13 s550 |
|---|---|---|---|---|---|
| yaw, target at 100-300 m | P σ 1.6° | +0.031 | +0.024 | +0.045 | +0.066 |
| yaw, target at 100-300 m | P σ 10° | +0.050 | +0.069 | +0.028 | +0.177 |
| yaw, enemy out of field (≥ 40°) | P σ 1.6° | −0.001 | +0.001 | +0.009 | +0.001 |
| yaw, enemy out of field (≥ 40°) | event GAE | +0.008 | +0.003 | −0.005 | +0.004 |
| yaw, enemy out of field (≥ 40°) | P + 1 × GAE | +0.004 | +0.003 | +0.003 | +0.003 |
| elevation (site) toward target | P (yaw only) | +0.006 | +0.006 | +0.001 | −0.002 |
| elevation toward target | site term alone, σ 1.6° | +0.112 | +0.135 | +0.148 | +0.142 |
| elevation toward target | P yaw × site, σ 1.6° | +0.060 | +0.062 | +0.056 | +0.059 |
| yaw toward target < 100 m | P yaw × site, σ 1.6° | −0.002 | +0.027 | +0.022 | +0.021 |

## Reading

Established:

- **P equals Φ for yaw.** Equal on three checkpoints, above on 543 save_13. It designates no target, and costs nothing in signal.
- **Events do not credit the gesture**, at any distance: 0.00 to 0.01, within standard error. Neither pointing, nor turning toward an off-field enemy.
- **The sum does no harm at low weight** (≤ 0.25), loses 5-15 % at 0.5 and 20-40 % at equal weight. It brings nothing measurable, on any gesture.
- **Elevation needs its own term**: 0.11 to 0.15 alone, 0.06 multiplied with yaw, and the multiplication also divides the yaw signal by two or more.
- **A wider dispersion carries further**: at 100-300 m, σ 10° gives +0.05 to +0.18 versus +0.02 to +0.07 for σ 1.6°. Clear effect on 543 (two saves), null or inverse on 541.

Not established:

- The absolute level against the 0.08 threshold (see limits). With σ 1.6°, P is at 0.05 on the three open-trigger checkpoints.
- That the sum settles target choice. Nothing credits the gesture toward an enemy on the flank: neither P, nor Φ, nor events.

## What it changes for lead 2 (see `2026-11-20_trigger-closure-diagnosis_508-543.md` §5, possibility 2)

- P can replace Φ without loss and without designating a target.
- The argument "events will choose the target" is not supported by the measurement: their signal on the gesture is null at one step. The sum stays harmless at weight ≤ 0.25.
- The cannon has two axes and one term per axis is needed.

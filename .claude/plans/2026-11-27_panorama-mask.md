# Plan — panorama encoder trained by an enemy-visibility mask

Work plan (skill `work-plan`): delete this file in the commit that completes it.

## Status

2026-11-27, nothing coded. Samuel chose the direction himself: « ne faut-il pas que le cnn
panoramique soit partagé entre l'acteur et la critique, et soit entraîné en mode masque sur les
ennemis visibles ? et ses features sont injectées (selon une méthode à benchmarker) à l'acteur et
la critique ? », then asked for the best version with « masque seul dans la loss » and said
« okay je l'ai […] je lancerai une nouvelle session pour le coder ». That validates the
direction only (G3): every line of §Open pieces goes through its gate before its code. The plan
re-opens, on the panorama encoder only and with new bench evidence, the presence-head lever
recorded as removed on 2026-11-25 in `docs/decisions.md`.

## Why

Runs 554 (row maximum) and 555 (row mean) feed a full-turn panorama to the actor and the critic,
each with its own panorama encoder. Measured on save 17 of both runs:

- the actor's panorama encoder is at the level of an untrained one; the bearing of an off-field
  enemy is not decodable from the liquid trunk (left/right 0.48–0.58, chance 0.50);
- the share of the advantage explained by the turret exploration noise towards the target is
  `|ρ| ≤ 0.015` in every distance and bearing band (noise floor): the policy gradient carries
  no usable signal to shape an encoder;
- the critic's encoder sees near enemies only: the return depends on the nearest enemy's
  position under 100 m (out-of-sample R² 0.21 at a 1–3 s horizon) and not at all at 100–200 m
  (R² 0.00 at every horizon), because no event involves an enemy beyond 100 m.

Bench conclusions (one-step policy-gradient bandit on labelled panoramas, see §Evidence):

- the supervised ranking of aggregations does not transfer to a policy gradient: every
  aggregation is within seed range at every signal level;
- what blocks is the representation: with an already-trained frozen encoder, the same weak
  signal trains the turret head; with the encoder trained by that signal, nothing is learned;
- a scalar target (value, even a value that would depend on far enemies) does not teach far
  bearing; a per-bearing target does;
- the more compact and explicit the quantity read by the head, the less the weak signal matters.

## Design

One panorama encoder, shared by the actor and the critic, trained only by a mask loss. The
trunks read a 3-value bearing summary computed from the predicted mask; policy and value
gradients stop at that summary.

```mermaid
flowchart TD
    P[Panorama image 512x32] --> E[Shared panorama encoder<br/>3 circular conv blocks]
    E --> F[Feature map<br/>64 channels x 4 rows x 64 columns]
    F --> A[conv 1x1 to 4 channels<br/>mean over the 4 rows]
    A --> L[conv 1x1 to 1 channel<br/>64 logits, one per bearing column]
    L --> S[sigmoid<br/>p_j, probability of a visible enemy]
    S --> B[Bearing summary<br/>3 values, no parameter]
    B -->|stop-gradient| TA[Actor trunk]
    B -->|stop-gradient| TC[Critic trunk]
    Y[Simulator state<br/>enemy positions, line of sight] --> M[Mask label y_j, weight m_j]
    L -.-> LOSS[Mask loss<br/>binary cross-entropy per column]
    M -.-> LOSS
    LOSS ==>|only gradient reaching the encoder| E

    subgraph Legend
        direction LR
        k1[solid arrow: forward pass, kept at inference]
        k2[dotted arrow: training only]
        k3[thick arrow: gradient]
    end
```

### Mask label

One label per bearing column, common to the 4 rows (no elevation information). With 64 columns
of 5.625° in the turret frame:

- `y_j = 1` if an enemy with a clear line of sight, closer than 400 m and within the panorama's
  vertical field, has its bearing inside column `j`; else 0;
- `m_j = 0` for an ambiguous column (an enemy overlaps it by its angular half-width
  `atan(4 m / distance)` but is occluded, too far or centred in a neighbouring column); else 1.

Labels come from the simulator state, never from the image, and exist at training time only.

### Loss

$$\mathcal{L}_{mask}=\frac{1}{\sum_j m_j}\sum_{j=1}^{64} m_j\,\mathrm{BCE}\big(\ell_j,\ y_j\big)$$

`ℓ_j`: logit of column `j`. Separate optimiser step from the policy and value losses, as nothing
else trains the encoder.

### Bearing summary

$$p_j=\sigma(\ell_j),\quad w_j=\frac{p_j}{\sum_k p_k},\quad
b=\Big(\sum_j w_j\cos\theta_j,\ \sum_j w_j\sin\theta_j,\ \max_j p_j\Big)$$

`θ_j`: bearing of the centre of column `j`. `b` replaces the 256 panorama features at the input
of both trunks. Per `docs/neural_architecture.md` the middle column of the panorama is the canon
heading and both ends meet straight behind: verify the column → bearing map in the code, the
bench panoramas were offset by 45° from the agent's.

## Evidence

Scripts and CSV in `outputs/comparisons/2026-11-25_exploration/` (local, not versioned):
`banc_bandit.py`, `banc_fige.py`, `banc_cibles.py`, `banc_injection.py`. Encoder pre-trained on
one target, frozen, then a linear turret head trained by a score-function gradient whose
advantage has correlation `ρ` with the useful exploration noise. Accuracy of the side of the
nearest off-field enemy (chance 0.50), under 60 m / 60–100 m / 100–200 m, mean of 2 seeds:

| encoder target → quantity read by the head | size | ρ = 0.05 | ρ = 0.02 |
|---|---|---|---|
| policy gradient itself (554/555 situation) | 256 | 0.50–0.53 / 0.52–0.55 / 0.51–0.56 | — |
| real critic encoder, 555 save 17 → features | 256 | 0.72 / 0.59 / 0.52 | 0.65 / 0.58 / 0.51 |
| mask 64 columns → encoder features | 256 | 0.67 / 0.67 / 0.67 | 0.64 / 0.63 / 0.65 |
| mask 64 columns → 64 probabilities | 64 | 0.70 / 0.71 / 0.70 | 0.65 / 0.65 / 0.64 |
| mask 16 cells → 16 logits | 16 | 0.68 / 0.72 / 0.73 | 0.67 / 0.71 / 0.72 |
| **mask 64 columns → bearing summary (this design)** | 3 | 0.73 / 0.73 / 0.74 | 0.70 / 0.71 / 0.71 |
| mask + per-column proximity → proximity-weighted summary | 3 | 0.80 / 0.82 / 0.75 | 0.76 / 0.78 / 0.73 |

Mask AUC on held-out panoramas: 0.84 (64 columns), 0.88 (16 cells).

Limits of the bench: one-step bandit, linear head, 2 seeds, bench panoramas offset by 45° from
the agent's views, task = side of the nearest enemy (favours the proximity weighting), and the
real `ρ` (≈ 0) is below the lowest level tested. The bench shows that a trained encoder divides
the signal needed; it does not show that the turret will aim in a real run.

Real-run measurements behind §Why: memory note `run-555-panorama-moyenne-debut` (and
`run-554-panorama-debut`), probe exports `outputs/train_55{4,5}_ppo_liquid/features/save_17_side1000*`.
Trap met while measuring `ρ`: the probe exports the action after tanh and `μ` before it; the
exploration noise is `ξ = (atanh(act) − μ) / σ`, and the good direction is `s = −sign(yaw)`.

## Out of scope

- The aiming credit itself stays near zero; the summary only makes what little there is usable.
- The summary averages bearings: two enemies on opposite sides cancel. The proximity variant
  (second label: `exp(-distance / 100 m)` per column) picks the nearest and is the best bench
  result; it was set aside for now to keep the mask as the only label.

## Validated pieces

None beyond the direction (see §Status).

## Open pieces

Each is a G3 piece (one `AskUserQuestion` per piece, technical detail first); 1, 3 and 4 are also
G1 (skill `design-gate`).

1. One shared encoder module owned outside the actor and the critic, or two encoders with the
   critic's weights copied; consequence on `actor.pt` / `critic.pt` and on the desktop game,
   which loads `actor.pt` only.
2. Stop-gradient at the summary (as benched), or policy/value gradients also allowed through.
3. What the trunks read: the summary alone (as benched), or the summary plus the 64
   probabilities.
4. Label source: where the per-enemy line of sight and bearing in the turret frame are computed,
   without putting a training-only signal in `arenai_model`.
5. Mask optimiser: learning rate, batch, number of epochs per rollout.
6. Scope: `ppo_liquid` only, or the four agents.
7. Fate of the 554/555 checkpoints, which no longer load.

## Steps

1. Read `docs/neural_architecture.md`, `docs/ARCHITECTURE.md` and the panorama encoder, actor,
   critic and trainer code of `arenai_agent` (`networks`, `agents/ppo_liquid`); note how the two
   networks are built, saved and loaded by the desktop game. Verify: the facts of this plan
   against the tree.
2. Put the open pieces to Samuel, in the order 1, 4, 2, 3, 6, 5, 7; record each answer here.
3. Mask label: compute per tank the 64-column `y`, `m` from the simulator state at each step and
   carry them in the rollout. Verify: unit test on a synthetic scene (one enemy at a known
   bearing, one occluded).
4. Shared encoder, mask head, bearing summary, stop-gradient, trunk inputs. Verify: unit test of
   the summary on a one-hot mask; shapes; the game still loads and plays a checkpoint.
5. Mask loss and its optimiser in the trainer(s) of the validated scope; CSV-only metric for the
   mask loss only if Samuel validates it (G2).
6. Update `docs/neural_architecture.md`, `docs/ARCHITECTURE.md`, `docs/USAGE.md` (new options),
   `docs/decisions.md` (decision row); `final-check`.
7. Smoke run into the scratchpad, then hand Samuel the binary and the command (G4: he launches).

## Evaluators

- Smoke run into the scratchpad: mask loss decreasing, held-out mask AUC against the bench's
  0.84.
- On the first saves of a real run: side of the nearest off-field enemy decodable from the
  trunk output (`decode_panorama.py`, 554/555 save 17: 0.48–0.58); prediction if the design
  works: clearly above 0.60 under 100 m and at 100–200 m.
- Verdict on the run: game match only (`docs/rl_method.md` M7), against 555 save 17.

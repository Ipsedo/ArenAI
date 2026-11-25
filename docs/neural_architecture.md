# Neural architecture

Actor and critic of `ppo` and `ppo_liquid`. Both agents share the three encoders and the heads;
they differ only by the **trunk**: an LTC cell (`ppo_liquid`) or an MLP (`ppo`). The actor and the
critic have the same structure and **no shared weight**.

Sizes below are the defaults (`liquid_ppo_hyperparams.h`, `ppo_hyperparams.h`) and the compile-time
constants of `arenai_core/types.h` and `arenai_model/constants.h`. $B$ is the batch size.

## Inputs

| input | tensor | content |
|---|---|---|
| vision | `uint8` $[B, 3, 128, 256]$ | camera on the canon, 79° wide (40° zoomed) |
| panorama | `uint8` $[B, 3, 32, 512]$ | 4 level views of 90° above the turret, side by side, left to right; the middle column is the canon heading, both ends meet straight behind; the tank's own parts are not drawn |
| state | `float` $[B, 146]$ | proprioception (144) + remaining shells ratio + cooldown ratio |

Both images are scaled to $[-1, 1]$ at the entry of their encoder: $x \leftarrow 2x/255 - 1$.

## Actor

```mermaid
flowchart TB
    subgraph VE["Vision encoder — ConvolutionNetwork"]
        direction TB
        V0["vision<br/>3 × 128 × 256"]
        V1["Conv 3×3, stride 2, pad 1 · GroupNorm(2) · SiLU<br/>16 × 64 × 128"]
        V2["Conv 3×3, stride 2, pad 1 · GroupNorm(4) · SiLU<br/>32 × 32 × 64"]
        V3["Conv 3×3, stride 2, pad 1 · GroupNorm(8) · SiLU<br/>64 × 16 × 32"]
        V4["Conv 3×3, stride 2, pad 1 · GroupNorm(16) · SiLU<br/>128 × 8 × 16"]
        V5["Conv 3×3, stride 2, pad 1 · GroupNorm(16) · SiLU<br/>128 × 4 × 8"]
        V6["Conv 3×3, stride 2, pad 1 · GroupNorm(16) · SiLU<br/>128 × 2 × 4"]
        V7["Conv 3×3, stride 2, pad 1 · GroupNorm(16) · SiLU<br/>128 × 1 × 2"]
        VF["Flatten<br/>256"]
        V0 --> V1 --> V2 --> V3 --> V4 --> V5 --> V6 --> V7 --> VF
    end

    subgraph PE["Panorama encoder — PanoramaNetwork"]
        direction TB
        P0["panorama<br/>3 × 32 × 512"]
        P1["Conv 3×3, stride 2, circular pad in width · GroupNorm(2) · SiLU<br/>16 × 16 × 256"]
        P2["Conv 3×3, stride 2, circular pad in width · GroupNorm(4) · SiLU<br/>32 × 8 × 128"]
        P3["Conv 3×3, stride 2, circular pad in width · GroupNorm(8) · SiLU<br/>64 × 4 × 64"]
        P4["Conv 1×1<br/>4 × 4 × 64"]
        P5["Max over the 4 rows<br/>4 × 64"]
        PF["Flatten<br/>256"]
        P0 --> P1 --> P2 --> P3 --> P4 --> P5 --> PF
    end

    subgraph SE["State encoder"]
        direction TB
        S0["state<br/>146"]
        S1["Linear (no bias) · LayerNorm · SiLU<br/>64"]
        S0 --> S1
    end

    VF --> CAT["Concatenate<br/>z ∈ ℝ^576"]
    PF --> CAT
    S1 --> CAT

    CAT --> TRUNK["Trunk<br/>ppo_liquid : LTC cell, 128 neurons, state x_t ∈ ℝ^128<br/>ppo : MLP 576 → 256 → 128<br/>h ∈ ℝ^128"]
    X["x_(t−1) ∈ ℝ^128<br/>(ppo_liquid only)"] -.-> TRUNK
    TRUNK -.-> XN["x_t"]

    TRUNK --> MU["Linear<br/>μ ∈ ℝ^4"]
    TRUNK --> SG["Linear · SoftClamp[−5, −0.357] · exp<br/>σ ∈ ℝ^4"]
    TRUNK --> DP["Linear · Sigmoid<br/>p ∈ [0, 1]^2"]

    MU --> A["a = tanh(u), u ~ 𝒩(μ, σ)<br/>dir_x, dir_y, canon_x, canon_y"]
    SG --> A
    DP --> D["d ~ Bernoulli(p)<br/>fire, zoom"]
```

## Critic

Same encoders and same trunk as the actor, with its own weights, ending on one value.

```mermaid
flowchart TB
    V["vision<br/>3 × 128 × 256"] --> VE["Vision encoder<br/>7 × (Conv 3×3 s2 · GroupNorm · SiLU) · Flatten<br/>256"]
    P["panorama<br/>3 × 32 × 512"] --> PE["Panorama encoder<br/>3 × (circular Conv 3×3 s2 · GroupNorm · SiLU) · Conv 1×1 · max over rows · Flatten<br/>256"]
    S["state<br/>146"] --> SE["State encoder<br/>Linear (no bias) · LayerNorm · SiLU<br/>64"]

    VE --> CAT["Concatenate<br/>z ∈ ℝ^576"]
    PE --> CAT
    SE --> CAT

    CAT --> TRUNK["Trunk<br/>ppo_liquid : LTC cell, 128 neurons, state x_t ∈ ℝ^128<br/>ppo : MLP 576 → 256 → 128<br/>h ∈ ℝ^128"]
    X["x_(t−1) ∈ ℝ^128<br/>(ppo_liquid only)"] -.-> TRUNK
    TRUNK -.-> XN["x_t"]

    TRUNK --> VAL["Linear<br/>V(s) ∈ ℝ"]
```

## Trunk

**LTC cell (`ppo_liquid`)**, `LiquidCell`. One decision integrates the state $x \in \mathbb{R}^{128}$
over $K = 6$ sub-steps of $\delta = \Delta t / K$, $\Delta t = 0.1$ s:

$$f = \operatorname{SiLU}(W_r\, x + W_z\, z + b), \qquad
x \leftarrow \frac{x + \delta\, f \odot a}{1 + \delta\,(1/\tau + f)}$$

$$h = \operatorname{SiLU}\big(\operatorname{LayerNorm}(W_o\, x)\big)$$

- $z \in \mathbb{R}^{576}$: the concatenated encodings; $W_z \in \mathbb{R}^{128 \times 576}$,
  $W_r \in \mathbb{R}^{128 \times 128}$, $W_o \in \mathbb{R}^{128 \times 128}$, none with a bias.
- $\tau = \operatorname{softplus}(\tau_{raw}) + 10^{-3} \in \mathbb{R}^{128}$: learned time
  constants, log-spaced at initialisation; $a \in \mathbb{R}^{128}$: learned, initialised at
  $1/\tau$.
- $x$ is carried from one decision to the next and reset at each episode start.

**MLP (`ppo`)**: two blocks `Linear (no bias) · LayerNorm · SiLU`, $576 \to 256 \to 128$; no memory.

## Heads

- **Continuous actions** (direction joystick, canon joystick): $u \sim \mathcal{N}(\mu, \sigma)$,
  $a = \tanh(u) \in [-1, 1]^4$. $\log \sigma$ goes through a `SoftClamp` (a sigmoid between the
  bounds) on $[-5, -0.357]$, so $\sigma \in [0.0067, 0.70]$.
- **Discrete actions** (fire, zoom): independent Bernoulli with probability $p$; fire is masked
  when the tank cannot fire.
- **Value**: one linear output, no activation.

## Panorama encoder details

- The panorama covers the full turn, so its left and right borders are neighbours: each strided
  convolution pads the width with the columns of the other border (circular) and the height with
  zeros.
- After three blocks one column spans 5.6° of bearing. The $1 \times 1$ convolution keeps 4
  channels per cell and the maximum over the 4 rows drops the height: the output is 4 channels ×
  64 bearing columns, flattened channel by channel.

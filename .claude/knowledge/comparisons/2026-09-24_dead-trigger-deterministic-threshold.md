# Dead trigger + deterministic threshold (run 413): game never fires

Runs: 409, 412, 413, 414, 415

2026-09-24: run 413 — p(fire) stays at initial_fire_proba after 3 M steps; in deterministic mode the 0.5 threshold is never crossed, so the game never fires.

Samuel's symptom in game on `train_413_ppo_liquid_normal`: « il se déplace bien mais ne tire presque jamais ». Neither a bad checkpoint (see comparisons/2026-09-20_game-does-not-fire-wrong-checkpoint.md) nor a bug: **the discrete head never learned**.

**Measured** (head `discrete.0` extracted from checkpoints, recipe below):

| checkpoint | p(fire) | logit std |
|---|---|---|
| 413 save_0 (init) | 0.400 | 0.01 |
| 413 save_28 (3.07 M steps) | **0.391** | 0.14 |
| 412 save_20 (2.23 M steps) | 0.372 | 0.18 |
| 409 save_57 (6.22 M steps) | **0.506** | 0.35 |

`initial_fire_proba = 0.40`. In game, `TorchLiquidPpoAgent::act(states, …)` calls `act(tensor, false)` ⇒ `bernoulli_max_action` = `(p > 0.5)`. At logit −0.44 ± 0.14, crossing 0.5 needs +3σ, ~0.1 % of states: **never**. Same for `p(zoom) ≈ 0.23`.

**Why the gradient is null.** Reload = 1 shell / 1.5 s, cooldown = 1/6 s (`jolt_enemy_tank.cpp`). At p = 0.39 and 30 Hz, the wait before the trigger fires is **2.6 frames (0.085 s)**, negligible against 1.5 s: the tank fires at the ammo ceiling whatever p over [~0.05, 1]. `fire` metric = 0.70/s ≈ 0.667 (regen) + 5 × hit rate. Behaviour insensitive to p ⇒ the head stays at its init. Same finding as run 353 (run 353 damage fix note), but here with the in-game consequence, which was missing.

So not a pathology but a **slow drift**: ~6 M steps (run 409) for the head to develop enough state dependence to cross 0.5. A run stopped at 3 M does not fire in deterministic, whatever else it is worth.

**Why (mechanism):** the deterministic 0.5 threshold turns an unlearned head into « never fires », while in training (sampled) the same network fires at the max allowed — the gap between training metrics and in-game feeling comes entirely from there.

**Rule:** before concluding anything about a checkpoint, print p(fire) and compare it to `initial_fire_proba` **and** to 0.5. Recipe without C++ build — `actor.pt` is an archive readable by `torch.jit.load`:

```python
import torch
sd = dict(torch.jit.load("outputs/<run>/save_N/actor.pt", map_location="cpu").state_dict())
w, b = sd['discrete.0.weight'], sd['discrete.0.bias']
logit = w.sum(1) * 0.205 + b          # 0.205 = E[SiLU(N(0,1))], output of LayerNorm+SiLU
print(torch.sigmoid(logit), w.norm(dim=1) * 0.558)   # 0.558 = std[SiLU(N(0,1))]
```
Same recipe on `sigma.0` / `mu.0` gives typical σ and μ — it is what exposed comparisons/2026-09-23_absolute-angle-aim-regression.md.

Leads discussed, **not validated**: solicit the fire action only on frames where `can_fire` is true (today sampled 30 times/s for ~0.7 useful shots/s), or raise stock/reload so cadence becomes a choice.

**Confirmed and refined at run 414 (2026-09-27).** The slow drift exists and the 0.5 threshold is crossed **earlier than estimated: ~3.8-4.8 M steps**, not ~6 M. p(fire) measured on 414 (recipe above, but **corrected for the LayerNorm affine** — see below): save_0 0.400 → save_35 0.499 → save_44 0.521 → save_106 0.536 → save_155 **0.583** (73 % of states above 0.5). So 414 fires in deterministic. `p(zoom)` stays 0.22 at all checkpoints ⟹ **zoom never triggers in game**, no run excepted.

Practical pitfall: `~/.cache/arenai/preferences.json` pointed to `save_44` (p = 0.52, 61 % of states), just above the threshold — testing there measures the threshold margin rather than the policy. Take the last checkpoint.

**Recipe correction.** `w.sum(1) * 0.205` assumes LayerNorm outputs raw N(0,1), but `liquid.to_output.1` has `elementwise_affine=true`: go through γ and β, else σ is wrong (on 414 the naive recipe gave σ_canon = 0.27 instead of 0.39 — enough to wrongly conclude σ had tightened).

```python
import torch
sd = dict(torch.jit.load("outputs/<run>/save_N/actor.pt", map_location="cpu").state_dict())
g, b_ln = sd['liquid.to_output.1.weight'], sd['liquid.to_output.1.bias']
h = torch.nn.functional.silu(torch.randn(200000, 1) * g + b_ln)   # realistic features
E, V = h.mean(0), h.var(0)
for head in ('mu', 'sigma', 'discrete'):
    w, b = sd[f'{head}.0.weight'], sd[f'{head}.0.bias']
    print(head, (w * E).sum(1) + b, ((w**2) * V).sum(1).sqrt())   # mean, std over states
# σ = exp(log(1e-4) * (1 - sigmoid(pre_act)))   ; p = sigmoid(pre_act)
```
The mean gives typical behaviour, the std the truly state-dependent part (on 414 save_155: μ_turret = +0.160 ± 0.229 ⟹ positive command in ~76 % of states, i.e. ~29°/s permanent sweep — the drift of comparisons/2026-08-17_agent-does-not-aim.md is still there).

**⚠ Recipe NOT reliable (found 2026-09-27, run 415).** Even LayerNorm-corrected, it assumes each `to_output` feature is N(0,1) *across states*; LayerNorm only normalises *across the features of one state*, so per-feature means and correlations are unknown without real states. Proof: on 415 save_7 it gives σ ≈ 0.10 on the 4 dims while the trainer measures `Hc` ≈ 0 (⟺ σ ≈ 0.25). All p(fire)/σ values drawn from weights alone (413 and 414) are indicative at best. For 413, « never fires » stays valid because Samuel **observed it in game**; for 414, « it will fire » is not established. Reliable sources: `Hc`/`Hd` of metrics.csv, and the in-game test.
(pitfall candidate for a skill/doc: weight-only probing recipe unreliable; deterministic threshold 0.5)

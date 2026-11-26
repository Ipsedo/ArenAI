---
name: ppo-vanilla-liquid-alignment
description: 2026-10-20 — ppo vanilla aligné sur ppo_liquid pour une comparaison (σ indépendant, facteurs d'entropie fixes, PID retiré, lr/hidden capteurs, masque fire) ; écarts restants
metadata:
  type: project
---

Samuel veut comparer ppo vanilla vs ppo_liquid (2026-10-20). Alignements faits à sa demande :
masque fire ([[ppo-liquid-agent]]), `Actor` vanilla en `sigma_logit` indépendant de l'état,
PID Lagrangien retiré du `PpoTrainer` → `--continuous_entropy_factor 0` /
`--discrete_entropy_factor 0.01` (métriques α_c/α_d disparues), critic lr 1e-4, hidden_size_sensors 64.
Samuel avait déjà mis les deux sur le même `ConvolutionNetwork` (stride 2 + spatial softmax, dbd11417).

Puis (même jour) tous les hyperparams communs alignés : vision_channels 16-32-64-128,
group_norm 2-4-8-16, minibatch 1536. Seul écart restant = structure (MLP [256,128] vs liquid+skip).
Code mort après le retrait : `PidLagrangianAlphaParameters`, `init_sigma_output_weights`,
`bernoulli_maximum_entropy` (seulement testés) — suppression proposée, en attente.
Vieux checkpoints vanilla (tête `sigma.0.*`) ne se rechargent plus.

2026-10-31 : après le run 383, **σ indépendant de l état pour les 3 agents** (Actor et LiquidActor : paramètre `log_sigma` [nb_actions], clampé [SIGMA_MIN, SIGMA_MAX] via state_free_sigma, SigmaOutput et init_sigma_output_weights supprimés) + actor_learning_rate 3e-4 → 1e-4. Anciens checkpoints (tête sigma.0.*) incompatibles.

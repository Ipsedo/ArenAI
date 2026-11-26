---
name: ppo-liquid-agent
description: "Agent LiquidPpo (2026-09-06) — CNN+MLP → LiquidCell (ex-LiquidRecurrent, fusionné 2026-09-17), TBPTT par chunks, hidden stockés dans le rollout, carry critique en GAE"
metadata:
  node_type: memory
  type: project
  originSessionId: bcf7911f-84bf-46ed-9846-c59fe08a8178
  modified: 2026-09-20T21:16:17.866Z
---

Agent `ppo_liquid` créé le 2026-09-06 (`arenai_agent/src/agents/ppo_liquid/`,
réseaux dans `src/networks/recurrent/liquid_actor|liquid_critic`). Design validé par Samuel :

- `LiquidRecurrent` a reçu (ajouts purs) `forward_step(x_t, input_t) -> (output, x_t_next)`
  appelé par le `forward` séquence, et `get_first_x` rendu public. x0 reste aléatoire (0.1·randn).
- Entraînement : TBPTT par chunks contigus (`--chunk_size`, défaut 30) ; le x_t de l'acteur est
  stocké à chaque step dans le rollout et sert d'init (détaché) de chunk. Le critique n'est pas
  exécuté à la collecte : ses hidden viennent de la passe GAE séquentielle (carry `critic_carry`
  conservé entre rollouts, re-tiré aux `episode_starts`).
- Reset des hidden : uniquement en fin d'épisode (holder `LiquidHiddenState` partagé
  agent/collector) ; les tanks morts en cours d'épisode sont couverts par le masque `valids`.
- Hyperparams liquid : `--neuron_number 512`, `--unfolding_steps 6`, `--delta_t 1/30` ;
  activation de cellule = tanh ; output_size du liquid = neuron_number (têtes branchées dessus).
- Seul fichier existant touché hors ajouts validés : `agents/agent_cli.cpp` (enregistrement).
  `factory_set.cpp` (chargement côté jeu) reste sur l'Actor feed-forward — [[ppo-reward-redesign-2026-08]].
- 2026-09-17 (validé Samuel) : `LiquidRecurrent` + `LiquidCell` fusionnés en `LiquidCell`
  (`networks/recurrent/liquid_cell.{h,cpp}`, `CellModel` conservé). Clés de checkpoint
  `liquid.cell.*` → `liquid.*` ; notebook de conversion des anciens `actor.pt`/`critic.pt` :
  `arenai_agent/python/notebooks/convert_liquid_cell_fusion.ipynb` (nécessite torch, absent des
  requirements). L'ordre des paramètres est inchangé → les `*_optim.pt` n'ont pas besoin de
  conversion. Interop sérialisation : `OutputArchive` C++ ↔ `torch.jit.load/save` Python ;
  piège — `Module::load` C++ exige TOUS les sous-modules dans l'archive, même sans paramètres
  (SiLU, Sigmoid…), il faut répliquer la hiérarchie complète, pas juste les chemins des poids.
- 2026-09-20 (validé Samuel) : gradient accumulation via `--micro_batch_size` (en steps, comme
  `minibatch_size`). Motif : l'Impala CNN ne tient pas en VRAM à minibatch 1024, et réduire la
  minibatch à 256 faisait diverger ev → accumulation = même gradient/même nb d'updates, mémoire
  bornée par la micro-batch. Gate KL, PID α et métriques agrégés sur le minibatch complet ; loss
  micro pondérée par sa part de lignes valides ; clip + step uniques. Test d'équivalence
  `GradientAccumulationMatchesFullMinibatch`. Samuel a poussé les défauts à epochs 4,
  minibatch 2048, micro 256.
- 2026-10-20 (validé Samuel) : masque d'action invalide sur fire. `EnemyTank::get_can_fire()`
  (lecture pure, ≠ callback `can_fire` qui remet le cooldown à 0) → `core::State::can_fire` →
  `TorchState::fire_mask` [N,1]. Log-prob ET entropie de fire masqués (collecte + trainer, via
  `discrete_action_mask`) ; Hd = moyenne sur les actions ayant un effet. Exact car `tick` précède
  la lecture de l'état. Refus explicite : passer par `get_proprioception`. ppo vanilla masqué aussi (2026-10-20, demande Samuel pour comparer vanilla/liquid) ; son PID α_d mesure l'entropie fire sur les seules lignes non masquées.

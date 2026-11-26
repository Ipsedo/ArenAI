---
name: ppo-full-batch-update
description: "2026-10-27 — tous les PPO font 1 pas d'optimiseur par epoch (gradients accumulés sur le rollout) ; run 365 montre que lr 1e-4 est ~10× trop faible dans ce régime"
metadata:
  node_type: memory
  type: project
  originSessionId: c9569aa6-9b48-42a8-82d4-dbf8c4b2e11c
  modified: 2026-10-27T19:54:03.264Z
---

Depuis 2026-10-27 (commit ca4e4bcc + portage ppo_liquid_impala), les quatre agents PPO accumulent les gradients sur tout le rollout : **nb d'updates = epochs**. `minibatch_size` ne borne plus que la mémoire GPU. KL vérifié une fois par epoch (epoch 0 ≈ 0 par construction, ce qui dilue la métrique `kl`) ; `kl`/`skip` enregistrés par epoch.

**Why:** demandé par Samuel après [[ppo-train-364-diagnosis]] (minibatch 512 impossible en mémoire GPU avec impala).

Run 365 (impala, epochs 4, lr 1e-4) : clip = 0, kl ≈ 3e-4, ev ≈ −0.2, Hc qui remonte → politique et critique quasi immobiles. 4 epochs coûtent ~30 % de débit vs 364 (2 epochs × 112 minibatches).

**How to apply:** dans ce régime, monter le lr (≈1e-3, échelle √(14400/128) ≈ ×10) avant les epochs (gratuites vs coûteuses) ; viser kl 0.01-0.03 et clip 0.1-0.2.

**Remplacé le 2026-10-28** (après [[ppo-train-369-diagnosis]], validé par Samuel) : `minibatch_size` supprimé au profit de `actor_batch_size` / `actor_gpu_batch_size` / `critic_batch_size` / `critic_gpu_batch_size` (défauts communs 512 / 128 / 512 / 128, lignes ; liquid : convertis en chunks de 30). Un pas Adam par batch d'optimisation mélangé (`shuffled_batches` dans update_utils), gradients accumulés par blocs GPU. KL testée avant chaque pas ; dépassement → acteur arrêté pour le reste du rollout ; `skip` = part des pas prévus non faits (1 valeur/rollout), `kl` = 1 valeur/pas tenté.
